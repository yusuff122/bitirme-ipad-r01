"""Extract frozen E1 features, build a normal-only memory bank, and calibrate threshold."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.video_loader import R01VideoClipDataset, collate_video_clips
from src.models.video_swin_baseline import MODEL_NAME, WEIGHTS_NAME, VideoSwinFeatureExtractor
from src.training.train_frame_baseline import validate_manifest
from src.utils.device import get_device
from src.utils.seed import set_seed


def _sequence_names(records):
    return sorted(str(x) for x in records.sequence.unique())


def cache_metadata(records, cfg, split):
    return {
        'split': split,
        'model_name': cfg['model_name'], 'weights_name': cfg['weights_name'],
        'clip_length': int(cfg['clip_length']), 'stride': int(cfg['stride']),
        'input_size': list(cfg['input_size']),
        'sequence_list': _sequence_names(records),
        'manifest_rows_sha256': hashlib.sha256(
            records[['sequence', 'frame_index', 'frame_name', 'path']].to_csv(index=False).encode()
        ).hexdigest(),
    }


def extract_embeddings(records, model, transform, cfg, device, split, cache_dir, use_cache=True):
    metadata = cache_metadata(records, cfg, split)
    vector_path = cache_dir / f'{split}_embeddings.pt'
    meta_path = cache_dir / f'{split}_embeddings.json'
    if use_cache and vector_path.exists() and meta_path.exists():
        old_metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if old_metadata != metadata:
            raise ValueError(f'{split} cache metadata mismatch at {cache_dir}; refusing stale embeddings')
        embeddings = torch.load(vector_path, map_location='cpu', weights_only=True).float()
        if embeddings.ndim != 2 or embeddings.shape[1] != model.embedding_dim:
            raise ValueError(f'Invalid {split} cache shape: {tuple(embeddings.shape)}')
        return embeddings, json.loads((cache_dir / f'{split}_clips.json').read_text(encoding='utf-8'))

    dataset = R01VideoClipDataset(records, transform, cfg['clip_length'], cfg['stride'])
    if not dataset:
        raise ValueError(f'{split} contains no full clips')
    loader = DataLoader(dataset, batch_size=int(cfg['batch_size']), shuffle=False,
                        num_workers=int(cfg.get('num_workers', 0)), collate_fn=collate_video_clips)
    vectors, clip_records = [], []
    with torch.inference_mode():
        for videos, metadata_batch in tqdm(loader, desc=f'{split} clips'):
            videos = videos.to(device, non_blocking=True)
            features = model(videos).float()
            if cfg['normalize_embeddings']:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            norms = torch.linalg.vector_norm(features, dim=1)
            if not torch.isfinite(features).all() or not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                raise RuntimeError(f'{split} embedding L2 normalization check failed; first norms={norms[:10].tolist()}')
            vectors.append(features.cpu())
            clip_records.extend(metadata_batch)
    embeddings = torch.cat(vectors).to(torch.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, vector_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (cache_dir / f'{split}_clips.json').write_text(json.dumps(clip_records), encoding='utf-8')
    return embeddings, clip_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/video_swin.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--cache-dir', default='models/E1_video_swin16_cache')
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8')) or {}
    cfg.update({
        'manifest': args.manifest or cfg.get('manifest', cfg.get('splits', 'data/processed/r01_manifest.csv')),
        'model_name': cfg.get('model_name', MODEL_NAME),
        'pretrained': cfg.get('pretrained', True),
        'weights_name': cfg.get('weights_name', WEIGHTS_NAME),
        'clip_length': int(cfg.get('clip_length', 16)), 'stride': int(cfg.get('stride', 4)),
        'batch_size': args.batch_size or int(cfg.get('batch_size', 4)),
        'input_size': cfg.get('input_size', [224, 224]),
        'normalize_embeddings': bool(cfg.get('normalize_embeddings', True)),
        'knn_k': int(cfg.get('knn_k', 5)), 'num_workers': int(cfg.get('num_workers', 0)),
        'threshold_method': 'quantile', 'threshold_quantile': float(cfg.get('threshold_quantile', 0.995)),
    })
    if cfg['model_name'] != MODEL_NAME or not cfg['pretrained'] or cfg['weights_name'] != WEIGHTS_NAME:
        raise ValueError('E1 requires the official pretrained torchvision Video Swin Tiny weights')
    if not cfg['normalize_embeddings']:
        raise ValueError('E1 requires L2-normalized embeddings')
    if cfg['knn_k'] != 5:
        raise ValueError('E1 baseline is configured for k=5')
    if list(cfg['input_size']) != [224, 224]:
        raise ValueError('Official KINETICS400_V1 transform produces 224x224 inputs; input_size must be [224, 224]')
    set_seed(int(cfg.get('seed', 42)))
    manifest = pd.read_csv(cfg['manifest'])
    train, val, _ = validate_manifest(manifest)
    device = get_device()
    model = VideoSwinFeatureExtractor(cfg['model_name'], cfg['pretrained'], cfg['weights_name']).to(device).eval()
    cfg['weights_name'] = model.weights_name
    transform = model.make_transform()
    cache_dir = Path(args.cache_dir)
    train_features, train_clips = extract_embeddings(train, model, transform, cfg, device, 'train', cache_dir, not args.no_cache)
    val_features, val_clips = extract_embeddings(val, model, transform, cfg, device, 'validation', cache_dir, not args.no_cache)
    from src.models.frame_baseline import knn_scores
    val_scores = knn_scores(val_features, train_features, cfg['knn_k']).numpy().astype(float)
    threshold = float(np.quantile(val_scores, cfg['threshold_quantile']))
    now = datetime.now().astimezone()
    run_dir = Path(args.output_dir) / f"E1_video_swin16_{now.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'embeddings').mkdir()
    torch.save(train_features.float(), run_dir / 'embeddings/train_embeddings.pt')
    torch.save(val_features.float(), run_dir / 'embeddings/validation_embeddings.pt')
    (run_dir / 'embeddings/train_clips.json').write_text(json.dumps(train_clips), encoding='utf-8')
    (run_dir / 'embeddings/validation_clips.json').write_text(json.dumps(val_clips), encoding='utf-8')
    torch.save(train_features.float(), run_dir / 'memory_bank.pt')
    threshold_info = {
        'method': 'quantile', 'quantile': cfg['threshold_quantile'], 'threshold': threshold,
        'num_validation_clips': int(len(val_scores)), 'mean': float(val_scores.mean()),
        'std': float(val_scores.std()), 'min': float(val_scores.min()), 'max': float(val_scores.max()),
        'p95': float(np.quantile(val_scores, .95)), 'p99': float(np.quantile(val_scores, .99)),
        'p995': float(np.quantile(val_scores, .995)), 'test_used_for_threshold': False,
    }
    test_rows = manifest[manifest.split == 'test']
    config_out = {
        **cfg, 'weights_name': model.weights_name, 'embedding_dim': model.embedding_dim,
        'device': str(device), 'timestamp': now.isoformat(), 'train_sequences': _sequence_names(train),
        'validation_sequences': _sequence_names(val), 'test_sequences': _sequence_names(test_rows),
        'num_train_clips': len(train_clips), 'num_validation_clips': len(val_clips),
        'num_test_frames': len(test_rows), 'cache_dir': str(cache_dir.resolve()),
        'train_cache_metadata': cache_metadata(train, cfg, 'train'),
        'validation_cache_metadata': cache_metadata(val, cfg, 'validation'),
    }
    (run_dir / 'config.json').write_text(json.dumps(config_out, indent=2), encoding='utf-8')
    (run_dir / 'threshold.json').write_text(json.dumps(threshold_info, indent=2), encoding='utf-8')
    (run_dir / 'run_summary.md').write_text(
        f"# E1 Video Swin run\n\n- Model: `{model.model_name}` ({model.weights_name})\n"
        f"- Frozen feature extraction; classification head removed; no fine-tuning.\n"
        f"- Input: B,C,T,H,W with T={cfg['clip_length']}; official deterministic weights transform.\n"
        f"- Embedding dimension: {model.embedding_dim}; L2 normalized; k={cfg['knn_k']} Euclidean.\n"
        f"- Train clips: {len(train_clips)}; validation clips: {len(val_clips)}.\n"
        f"- Threshold: normal validation {cfg['threshold_quantile']:.3%} quantile = {threshold:.8f}.\n"
        f"- Test frames/labels were not used for feature extraction or threshold calibration.\n",
        encoding='utf-8')
    print(json.dumps({'run_dir': str(run_dir), 'device': str(device), 'model_name': model.model_name,
        'weights_name': model.weights_name, 'input_batch_shape': [cfg['batch_size'], 3, cfg['clip_length'], *cfg['input_size']],
        'embedding_dim': model.embedding_dim, 'train_clip_count': len(train_clips),
        'validation_clip_count': len(val_clips), 'memory_bank_shape': list(train_features.shape),
        'embedding_norm_first10': torch.linalg.vector_norm(train_features, dim=1)[:10].tolist(),
        'threshold': threshold, 'run_dir': str(run_dir)}, indent=2))


if __name__ == '__main__':
    main()
