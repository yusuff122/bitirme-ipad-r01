"""Fit the E0 normal-only memory bank and validation-only threshold."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.ipad_r01_dataset import R01FrameDataset
from src.utils.device import get_device
from src.utils.seed import set_seed


def _load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _embeddings(records, model, transform, device, batch_size, normalize):
    dataset = R01FrameDataset(records, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc='Embedding frames', leave=False):
            features = model(images.to(device)).float()
            if normalize:
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            batches.append(features.cpu())
    if not batches:
        raise ValueError('Cannot extract embeddings from an empty partition')
    return torch.cat(batches).to(torch.float32)


def validate_manifest(manifest):
    required = {'path', 'sequence', 'frame_name', 'frame_index', 'split', 'label'}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f'Manifest is missing columns: {sorted(missing)}')
    if manifest.duplicated(['split', 'sequence', 'frame_name']).any():
        raise ValueError('Manifest contains duplicate split+sequence+frame_name keys')
    if manifest[['path', 'sequence', 'frame_name', 'frame_index', 'split', 'label']].isna().any().any():
        # Training/validation labels must be explicit normal labels; test labels are mandatory too.
        raise ValueError('Manifest required columns contain null values')
    allowed = {'train', 'val', 'test'}
    if not set(manifest.split.unique()).issubset(allowed):
        raise ValueError(f'Unexpected split values: {set(manifest.split.unique()) - allowed}')
    train = manifest[manifest.split == 'train']
    val = manifest[manifest.split == 'val']
    test = manifest[manifest.split == 'test']
    if train.empty or val.empty or test.empty:
        raise ValueError('Train, validation, and test splits must all be non-empty')
    if train.label.isna().any() or val.label.isna().any() or not set(train.label.unique()).issubset({0}) or not set(val.label.unique()).issubset({0}):
        raise ValueError('All training and validation frames must be normal (label 0)')
    partitions = [set(x.sequence.unique()) for x in (train, val, test)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Sequence leakage: a sequence appears in multiple splits')
    # Drop test labels from the training path; final evaluation reads them separately.
    return train, val, test.drop(columns=['label'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/frame_baseline.yaml')
    parser.add_argument('--manifest')
    parser.add_argument('--root')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-train-frames', type=int, help='Smoke-test override; does not alter config.')
    parser.add_argument('--max-val-frames', type=int, help='Smoke-test override; does not alter config.')
    args = parser.parse_args()
    cfg = _load_config(args.config)
    cfg['manifest'] = args.manifest or cfg.get('manifest', 'data/processed/r01_manifest.csv')
    cfg['dataset_root'] = args.root or cfg.get('dataset_root', 'data/r01')
    cfg['batch_size'] = args.batch_size or cfg.get('batch_size', 32)
    cfg.setdefault('seed', 42)
    cfg.setdefault('image_size', 256)
    cfg.setdefault('pretrained', True)
    cfg.setdefault('score_method', 'knn')
    cfg.setdefault('knn_k', 5)
    cfg.setdefault('normalize_embeddings', True)
    cfg.setdefault('threshold_method', 'quantile')
    cfg.setdefault('threshold_quantile', 0.995)
    if cfg['score_method'] != 'knn':
        raise ValueError('Only score_method=knn is implemented')
    if cfg['threshold_method'] != 'quantile':
        raise ValueError('Only threshold_method=quantile is implemented')
    if not 0 < float(cfg['threshold_quantile']) < 1:
        raise ValueError('threshold_quantile must be between 0 and 1')

    manifest = pd.read_csv(cfg['manifest'])
    train, val, _ = validate_manifest(manifest)
    if args.max_train_frames:
        train = train.groupby('sequence', sort=False, group_keys=False).head(args.max_train_frames)
    if args.max_val_frames:
        val = val.groupby('sequence', sort=False, group_keys=False).head(args.max_val_frames)
    set_seed(int(cfg['seed']))
    device = get_device()
    from torchvision import transforms
    from src.models.frame_baseline import ResNet18FeatureExtractor, knn_scores
    transform = transforms.Compose([
        transforms.Resize((int(cfg['image_size']), int(cfg['image_size']))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    model = ResNet18FeatureExtractor(pretrained=bool(cfg['pretrained'])).to(device).eval()
    memory = _embeddings(train, model, transform, device, int(cfg['batch_size']), bool(cfg['normalize_embeddings']))
    max_samples = cfg.get('max_memory_samples')
    if max_samples and memory.shape[0] > int(max_samples):
        generator = torch.Generator().manual_seed(int(cfg['seed']))
        selected = torch.randperm(memory.shape[0], generator=generator)[:int(max_samples)]
        memory = memory[selected]
    val_features = _embeddings(val, model, transform, device, int(cfg['batch_size']), bool(cfg['normalize_embeddings']))
    val_scores = knn_scores(val_features, memory, int(cfg['knn_k'])).numpy().astype(float)
    q = float(cfg['threshold_quantile'])
    threshold = float(np.quantile(val_scores, q))
    now = datetime.now().astimezone()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    run_dir = Path('models') / f'E0_frame_resnet18_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    val_summary = {
        'method': cfg['threshold_method'], 'quantile': q, 'threshold': threshold,
        'num_validation_frames': int(len(val_scores)),
        'validation_score_mean': float(val_scores.mean()),
        'validation_score_std': float(val_scores.std()),
        'validation_score_min': float(val_scores.min()),
        'validation_score_max': float(val_scores.max()),
        'validation_score_p95': float(np.quantile(val_scores, .95)),
        'validation_score_p99': float(np.quantile(val_scores, .99)),
        'validation_score_p995': float(np.quantile(val_scores, .995)),
        'test_used_for_threshold': False,
    }
    train_sequences = sorted(train.sequence.unique().tolist())
    val_sequences = sorted(val.sequence.unique().tolist())
    test_sequences = sorted(manifest.loc[manifest.split == 'test', 'sequence'].unique().tolist())
    config_out = {
        **cfg, 'seed': int(cfg['seed']), 'backbone': 'resnet18',
        'weights': 'ResNet18_Weights.DEFAULT' if cfg['pretrained'] else None,
        'weights_name': model.weights_name,
        'embedding_dim': int(model.embedding_dim), 'device': str(device), 'timestamp': now.isoformat(),
        'train_sequences': train_sequences, 'validation_sequences': val_sequences,
        'test_sequences': test_sequences, 'num_train_frames': int(len(train)),
        'num_validation_frames': int(len(val)),
        'num_test_frames': int((manifest.split == 'test').sum()),
        'memory_bank_samples': int(memory.shape[0]),
        'smoke_test_limits': {'max_train_frames': args.max_train_frames, 'max_val_frames': args.max_val_frames},
    }
    torch.save(memory.to(torch.float32), run_dir / 'memory_bank.pt')
    (run_dir / 'config.json').write_text(json.dumps(config_out, indent=2), encoding='utf-8')
    (run_dir / 'threshold.json').write_text(json.dumps(val_summary, indent=2), encoding='utf-8')
    summary = f"""# E0 run summary

- Timestamp: {now.isoformat()}
- Device: {device}
- Backbone: frozen pretrained ResNet18 ({config_out['weights']})
- Embedding dimension: {model.embedding_dim}
- Dataset root: `{cfg['dataset_root']}`
- Training: {len(train_sequences)} normal sequences, {len(train)} frames
- Validation: {len(val_sequences)} held-out normal sequences, {len(val)} frames
- Test sequences recorded for evaluation: {len(test_sequences)} ({config_out['num_test_frames']} frames); test pixels/labels were not read by training.
- Score: Euclidean k-NN mean distance, k={cfg['knn_k']}, L2 normalized={cfg['normalize_embeddings']}
- Threshold: validation normal score {q:.3%} quantile = {threshold:.8f}
- Memory bank embeddings: {len(memory)} float32 vectors

The training command did not load test frames or test labels. Run `python -m src.evaluation.evaluate_frame --run-dir {run_dir}` for final evaluation.
"""
    (run_dir / 'run_summary.md').write_text(summary, encoding='utf-8')
    print(f'Run saved: {run_dir}')
    print(json.dumps({'train_sequences': len(train_sequences), 'train_frames': len(train), 'validation_sequences': len(val_sequences), 'validation_frames': len(val), 'threshold': threshold, 'device': str(device)}, indent=2))


if __name__ == '__main__':
    main()
