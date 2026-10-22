"""One-sequence-per-split E1 pipeline smoke check; never starts a full run."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data.video_loader import map_clip_scores_to_frames
from src.evaluation.evaluate_frame import exact_test_alignment
from src.models.frame_baseline import knn_scores
from src.models.video_swin_baseline import MODEL_NAME, WEIGHTS_NAME, VideoSwinFeatureExtractor
from src.training.train_video_baseline import extract_embeddings
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/video_swin.yaml')
    parser.add_argument('--manifest', default='data/processed/r01_manifest.csv')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    manifest = pd.read_csv(args.manifest)
    train_seq = sorted(manifest.loc[manifest.split == 'train', 'sequence'].unique())[0]
    val_seq = sorted(manifest.loc[manifest.split == 'val', 'sequence'].unique())[0]
    test_seq = sorted(manifest.loc[manifest.split == 'test', 'sequence'].unique())[0]
    subsets = {
        'train': manifest[(manifest.split == 'train') & (manifest.sequence == train_seq)].copy(),
        'validation': manifest[(manifest.split == 'val') & (manifest.sequence == val_seq)].copy(),
        'test': manifest[(manifest.split == 'test') & (manifest.sequence == test_seq)].copy(),
    }
    cfg.update({'model_name': cfg.get('model_name', MODEL_NAME), 'weights_name': cfg.get('weights_name', WEIGHTS_NAME),
                'pretrained': True, 'clip_length': 16, 'stride': 4,
                'input_size': cfg.get('input_size', [224, 224]), 'batch_size': args.batch_size,
                'num_workers': 0, 'normalize_embeddings': True})
    if list(cfg['input_size']) != [224, 224]:
        raise ValueError('The official KINETICS400_V1 transform outputs 224x224 clips')
    device = get_device()
    if device.type == 'cpu':
        raise RuntimeError('Smoke requested GPU inference; this runtime has neither CUDA nor MPS')
    model = VideoSwinFeatureExtractor(cfg['model_name'], True, cfg['weights_name']).to(device).eval()
    cfg['weights_name'] = model.weights_name
    transform = model.make_transform()
    extracted = {}
    for split, records in subsets.items():
        embeddings, clips = extract_embeddings(records, model, transform, cfg, device, split,
                                               Path('/tmp/e1_video_smoke_cache'), use_cache=False)
        extracted[split] = (embeddings, clips)
        print(f'{split}: sequences={records.sequence.nunique()} clips={len(clips)} embeddings={tuple(embeddings.shape)}')
        print(f'{split}: first video tensor shape=(3,{cfg["clip_length"]},{cfg["input_size"][0]},{cfg["input_size"][1]})')
        print(f'{split}: embedding norm first10={torch.linalg.vector_norm(embeddings, dim=1)[:10].tolist()}')
    train_features, _ = extracted['train']
    val_features, val_clips = extracted['validation']
    test_features, test_clips = extracted['test']
    bank = train_features
    if len(bank) < 5:
        raise RuntimeError(f'Insufficient train clips in smoke subset: {len(bank)}')
    val_scores = knn_scores(val_features, bank, 5)
    threshold = float(torch.quantile(val_scores, .995))
    test_scores = knn_scores(test_features, bank, 5).numpy()
    mapped = pd.DataFrame(map_clip_scores_to_frames(test_clips, test_scores, subsets['test']))
    predictions = mapped[['split', 'sequence', 'frame_name', 'frame_index', 'label', 'anomaly_score']].rename(columns={'label': 'ground_truth'})
    predictions['threshold'] = threshold
    predictions['prediction'] = (predictions.anomaly_score >= threshold).astype(int)
    expected = subsets['test'][['split', 'sequence', 'frame_name', 'frame_index', 'label']].rename(columns={'label': 'ground_truth'})
    aligned = exact_test_alignment(predictions, expected)
    print(json.dumps({'device': str(device), 'model_name': model.model_name, 'weights_name': model.weights_name,
                      'embedding_dim': model.embedding_dim, 'input_shape': [args.batch_size, 3, 16, 224, 224],
                      'memory_bank_shape': list(bank.shape), 'validation_clip_count': len(val_clips),
                      'test_clip_count': len(test_clips), 'smoke_test_frames_aligned': len(aligned),
                      'threshold': threshold, 'knn_example': float(test_scores[0]),
                      'status': 'PASS; one sequence per split only; full E1 not run'}, indent=2))


if __name__ == '__main__':
    main()
