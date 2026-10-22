"""Evaluate E1 clip scores on all official test frames with exact key alignment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.video_loader import R01VideoClipDataset, collate_video_clips, map_clip_scores_to_frames
from src.evaluation.evaluate_frame import exact_test_alignment
from src.evaluation.metrics import calculate_metrics
from src.models.frame_baseline import knn_scores
from src.models.video_swin_baseline import VideoSwinFeatureExtractor
from src.training.train_video_baseline import cache_metadata
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--cache-dir')
    parser.add_argument('--reports-dir', default='outputs/reports')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    manifest_path = args.manifest or cfg['manifest']
    manifest = pd.read_csv(manifest_path)
    test = manifest.loc[manifest.split == 'test'].copy()
    if test.empty or test.label.isna().any() or not set(test.label.astype(int).unique()).issubset({0, 1}):
        raise ValueError('Test manifest requires non-empty official binary labels')
    if len(test) != int(cfg['num_test_frames']):
        raise ValueError(f"Test frame count changed: run={cfg['num_test_frames']}, current={len(test)}")
    device = get_device()
    model = VideoSwinFeatureExtractor(cfg['model_name'], cfg['pretrained'], cfg['weights_name']).to(device).eval()
    if model.weights_name != cfg['weights_name']:
        raise ValueError(f"Feature weight mismatch: run={cfg['weights_name']}, loaded={model.weights_name}")
    if model.embedding_dim != int(cfg['embedding_dim']):
        raise ValueError('Embedding dimension differs from saved run config')
    bank = torch.load(run_dir / 'memory_bank.pt', map_location='cpu', weights_only=True).float()
    if bank.ndim != 2 or bank.shape[1] != model.embedding_dim or len(bank) < cfg['knn_k']:
        raise ValueError(f'Invalid memory bank shape: {tuple(bank.shape)}')

    embedding_dir = run_dir / 'embeddings'
    embedding_dir.mkdir(exist_ok=True)
    cache_dir = Path(args.cache_dir or cfg.get('cache_dir', 'models/E1_video_swin16_cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    vector_path, meta_path = cache_dir / 'test_embeddings.pt', cache_dir / 'test_embeddings.json'
    expected_metadata = cache_metadata(test, cfg, 'test')
    clip_path = cache_dir / 'test_clips.json'
    if not args.no_cache and vector_path.exists() and meta_path.exists() and clip_path.exists():
        saved_metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if saved_metadata != expected_metadata:
            raise ValueError('Saved test cache metadata mismatch; refusing stale embeddings')
        test_features = torch.load(vector_path, map_location='cpu', weights_only=True).float()
        clip_records = json.loads(clip_path.read_text(encoding='utf-8'))
    else:
        dataset = R01VideoClipDataset(test, model.make_transform(), cfg['clip_length'], cfg['stride'])
        loader = DataLoader(dataset, batch_size=args.batch_size or cfg['batch_size'], shuffle=False,
                            num_workers=cfg.get('num_workers', 0), collate_fn=collate_video_clips)
        feature_batches, clip_records = [], []
        with torch.inference_mode():
            for videos, metadata_batch in loader:
                features = model(videos.to(device, non_blocking=True)).float()
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                norms = torch.linalg.vector_norm(features, dim=1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                    raise RuntimeError(f'Test embedding norm check failed: {norms[:10].tolist()}')
                feature_batches.append(features.cpu())
                clip_records.extend(metadata_batch)
        if not feature_batches:
            raise ValueError('No test clips were created')
        test_features = torch.cat(feature_batches).float()
        if not args.no_cache:
            torch.save(test_features, vector_path)
            meta_path.write_text(json.dumps(expected_metadata, indent=2), encoding='utf-8')
            clip_path.write_text(json.dumps(clip_records), encoding='utf-8')
    if test_features.ndim != 2 or test_features.shape[1] != model.embedding_dim or len(test_features) != len(clip_records):
        raise ValueError('Test embedding cache shape/count mismatch')
    if not torch.allclose(torch.linalg.vector_norm(test_features, dim=1), torch.ones(len(test_features)), atol=1e-4, rtol=1e-4):
        raise ValueError('Cached test embeddings are not L2 normalized')
    torch.save(test_features.float(), embedding_dir / 'test_clip_embeddings.pt')
    (embedding_dir / 'test_clips.json').write_text(json.dumps(clip_records), encoding='utf-8')
    (embedding_dir / 'test_cache_metadata.json').write_text(json.dumps(expected_metadata, indent=2), encoding='utf-8')
    clip_scores = knn_scores(test_features, bank, int(cfg['knn_k'])).numpy()
    frame_rows = map_clip_scores_to_frames(clip_records, clip_scores, test)
    frame_scores = pd.DataFrame(frame_rows)
    threshold_info = json.loads((run_dir / 'threshold.json').read_text(encoding='utf-8'))
    threshold = float(threshold_info['threshold'])
    predictions = frame_scores[['split', 'sequence', 'frame_name', 'frame_index', 'label', 'anomaly_score', 'num_covering_clips']].copy()
    predictions = predictions.rename(columns={'label': 'ground_truth'})
    predictions['threshold'] = threshold
    predictions['prediction'] = (predictions.anomaly_score >= threshold).astype(int)
    # Compare with labels from the source manifest on exact keys before saving.
    expected = test[['split', 'sequence', 'frame_name', 'frame_index', 'label']].rename(columns={'label': 'ground_truth'})
    assert_unique = predictions.duplicated(['split', 'sequence', 'frame_name']).any()
    if assert_unique:
        raise ValueError('Duplicate prediction keys')
    aligned = exact_test_alignment(predictions, expected)
    if len(aligned) != 3685:
        raise ValueError(f'Expected 3685 official test-frame scores, got {len(aligned)}')
    aligned = aligned[['sequence', 'frame_name', 'frame_index', 'ground_truth', 'anomaly_score', 'threshold', 'prediction', 'num_covering_clips']]
    aligned.to_csv(run_dir / 'predictions.csv', index=False)
    metrics = calculate_metrics(aligned.ground_truth, aligned.anomaly_score, threshold)
    metrics.update({'threshold': threshold, 'threshold_method': threshold_info['method'],
                    'num_test_frames': len(aligned), 'num_test_clips': len(test_features),
                    'test_used_for_threshold': False, 'weights_name': model.weights_name})
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    figures = run_dir / 'figures'
    figures.mkdir(exist_ok=True)
    matrix = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], xlabel='Predicted', ylabel='Ground truth', title='E1 Confusion Matrix')
    for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha='center', va='center')
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figures / 'confusion_matrix.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, name, color in [(0, 'Normal test', 'tab:blue'), (1, 'Anomaly test', 'tab:red')]:
        values = aligned.loc[aligned.ground_truth == label, 'anomaly_score']
        ax.hist(values, bins=40, alpha=.55, density=True, label=name, color=color)
    ax.axvline(threshold, color='black', linestyle='--', label=f'Validation threshold={threshold:.4g}')
    ax.set(xlabel='Frame anomaly score (overlap-mean clip k-NN distance)', ylabel='Density', title='E1 Test Score Distributions'); ax.legend(); fig.tight_layout(); fig.savefig(figures / 'anomaly_score_distribution.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    for seq in ['testing/01', 'testing/08', 'testing/15']:
        sample = aligned[aligned.sequence == seq]
        if not sample.empty:
            ax.plot(sample.frame_index, sample.anomaly_score, label=seq, linewidth=1)
    ax.axhline(threshold, color='black', linestyle='--', label='Validation threshold')
    ax.set(xlabel='Frame index', ylabel='Anomaly score', title='E1 Example Test Timelines'); ax.legend(); fig.tight_layout(); fig.savefig(figures / 'anomaly_timeline_examples.png', dpi=160); plt.close(fig)

    report_dir = Path(args.reports_dir); report_dir.mkdir(parents=True, exist_ok=True)
    e0 = {'auroc': .7282238589, 'aupr': .4883859957, 'f1': .5856975381,
          'precision': .4148123547, 'recall': .9960127592, 'accuracy': .5204884668}
    keys = ['auroc', 'aupr', 'f1', 'precision', 'recall', 'accuracy']
    comparison = {key: {'E0': e0[key], 'E1': metrics[key]} for key in keys}
    (report_dir / 'E0_vs_E1.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    lines = ['| Metric | E0 | E1 |', '|---|---:|---:|']
    lines += [f"| {key.upper() if key in ('auroc','aupr') else key.title()} | {e0[key]:.6f} | {metrics[key]:.6f} |" for key in keys]
    (report_dir / 'E0_vs_E1.md').write_text('# E0 vs E1\n\n' + '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'metrics': metrics, 'test_clip_count': len(test_features),
                      'test_embedding_shape': list(test_features.shape), 'embedding_norm_first10': torch.linalg.vector_norm(test_features, dim=1)[:10].tolist()}, indent=2))


if __name__ == '__main__':
    main()
