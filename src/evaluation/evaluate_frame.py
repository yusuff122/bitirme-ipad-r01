"""Infer the official R01 test set and evaluate with exact frame-key alignment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.ipad_r01_dataset import R01FrameDataset
from src.evaluation.metrics import calculate_metrics
from src.utils.device import get_device


KEYS = ['split', 'sequence', 'frame_name']


def assert_unique_keys(df, name):
    if df.duplicated(KEYS).any():
        examples = df.loc[df.duplicated(KEYS, keep=False), KEYS].head().to_dict('records')
        raise ValueError(f'{name} has duplicate frame keys {KEYS}: {examples}')


def exact_test_alignment(predictions, expected):
    assert_unique_keys(predictions, 'Predictions')
    assert_unique_keys(expected, 'Expected test manifest')
    expected_for_merge = expected.rename(columns={'ground_truth': 'expected_ground_truth'})
    merged = expected_for_merge.merge(predictions, on=KEYS, how='outer', suffixes=('_expected', '_prediction'), indicator=True, validate='one_to_one')
    bad = merged[merged['_merge'] != 'both']
    if len(bad):
        raise ValueError(f'Prediction/ground-truth key mismatch: {len(bad)} missing or extra rows; examples: {bad[KEYS + ["_merge"]].head().to_dict("records")}')
    if len(merged) != len(expected) or len(merged) != len(predictions):
        raise ValueError(f'Alignment count mismatch: expected={len(expected)}, predictions={len(predictions)}, merged={len(merged)}')
    if not (merged['split'] == 'test').all():
        raise ValueError('Exact merge unexpectedly contains non-test rows')
    if 'frame_index_expected' in merged and 'frame_index_prediction' in merged:
        if not (merged['frame_index_expected'].astype(int) == merged['frame_index_prediction'].astype(int)).all():
            raise ValueError('frame_index differs between prediction and expected manifest for an exact key')
        merged['frame_index'] = merged['frame_index_expected'].astype(int)
        merged = merged.drop(columns=['frame_index_expected', 'frame_index_prediction'])
    if 'expected_ground_truth' in merged and not (merged['expected_ground_truth'].astype(int) == merged['ground_truth'].astype(int)).all():
        raise ValueError('Prediction ground_truth differs from official manifest label after exact merge')
    return merged.drop(columns=['_merge'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-test-frames', type=int, help='Smoke-test override; does not alter saved run config.')
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    manifest_path = args.manifest or run_cfg['manifest']
    manifest = pd.read_csv(manifest_path)
    required = {'path', 'sequence', 'frame_name', 'frame_index', 'split', 'label'}
    if not required.issubset(manifest.columns):
        raise ValueError(f'Manifest missing columns: {sorted(required-set(manifest.columns))}')
    test = manifest[manifest.split == 'test'].copy()
    if test.empty or test.label.isna().any() or not set(test.label.unique()).issubset({0, 1}):
        raise ValueError('Test manifest needs non-empty binary ground_truth labels')
    assert_unique_keys(test, 'Expected test manifest')
    if args.max_test_frames:
        test = test.groupby('sequence', sort=False, group_keys=False).head(args.max_test_frames).copy()

    device = get_device()
    from torchvision import transforms
    from src.models.frame_baseline import ResNet18FeatureExtractor, knn_scores
    model = ResNet18FeatureExtractor(pretrained=False).to(device).eval()
    weights = torch.load(run_dir / 'memory_bank.pt', map_location='cpu', weights_only=True).float()
    if weights.ndim != 2 or weights.shape[1] != model.embedding_dim:
        raise ValueError(f'Invalid memory bank shape: {tuple(weights.shape)}; expected (N,{model.embedding_dim})')
    transform = transforms.Compose([
        transforms.Resize((int(run_cfg['image_size']), int(run_cfg['image_size']))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = R01FrameDataset(test, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size or int(run_cfg['batch_size']), shuffle=False, num_workers=0)
    feature_batches = []
    with torch.inference_mode():
        for images, _ in loader:
            feat = model(images.to(device)).float()
            if run_cfg['normalize_embeddings']:
                feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            feature_batches.append(feat.cpu())
    if not feature_batches:
        raise ValueError('No test images were loaded')
    features = torch.cat(feature_batches)
    scores = knn_scores(features, weights, int(run_cfg['knn_k'])).numpy().astype(float)
    threshold_info = json.loads((run_dir / 'threshold.json').read_text(encoding='utf-8'))
    threshold = float(threshold_info['threshold'])
    predictions = test[['split', 'sequence', 'frame_name', 'frame_index']].reset_index(drop=True).copy()
    predictions['ground_truth'] = test.label.astype(int).to_numpy()
    predictions['anomaly_score'] = scores
    predictions['threshold'] = threshold
    predictions['prediction'] = (scores >= threshold).astype(int)
    aligned = exact_test_alignment(predictions, test.rename(columns={'label': 'ground_truth'}))
    aligned = aligned[['sequence', 'frame_name', 'frame_index', 'ground_truth', 'anomaly_score', 'threshold', 'prediction']]
    aligned.to_csv(run_dir / 'predictions.csv', index=False)

    metrics = calculate_metrics(aligned.ground_truth, aligned.anomaly_score, threshold)
    tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
    metrics['fpr'] = float(fp / (fp + tn)) if fp + tn else 0.0
    metrics['fnr'] = float(fn / (fn + tp)) if fn + tp else 0.0
    metrics.update({'threshold': threshold, 'threshold_method': threshold_info['method'], 'num_test_frames': int(len(aligned)), 'test_used_for_threshold': False})
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    figures = Path('outputs/figures')
    figures.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], xlabel='Predicted', ylabel='Ground truth', title='E0 Confusion Matrix')
    for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha='center', va='center')
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figures / f'{run_dir.name}_confusion_matrix.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, name, color in [(0, 'Normal test', 'tab:blue'), (1, 'Anomaly test', 'tab:red')]:
        values = aligned.loc[aligned.ground_truth == label, 'anomaly_score']
        if len(values): ax.hist(values, bins=40, alpha=.55, density=True, label=name, color=color)
    ax.axvline(threshold, color='black', linestyle='--', label=f'Validation threshold={threshold:.4g}')
    ax.set(xlabel='Anomaly score (mean k-NN Euclidean distance)', ylabel='Density', title='E0 Test Score Distributions')
    ax.legend(); fig.tight_layout(); fig.savefig(figures / f'{run_dir.name}_score_distribution.png', dpi=160); plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f'Predictions: {run_dir / "predictions.csv"}')


if __name__ == '__main__':
    main()
