"""Bounded E0 integration smoke test; never starts the full baseline."""
import argparse
import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.device import get_device
from src.training.train_frame_baseline import validate_manifest
from src.evaluation.evaluate_frame import exact_test_alignment
from src.evaluation.metrics import calculate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/processed/r01_manifest.csv')
    parser.add_argument('--config', default='configs/frame_baseline.yaml')
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    train, val, test = validate_manifest(manifest)
    assert (train.label == 0).all() and (val.label == 0).all()
    assert not (set(train.sequence) & set(val.sequence))
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg['dataset_root'])
    # Verify label-to-folder relationship and exact lengths directly from files.
    labeled_test = manifest[manifest.split == 'test'].copy()
    for sequence, rows in labeled_test.groupby('sequence'):
        seq_id = sequence.rsplit('/', 1)[-1]
        label_path = root / 'test_label' / f'{int(seq_id):03d}.npy'
        frame_dir = root / 'testing' / 'frames' / seq_id
        labels = np.load(label_path, allow_pickle=False).reshape(-1)
        frame_paths = sorted(frame_dir.glob('*.jpg'), key=lambda p: int(p.stem))
        assert len(labels) == len(frame_paths) == len(rows), (sequence, len(labels), len(frame_paths), len(rows))
        by_name = dict(zip([p.name for p in frame_paths], labels.astype(int)))
        for row in rows.itertuples():
            assert int(row.label) == int(by_name[row.frame_name]), (sequence, row.frame_name)
    print(f'Dataset checks OK: train={train.sequence.nunique()} sequences/{len(train)} frames; val={val.sequence.nunique()} sequences/{len(val)} frames; test={test.sequence.nunique()} sequences/{len(test)} frames')

    # Exercise exact merge and metrics without reading/inferencing the full test set.
    small = labeled_test.groupby('sequence', sort=False, group_keys=False).head(2).copy()
    dummy = small[['split', 'sequence', 'frame_name', 'frame_index']].copy()
    dummy['ground_truth'] = small.label.astype(int).to_numpy()
    dummy['anomaly_score'] = np.linspace(.1, .9, len(dummy))
    dummy['threshold'] = .5
    dummy['prediction'] = (dummy.anomaly_score >= .5).astype(int)
    aligned = exact_test_alignment(dummy, small.rename(columns={'label': 'ground_truth'}))
    assert len(aligned) == len(small)
    calculate_metrics(aligned.ground_truth, aligned.anomaly_score, .5)
    print('Exact merge and metrics checks OK')

    if importlib.util.find_spec('torchvision') is None:
        print('Model integration smoke test NOT RUN: torchvision is not installed. Install dependencies with: python -m pip install -r requirements.txt')
        return

    # Run an end-to-end bounded train/validation + one test sequence inference.
    smoke_dir = ROOT / 'outputs' / 'smoke_test'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_config = smoke_dir / 'frame_baseline_smoke.yaml'
    cfg = json.loads('{}')
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg.update({'manifest': str(manifest_path.resolve()), 'dataset_root': str(root.resolve()), 'pretrained': True, 'batch_size': 2, 'image_size': 256, 'knn_k': 2, 'threshold_quantile': .995})
    smoke_config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    command = [sys.executable, '-m', 'src.training.train_frame_baseline', '--config', str(smoke_config), '--max-train-frames', '2', '--max-val-frames', '2']
    print('Running bounded feature/threshold smoke test (2 frames per selected train/val sequence)...')
    subprocess.run(command, cwd=ROOT, check=True)
    runs = sorted((ROOT / 'models').glob('E0_frame_resnet18_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError('Smoke training did not create a run')
    run_dir = runs[0]
    # Inference stays bounded to the first test sequence and the same exact-key protocol.
    subprocess.run([sys.executable, '-m', 'src.evaluation.evaluate_frame', '--run-dir', str(run_dir), '--max-test-frames', '2'], cwd=ROOT, check=True)
    required_files = ['config.json', 'threshold.json', 'memory_bank.pt', 'run_summary.md', 'predictions.csv', 'metrics.json']
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing: raise RuntimeError(f'Smoke run missing outputs: {missing}')
    print(f'End-to-end smoke test passed on {get_device()}: {run_dir}')


if __name__ == '__main__':
    main()
