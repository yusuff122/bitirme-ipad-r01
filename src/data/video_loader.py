"""Sequence-local fixed-length clip extraction for frame manifests."""
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def clip_start_positions(num_frames: int, clip_length: int = 16, stride: int = 4):
    if clip_length < 1 or stride < 1:
        raise ValueError('clip_length and stride must be positive')
    if num_frames < clip_length:
        return []
    last = num_frames - clip_length
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)  # tail-align to cover every sequence's final frame
    return starts


def build_clip_records(records, clip_length=16, stride=4):
    """Build clip metadata without ever joining rows from different sequences."""
    grouped = defaultdict(list)
    for row in records.to_dict('records'):
        grouped[str(row['sequence'])].append(row)
    clips = []
    for sequence in sorted(grouped):
        rows = sorted(grouped[sequence], key=lambda x: int(x['frame_index']))
        indices = [int(r['frame_index']) for r in rows]
        if len(indices) > 1 and any(b - a != 1 for a, b in zip(indices, indices[1:])):
            raise ValueError(f'Non-contiguous frame indices in {sequence}; refusing to form ambiguous clips')
        for start in clip_start_positions(len(rows), clip_length, stride):
            selected = rows[start:start + clip_length]
            clips.append({
                'sequence': sequence,
                'start_frame': int(selected[0]['frame_index']),
                'end_frame': int(selected[-1]['frame_index']),
                'frame_names': [str(r['frame_name']) for r in selected],
                'frame_indices': [int(r['frame_index']) for r in selected],
                'paths': [str(r['path']) for r in selected],
            })
    return clips


class R01VideoClipDataset(Dataset):
    """Returns official-transform-ready ``T,C,H,W`` uint8 clips plus metadata."""
    def __init__(self, records, video_transform, clip_length=16, stride=4):
        self.video_transform = video_transform
        self.clips = build_clip_records(records, clip_length, stride)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        meta = self.clips[index]
        frames = []
        for path in meta['paths']:
            with Image.open(Path(path)) as image:
                image = image.convert('RGB')
                frames.append(torch.from_numpy(__import__('numpy').array(image, copy=True)))
        # The torchvision VideoClassification weights transform expects T,C,H,W.
        video = torch.stack(frames).permute(0, 3, 1, 2).contiguous()
        return self.video_transform(video), {k: v for k, v in meta.items() if k != 'paths'}


def collate_video_clips(batch):
    videos, metadata = zip(*batch)
    return torch.stack(videos), list(metadata)


def map_clip_scores_to_frames(clip_records, clip_scores, frame_records):
    """Overlap-average clip scores onto every manifest frame; assert full coverage."""
    if len(clip_records) != len(clip_scores):
        raise ValueError('clip metadata and score counts differ')
    sums = defaultdict(float)
    counts = defaultdict(int)
    for clip, score in zip(clip_records, clip_scores):
        value = float(score)
        for name in clip['frame_names']:
            key = (str(clip['sequence']), str(name))
            sums[key] += value
            counts[key] += 1
    output = []
    for row in frame_records.to_dict('records'):
        key = (str(row['sequence']), str(row['frame_name']))
        n = counts.get(key, 0)
        if not n:
            raise ValueError(f'No clip covers frame {key}; refusing incomplete frame timeline')
        output.append({**row, 'anomaly_score': sums[key] / n, 'num_covering_clips': n})
    if len(output) != len(frame_records):
        raise ValueError(f'Frame coverage mismatch: {len(output)} != {len(frame_records)}')
    return output
