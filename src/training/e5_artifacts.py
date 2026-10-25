"""Validate and load E4 adapted embeddings without recomputing the backbone."""
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.video_loader import R01VideoClipDataset


def build_split_clip_records(split_frames,clip_length=16,stride=4):
    # Constructor builds metadata only; it does not open or decode frames.
    return R01VideoClipDataset(split_frames,None,clip_length,stride).clips


def _compare_clip_rows(actual,expected,name):
    fields=('sequence','start_frame','end_frame','frame_indices','frame_names')
    if len(actual)!=len(expected): raise ValueError(f'{name} clip metadata count {len(actual)} != expected {len(expected)}')
    for i,(a,b) in enumerate(zip(actual,expected)):
        for field in fields:
            if a.get(field)!=b.get(field): raise ValueError(f'{name} embedding order/clip metadata mismatch at row {i}, field={field}')


def _verify_csv_clip_order(path,expected,name):
    if not Path(path).is_file(): return
    table=pd.read_csv(path)
    if len(table)!=len(expected): raise ValueError(f'{name} E4 clip score sidecar count differs from embeddings')
    for i,row in table.iterrows():
        exp=expected[i]
        for field in ('sequence','start_frame','end_frame'):
            if field in row and str(row[field])!=str(exp[field]): raise ValueError(f'{name} E4 embedding order mismatch at row {i}, {field}')
        for field in ('frame_names','frame_indices'):
            if field in table.columns and isinstance(row[field],str):
                try: value=ast.literal_eval(row[field])
                except (ValueError,SyntaxError) as exc: raise ValueError(f'{name} E4 sidecar has unreadable {field} at row {i}') from exc
                expected_value=exp[field]
                if field=='frame_names': value=[str(x) for x in value]; expected_value=[str(x) for x in expected_value]
                else: value=[int(x) for x in value]; expected_value=[int(x) for x in expected_value]
                if value!=expected_value: raise ValueError(f'{name} E4 embedding order mismatch at row {i}, {field}')


def load_e4_metadata_and_embeddings(e4_dir,manifest,include_test=False):
    e4=Path(e4_dir)
    if e4.name!='E4_adapted_swin_20261022_175740': raise ValueError(f'E5 must use the designated E4 run E4_adapted_swin_20261022_175740, got {e4}')
    if not e4.is_dir(): raise FileNotFoundError(f'Real E4 run directory is unavailable: {e4}. Mount Google Drive and verify the configured path; synthetic caches are not accepted.')
    cfg=json.loads((e4/'config.json').read_text(encoding='utf-8'))
    if cfg.get('synthetic_smoke_cache') or cfg.get('synthetic'): raise ValueError('Synthetic E4 cache is forbidden for E5.')
    for key,value in {'embedding_dim':768,'clip_length':16,'stride':4,'weights_name':'Swin3D_T_Weights.KINETICS400_V1'}.items():
        if cfg.get(key)!=value: raise ValueError(f'E4 metadata mismatch {key}: expected {value}, got {cfg.get(key)}')
    if cfg.get('model_name')!='swin3d_t' or cfg.get('pretrained') is not True: raise ValueError('E4 embeddings must come from pretrained torchvision swin3d_t')
    splits={name:manifest[manifest.split==name].copy() for name in ('train','val','test')}
    expected_sequences={'train':sorted(splits['train'].sequence.unique().tolist()),'val':sorted(splits['val'].sequence.unique().tolist()),'test':sorted(splits['test'].sequence.unique().tolist())}
    expected_counts={'train':27,'val':7,'test':15}
    for split,count in expected_counts.items():
        if len(expected_sequences[split])!=count: raise ValueError(f'E5 expected {count} {split} sequences; found {len(expected_sequences[split])}')
    expected_frame_rows={'train':6196,'val':1612,'test':3685}
    for split,count in expected_frame_rows.items():
        if len(splits[split])!=count: raise ValueError(f'E5 expected {count} {split} manifest frames; found {len(splits[split])}')
    declared={'train':cfg.get('train_sequences'),'val':cfg.get('validation_sequences'),'test':cfg.get('test_sequences')}
    for key in expected_sequences:
        if declared[key] is not None and expected_sequences[key]!=declared[key]: raise ValueError(f'E4 {key} sequences differ from current manifest')
    if any(not s.startswith('training/') for split in ('train','val') for s in expected_sequences[split]): raise ValueError('Train/validation must contain only normal training sequences')
    if any(not s.startswith('testing/') for s in expected_sequences['test']): raise ValueError('Test split must contain official testing sequences')
    clips={split:build_split_clip_records(rows,16,4) for split,rows in splits.items()}
    expected_clip_counts={'train':1479,'val':385,'test':882}
    for split,count in expected_clip_counts.items():
        if len(clips[split])!=count: raise ValueError(f'E5 expected {count} {split} clips from the manifest; found {len(clips[split])}')
    for split,key in [('train','adapted_train_embeddings.pt'),('val','adapted_validation_embeddings.pt'),('test','adapted_test_embeddings.pt')]:
        path=e4/key
        if not path.is_file(): raise FileNotFoundError(f'Missing saved E4 embeddings: {path}')
    train=torch.load(e4/'adapted_train_embeddings.pt',map_location='cpu',weights_only=True).float()
    val=torch.load(e4/'adapted_validation_embeddings.pt',map_location='cpu',weights_only=True).float()
    for name,z in [('train',train),('validation',val)]:
        expected_len=len(clips['train' if name=='train' else 'val'])
        if z.shape!=(expected_len,768) or not torch.isfinite(z).all(): raise ValueError(f'E4 {name} embedding shape/value mismatch: {tuple(z.shape)}, expected {(expected_len,768)}')
        norms=z.norm(dim=1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=1e-4): raise ValueError(f'E4 {name} embeddings are not L2-normalized')
    # E4 validation sidecar is saved alongside the validation feature matrix.
    val_sidecar=e4/'validation_clips.json'
    if not val_sidecar.is_file(): raise FileNotFoundError(f'Missing E4 validation clip order sidecar: {val_sidecar}')
    _compare_clip_rows(json.loads(val_sidecar.read_text()),clips['val'],'validation')
    train_sidecar=e4/'train_clips.json'
    if train_sidecar.is_file(): _compare_clip_rows(json.loads(train_sidecar.read_text()),clips['train'],'train')
    test_sidecar=e4/'clip_scores.csv'
    if not test_sidecar.is_file(): raise FileNotFoundError(f'Missing E4 test clip order sidecar: {test_sidecar}')
    _verify_csv_clip_order(test_sidecar,clips['test'],'test')
    if cfg.get('train_clip_count') is not None and int(cfg['train_clip_count'])!=len(clips['train']): raise ValueError('E4 train clip count differs from manifest')
    if cfg.get('validation_clip_count') is not None and int(cfg['validation_clip_count'])!=len(clips['val']): raise ValueError('E4 validation clip count differs from manifest')
    if cfg.get('test_clip_count') is not None and int(cfg['test_clip_count'])!=len(clips['test']): raise ValueError('E4 test clip count differs from manifest')
    bank_path=e4/'memory_bank.pt'
    if not bank_path.is_file(): raise FileNotFoundError(f'Missing E4 train memory bank for train embedding order check: {bank_path}')
    memory=torch.load(bank_path,map_location='cpu',weights_only=True).float()
    if memory.shape!=train.shape or not torch.equal(memory,train): raise ValueError('E4 train embedding rows do not exactly match its saved normal memory bank')
    test_path=e4/'adapted_test_embeddings.pt'
    test=torch.load(test_path,map_location='cpu',weights_only=True).float()
    if test.shape!=(len(clips['test']),768) or not torch.isfinite(test).all(): raise ValueError(f'E4 test embedding shape/value mismatch: {tuple(test.shape)}')
    test_norms=test.norm(dim=1)
    if not torch.allclose(test_norms,torch.ones_like(test_norms),atol=1e-4): raise ValueError('E4 test embeddings are not L2-normalized')
    result={'config':cfg,'splits':splits,'clips':clips,'train_embeddings':train,'validation_embeddings':val}
    if include_test: result['test_embeddings']=test
    return result
