"""E5 cache/order/window/gradient smoke test; never loads test labels."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.models.temporal_gru import GRUTemporalPredictor,next_embedding_mse
from src.training.e5_artifacts import load_e4_metadata_and_embeddings
from src.training.train_temporal_gru import TemporalWindowDataset,clip_score_table,map_clip_scores_to_frames
from src.utils.device import get_device
from src.utils.seed import set_seed


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/e5_adapted_gru.yaml'); p.add_argument('--e4-run-dir'); p.add_argument('--manifest'); a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); cfg['e4_run_dir']=a.e4_run_dir or cfg['e4_run_dir']; cfg['manifest']=a.manifest or cfg['manifest']
    if str(Path(cfg['repo_root']).resolve())!=str(ROOT.resolve()): raise RuntimeError(f"Run from the configured repo root: {cfg['repo_root']} (current script root: {ROOT})")
    manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split']); data=load_e4_metadata_and_embeddings(cfg['e4_run_dir'],manifest,include_test=True); clips=data['clips']
    datasets={'train':TemporalWindowDataset(data['train_embeddings'],clips['train'],8),'validation':TemporalWindowDataset(data['validation_embeddings'],clips['val'],8),'test':TemporalWindowDataset(data['test_embeddings'],clips['test'],8)}
    counts={k:len(v) for k,v in datasets.items()}; assert counts=={'train':1263,'validation':329,'test':762},counts
    for dataset in datasets.values():
        for seq,context,target in dataset.windows:
            ids=context+[target]
            if any(dataset.clips[i]['sequence']!=seq for i in ids): raise ValueError('Temporal window crossed a sequence boundary')
    # Verify clip/embedding row order through per-sequence monotonic clip positions.
    for split,key in [('train','train'),('validation','val'),('test','test')]:
        sequence_order=[c['sequence'] for c in clips[key]]
        if sequence_order!=sorted(sequence_order): raise ValueError(f'{split} E4 embedding sequence order is not sorted')
        for seq in set(sequence_order):
            starts=[c['start_frame'] for c in clips[key] if c['sequence']==seq]
            if starts!=sorted(starts): raise ValueError(f'{split} E4 clip row order is not temporal for {seq}')
    set_seed(cfg['seed']); device=get_device(); model=GRUTemporalPredictor(768,256,1,0.0).to(device); x,y,seq,target_id=datasets['train'][0]; xx=x.unsqueeze(0).to(device); yy=y.unsqueeze(0).to(device)
    prediction=model(xx); loss=next_embedding_mse(prediction,yy); assert xx.shape==(1,8,768) and yy.shape==(1,768) and prediction.shape==(1,768) and torch.isfinite(loss)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']); batch_x=torch.stack([datasets['train'][i][0] for i in range(32)]).to(device); batch_y=torch.stack([datasets['train'][i][1] for i in range(32)]).to(device)
    with torch.no_grad(): initial=float(next_embedding_mse(model(batch_x),batch_y))
    grads_finite=True
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True); out=model(batch_x); l=next_embedding_mse(out,batch_y); l.backward()
        grads_finite &= all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()); optimizer.step()
    with torch.no_grad(): final=float(next_embedding_mse(model(batch_x),batch_y))
    if not grads_finite or not final<initial: raise RuntimeError(f'E5 smoke gradient/loss check failed: {initial} -> {final}')
    val_clip_scores=clip_score_table(model,data['validation_embeddings'],clips['val'],8,device); val_frames=map_clip_scores_to_frames(clips['val'],val_clip_scores)
    if len(val_frames)!=1612: raise ValueError(f'Expected 1612 validation frame scores, got {len(val_frames)}')
    test_clip_scores=clip_score_table(model,data['test_embeddings'],clips['test'],8,device); test_frames=map_clip_scores_to_frames(clips['test'],test_clip_scores)
    if len(test_frames)!=3685 or test_frames.duplicated(['split','sequence','frame_name']).any(): raise ValueError('E5 test frame mapping is not exact')
    result={'e4_saved_embeddings_loaded':True,'synthetic_cache_used':False,'e4_run_dir':str(Path(cfg['e4_run_dir']).resolve()),'embedding_shapes':{'train':list(data['train_embeddings'].shape),'validation':list(data['validation_embeddings'].shape),'test':list(data['test_embeddings'].shape)},'clip_counts':{'train':len(clips['train']),'validation':len(clips['val']),'test':len(clips['test'])},'window_counts':counts,'input_shape':list(xx.shape),'target_shape':list(yy.shape),'prediction_shape':list(prediction.shape),'first_window_sequence':seq,'first_window_target_clip_index':target_id,'embedding_order_verified':True,'sequence_boundaries_preserved':True,'initial_smoke_loss':initial,'final_smoke_loss':final,'all_gru_gradients_finite':grads_finite,'validation_frame_count':len(val_frames),'test_frame_count_label_free':len(test_frames),'test_labels_loaded':False,'full_training_started':False}
    out=Path(cfg['smoke_report']); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print('E5 smoke test PASSED using the validated real E4 saved embeddings; no full training started and test labels were not read.')

if __name__=='__main__':
    try: main()
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        raise SystemExit(f'E5 SMOKE TEST FAILED CLOSED: {exc}') from exc
