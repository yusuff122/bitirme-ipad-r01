"""Small E2 pipeline smoke test using cached E1 embeddings; not a full run."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml

from src.models.temporal_gru import GRUTemporalPredictor,next_embedding_mse
from src.training.train_temporal_gru import TemporalWindowDataset,clip_score_table,map_clip_scores_to_frames,validate_e1_cache
from src.utils.device import get_device
from src.utils.seed import set_seed


def main():
    config=yaml.safe_load((ROOT/'configs/temporal_gru.yaml').read_text())
    manifest=pd.read_csv(config['manifest'],usecols=['path','sequence','frame_name','frame_index','split'])
    e1=ROOT/'outputs/reports/e1_diagnostics/source/E1_video_swin16_20261022_090120'
    if not e1.is_dir(): e1=Path(config['e1_run_dir'])
    e1cfg,splits,clips,emb=validate_e1_cache(e1,manifest)
    print('Reused E1 embedding cache:',e1)
    print('Cache metadata:',e1cfg['model_name'],e1cfg['weights_name'],'clip=',e1cfg['clip_length'],'stride=',e1cfg['stride'],'dim=',e1cfg['embedding_dim'])

    train_seq=sorted(splits['train'].sequence.unique())[0]
    val_seq=sorted(splits['val'].sequence.unique())[0]
    test_seq=sorted(splits['test'].sequence.unique())[0]
    train_ids=[i for i,c in enumerate(clips['train']) if c['sequence']==train_seq]
    val_ids=[i for i,c in enumerate(clips['validation']) if c['sequence']==val_seq]
    test_ids=[i for i,c in enumerate(clips['test']) if c['sequence']==test_seq]
    def subset_records(records,ids): return [records[i] for i in ids]
    train_clips=subset_records(clips['train'],train_ids); val_clips=subset_records(clips['validation'],val_ids); test_clips=subset_records(clips['test'],test_ids)
    train_ds=TemporalWindowDataset(emb['train'][train_ids],train_clips,config['context_length'])
    val_ds=TemporalWindowDataset(emb['validation'][val_ids],val_clips,config['context_length'])
    all_test_windows=sum(max(0,n-config['context_length']) for n in pd.Series([c['sequence'] for c in clips['test']]).value_counts())
    print('One-sequence temporal windows: train=',len(train_ds),'validation=',len(val_ds),'test=',max(0,len(test_clips)-config['context_length']))
    train_window_count=sum(max(0,n-config['context_length']) for n in pd.Series([c['sequence'] for c in clips['train']]).value_counts())
    val_window_count=sum(max(0,n-config['context_length']) for n in pd.Series([c['sequence'] for c in clips['validation']]).value_counts())
    print('Full cached split windows: train=',train_window_count,'validation=',val_window_count,'test=',all_test_windows)
    x0,y0,seq0,_=train_ds[0]
    set_seed(config['seed']); device=get_device(); model=GRUTemporalPredictor(768,config['hidden_dim'],1,0.0).to(device)
    xx=x0.unsqueeze(0).to(device); yy=y0.unsqueeze(0).to(device); pred=model(xx)
    loss=next_embedding_mse(pred,yy)
    param_count=sum(p.numel() for p in model.parameters())
    print('GRU trainable parameter count:',param_count)
    print('Input tensor shape:',tuple(xx.shape),'target shape:',tuple(yy.shape),'GRU output/prediction shape:',tuple(pred.shape),'single batch loss:',float(loss.detach()))
    assert xx.shape==(1,config['context_length'],768) and yy.shape==(1,768) and pred.shape==(1,768)
    assert torch.isfinite(loss)
    # Exercise a handful of optimizer steps on one fixed batch and require loss to fall.
    batch_x=torch.stack([train_ds[i][0] for i in range(min(32,len(train_ds)))]).to(device)
    batch_y=torch.stack([train_ds[i][1] for i in range(min(32,len(train_ds)))]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    with torch.no_grad(): initial=float(next_embedding_mse(model(batch_x),batch_y))
    losses=[]
    for _ in range(12):
        model.train(); opt.zero_grad(set_to_none=True); p=model(batch_x); l=next_embedding_mse(p,batch_y); l.backward(); opt.step(); losses.append(float(l.detach().cpu()))
    model.eval()
    with torch.no_grad(): final=float(next_embedding_mse(model(batch_x),batch_y))
    print('Smoke optimization loss:',initial,'->',final,'(12 repeated mini-batch steps)')
    assert np.isfinite(losses).all() and final<initial, 'Smoke optimizer loss did not decrease'
    # Prove no constructed context crosses a sequence boundary.
    for dataset in (train_ds,val_ds):
        for seq,context_ids,target_id in dataset.windows:
            assert all(dataset.clips[i]['sequence']==seq for i in context_ids+[target_id])
    print('Sequence boundary isolation: OK')

    # Validate frame-level validation threshold mapping at full validation split granularity.
    val_scores=clip_score_table(model,emb['validation'],clips['validation'],config['context_length'],device)
    val_frames=map_clip_scores_to_frames(clips['validation'],val_scores)
    assert len(val_frames)==len(splits['val'])==1612 and val_frames.anomaly_score.notna().all()
    val_threshold=float(np.quantile(val_frames.anomaly_score,config['threshold_quantile']))
    print('Validation frame mapping:',len(val_frames),'frames; frame-level q=.995 threshold:',val_threshold)

    # Test mapping diagnostic is label-free; use cached test embeddings only.
    all_test_emb=torch.load(e1/'embeddings/test_clip_embeddings.pt',map_location='cpu',weights_only=True).float()
    test_scores=clip_score_table(model,all_test_emb,clips['test'],config['context_length'],device)
    test_frames=map_clip_scores_to_frames(clips['test'],test_scores)
    assert len(test_frames)==3685 and not test_frames.duplicated(['split','sequence','frame_name']).any()
    assert test_frames.anomaly_score.notna().all()
    print('Label-free test frame mapping:',len(test_frames),'frames, exact expected=3685; score finite everywhere')
    result={'cache_reused':True,'e1_cache_dir':str(e1),'model_name':e1cfg['model_name'],'weights_name':e1cfg['weights_name'],
            'train_windows_one_sequence':len(train_ds),'validation_windows_one_sequence':len(val_ds),'test_windows_one_sequence':max(0,len(test_clips)-config['context_length']),
            'train_windows_full_split':train_window_count,'validation_windows_full_split':val_window_count,'test_windows_full_split':all_test_windows,
            'input_shape':list(xx.shape),'target_shape':list(yy.shape),'prediction_shape':list(pred.shape),'parameter_count':param_count,
            'initial_smoke_loss':initial,'final_smoke_loss':final,'validation_frame_count':len(val_frames),'validation_threshold_smoke':val_threshold,
            'label_free_test_frame_count':len(test_frames),'sequence_boundary_isolation':True,'full_training_started':False,'test_labels_loaded':False}
    result_path=ROOT/'outputs/reports/e2_smoke_test.json'; result_path.parent.mkdir(parents=True,exist_ok=True); result_path.write_text(json.dumps(result,indent=2))
    print('E2 smoke test PASSED. No full E2 run was started and no test labels were loaded.')


if __name__=='__main__': main()
