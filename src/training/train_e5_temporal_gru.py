"""E5: train E2's GRU predictor on saved R01-adapted E4 embeddings."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_metrics
from src.models.temporal_gru import GRUTemporalPredictor, next_embedding_mse
from src.training.e5_artifacts import load_e4_metadata_and_embeddings
from src.training.train_temporal_gru import TemporalWindowDataset, map_clip_scores_to_frames, clip_score_table
from src.utils.device import get_device
from src.utils.seed import set_seed


def _plot_history(history,path):
    h=pd.DataFrame(history); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(h.epoch,h.train_loss,label='Train MSE'); ax.plot(h.epoch,h.validation_loss,label='Validation MSE'); ax.set(xlabel='Epoch',ylabel='Next-embedding MSE',title='E5 Training'); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/e5_adapted_gru.yaml'); p.add_argument('--e4-run-dir'); p.add_argument('--manifest'); p.add_argument('--output-root'); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8')); cfg['e4_run_dir']=a.e4_run_dir or cfg['e4_run_dir']; cfg['manifest']=a.manifest or cfg['manifest']; cfg['output_root']=a.output_root or cfg['output_root']
    manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split'])
    data=load_e4_metadata_and_embeddings(cfg['e4_run_dir'],manifest,include_test=False); clips=data['clips']; splits=data['splits']
    train_ds=TemporalWindowDataset(data['train_embeddings'],clips['train'],cfg['context_length']); val_ds=TemporalWindowDataset(data['validation_embeddings'],clips['val'],cfg['context_length'])
    test_windows=sum(max(0,n-cfg['context_length']) for n in pd.Series([c['sequence'] for c in clips['test']]).value_counts())
    counts={'train':len(train_ds),'validation':len(val_ds),'test':int(test_windows)}
    if counts!={'train':1263,'validation':329,'test':762}: raise ValueError(f'E5 temporal window count mismatch: {counts}')
    if len(data['train_embeddings'])!=1479 or len(data['validation_embeddings'])!=385: raise ValueError('E4 embedding counts differ from expected train/validation clip order')
    set_seed(cfg['seed']); device=get_device(); model=GRUTemporalPredictor(cfg['input_dim'],cfg['hidden_dim'],cfg['num_layers'],cfg['dropout']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']); batch_size=a.batch_size or cfg['batch_size']
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=cfg['num_workers']); val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=cfg['num_workers'])
    run_time=datetime.now().astimezone(); run=Path(cfg['output_root'])/f"E5_adapted_gru_{run_time.strftime('%Y%m%d_%H%M%S')}"; run.mkdir(parents=True,exist_ok=False); (run/'figures').mkdir()
    run_cfg={**cfg,'e4_run_dir':str(Path(cfg['e4_run_dir']).resolve()),'manifest':str(Path(cfg['manifest']).resolve()),'device':str(device),'timestamp':run_time.isoformat(),'model':'GRUTemporalPredictor','train_sequences':sorted(splits['train'].sequence.unique()),'validation_sequences':sorted(splits['val'].sequence.unique()),'test_sequences':sorted(splits['test'].sequence.unique()),'train_clip_count':len(clips['train']),'validation_clip_count':len(clips['val']),'test_clip_count':len(clips['test']),'train_temporal_windows':counts['train'],'validation_temporal_windows':counts['validation'],'test_temporal_windows':counts['test'],'num_train_frames':len(splits['train']),'num_validation_frames':len(splits['val']),'num_test_frames':len(splits['test']),'e4_embeddings_reused':True,'embedding_order':'sorted sequence then ascending start_frame, matching E4 R01VideoClipDataset extraction; validation JSON and test clip CSV sidecars are cross-checked when present','test_labels_read':False,'threshold_quantile':cfg['threshold_quantile']}
    (run/'config.json').write_text(json.dumps(run_cfg,indent=2),encoding='utf-8')
    best=float('inf'); stale=0; best_state=None; best_epoch=None; history=[]
    for epoch in range(1,(a.epochs or cfg['epochs'])+1):
        model.train(); tr=[]
        for x,y,_,_ in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); loss=next_embedding_mse(model(x),y)
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite E5 training loss in epoch {epoch}')
            loss.backward(); optimizer.step(); tr.append(float(loss.detach().cpu()))
        model.eval(); va=[]
        with torch.inference_mode():
            for x,y,_,_ in val_loader:
                loss=next_embedding_mse(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss): raise ValueError(f'Nonfinite E5 validation loss in epoch {epoch}')
                va.append(float(loss.cpu()))
        train_loss=float(np.mean(tr)); val_loss=float(np.mean(va)); history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss}); print(f'Epoch {epoch:02d}: train MSE={train_loss:.8f}; validation MSE={val_loss:.8f}')
        pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
        if val_loss<best:
            best=val_loss; best_epoch=epoch; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':best_state,'config':run_cfg,'best_epoch':best_epoch,'best_validation_loss':best},run/'best_checkpoint.pt')
        else:
            stale+=1
            if stale>=cfg['patience']: print('Early stop by validation normal MSE; best epoch:',best_epoch); break
    if best_state is None: raise RuntimeError('No E5 checkpoint was selected')
    model.load_state_dict(best_state); model.to(device).eval(); val_scores=clip_score_table(model,data['validation_embeddings'],clips['val'],cfg['context_length'],device)
    frame_scores=map_clip_scores_to_frames(clips['val'],val_scores); expected=splits['val'][['sequence','frame_name','frame_index']]
    merged=expected.merge(frame_scores,on=['sequence','frame_name','frame_index'],how='outer',validate='one_to_one',indicator=True)
    if len(merged)!=1612 or not (merged._merge=='both').all() or merged.anomaly_score.isna().any(): raise ValueError('Validation frame mapping incomplete or duplicated')
    scores=merged.anomaly_score.to_numpy(float); threshold=float(np.quantile(scores,cfg['threshold_quantile']))
    threshold_info={'threshold':threshold,'method':'validation normal frame-score quantile','quantile':cfg['threshold_quantile'],'score_granularity':'frame score after mean aggregation of covering clip MSE scores','num_validation_frames':len(scores),'validation_score_mean':float(scores.mean()),'validation_score_std':float(scores.std()),'validation_score_min':float(scores.min()),'validation_score_max':float(scores.max()),'validation_score_p95':float(np.quantile(scores,.95)),'validation_score_p99':float(np.quantile(scores,.99)),'validation_score_p995':float(np.quantile(scores,.995)),'test_used_for_threshold':False}
    run_cfg.update({'best_epoch':best_epoch,'best_validation_loss':best,'threshold':threshold,'test_labels_read':False})
    (run/'config.json').write_text(json.dumps(run_cfg,indent=2),encoding='utf-8'); (run/'threshold.json').write_text(json.dumps(threshold_info,indent=2),encoding='utf-8')
    torch.save({'state_dict':best_state,'config':run_cfg,'best_epoch':best_epoch,'best_validation_loss':best},run/'model.pt')
    pd.DataFrame(clips['val']).assign(anomaly_score=val_scores).to_csv(run/'validation_clip_scores.csv',index=False); _plot_history(history,run/'figures'/'training_curve.png')
    (run/'run_summary.md').write_text(f"# E5 adapted-feature GRU\n\nE4 run: `{cfg['e4_run_dir']}`; saved adapted embeddings reused without recomputation. GRU 768→256, context={cfg['context_length']}. Train/validation/test windows: {counts}. Best epoch={best_epoch}; validation normal frame q={cfg['threshold_quantile']} threshold={threshold:.8f}. Test labels read during training: false.\n")
    print(json.dumps({'run_dir':str(run),'best_epoch':best_epoch,'best_validation_loss':best,'threshold':threshold,'window_counts':counts,'e4_embeddings_reused':True,'test_labels_read':False},indent=2))

if __name__=='__main__': main()
