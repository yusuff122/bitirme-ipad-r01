"""Finish E4 artifacts from an already-trained best checkpoint; never trains."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.video_loader import map_clip_scores_to_frames
from src.models.adapted_video_swin import AdaptedVideoSwin, WEIGHTS_NAME
from src.models.frame_baseline import knn_scores
from src.training.train_adapted_video_swin import extract_embeddings, validate_split_manifest
from src.training.train_temporal_gru import validate_e1_cache
from src.utils.device import get_device


def main():
    p=argparse.ArgumentParser(description='Extract E4 train/validation embeddings and calibrate threshold from a saved checkpoint; no training occurs.')
    p.add_argument('--run-dir',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--config',default='configs/video_swin_adapted.yaml'); p.add_argument('--e1-run-dir'); p.add_argument('--manifest'); a=p.parse_args()
    run=Path(a.run_dir); checkpoint_path=Path(a.checkpoint); ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if not isinstance(ckpt,dict): raise ValueError('Unsupported E4 checkpoint file')
    state=ckpt.get('state_dict',ckpt); saved_cfg=ckpt.get('config',{})
    config_path=run/'config.json'
    if config_path.exists(): cfg={**yaml.safe_load(Path(a.config).read_text()),**json.loads(config_path.read_text())}
    else: cfg={**yaml.safe_load(Path(a.config).read_text()),**saved_cfg}
    if a.manifest: cfg['manifest']=a.manifest
    if a.e1_run_dir: cfg['e1_run_dir']=a.e1_run_dir
    if not cfg.get('manifest') or not cfg.get('e1_run_dir'): raise ValueError('Need manifest and E1 run path from config or CLI')
    if cfg.get('weights_name',WEIGHTS_NAME)!=WEIGHTS_NAME: raise ValueError('Checkpoint config weights mismatch')
    manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split']); splits=validate_split_manifest(manifest); e1cfg,e1splits,e1clips,_=validate_e1_cache(Path(cfg['e1_run_dir']),manifest)
    device=get_device(); model=AdaptedVideoSwin(pretrained=True).to(device); missing,unexpected=model.load_state_dict(state,strict=False)
    if missing or unexpected: raise ValueError(f'Checkpoint/model parameter mismatch; missing={missing[:5]}, unexpected={unexpected[:5]}')
    model.eval(); cfg.setdefault('clip_length',16); cfg.setdefault('stride',4); cfg.setdefault('num_workers',0); cfg.setdefault('mixed_precision',True); cfg.setdefault('knn_k',5); cfg.setdefault('threshold_quantile',.995)
    transform=model.make_transform(); train_emb,train_clips=extract_embeddings(splits['train'],model,transform,device,cfg,'train'); val_emb,val_clips=extract_embeddings(splits['val'],model,transform,device,cfg,'validation')
    for name,actual,cached in [('train',train_clips,e1clips['train']),('validation',val_clips,e1clips['validation'])]:
        keys=('sequence','start_frame','end_frame','frame_names','frame_indices')
        if len(actual)!=len(cached) or any(any(actual[i][k]!=cached[i][k] for k in keys) for i in range(len(actual))): raise ValueError(f'{name} extraction clip indexing differs from validated E1 cache')
    if len(train_emb)!=1479 or len(val_emb)!=385: raise ValueError(f'Unexpected train/validation clip counts: {len(train_emb)}/{len(val_emb)}')
    clip_scores=knn_scores(val_emb,train_emb,cfg['knn_k']).numpy(); frames=pd.DataFrame(map_clip_scores_to_frames(val_clips,clip_scores,splits['val']))
    if len(frames)!=1612 or frames.anomaly_score.isna().any(): raise ValueError('Validation frame mapping is incomplete')
    threshold=float(np.quantile(frames.anomaly_score,cfg['threshold_quantile']))
    run.mkdir(parents=True,exist_ok=True); (run/'figures').mkdir(exist_ok=True)
    best_epoch=ckpt.get('best_epoch',saved_cfg.get('best_epoch',cfg.get('best_epoch'))); best_loss=ckpt.get('best_validation_contrastive_loss',saved_cfg.get('best_validation_contrastive_loss'))
    cfg.update({'e1_run_dir':str(Path(cfg['e1_run_dir']).resolve()),'manifest':str(Path(cfg['manifest']).resolve()),'device':str(device),'weights_name':WEIGHTS_NAME,'embedding_dim':768,'projection_dim':128,'train_clip_count':len(train_emb),'validation_clip_count':len(val_emb),'test_clip_count':882,'num_test_frames':3685,'knn_k':cfg['knn_k'],'best_epoch':best_epoch,'best_validation_contrastive_loss':best_loss,'test_labels_read':False,'e1_cache_metadata_validated':True,'finalized_from_checkpoint':str(checkpoint_path.resolve())})
    model_payload={'state_dict':model.state_dict(),'config':cfg,'best_epoch':best_epoch,'best_validation_contrastive_loss':best_loss}; torch.save(model_payload,run/'model.pt'); torch.save({k:v for k,v in model.state_dict().items() if k.startswith('projection_head.')},run/'projection_head.pt'); torch.save(train_emb,run/'adapted_train_embeddings.pt'); torch.save(val_emb,run/'adapted_validation_embeddings.pt'); torch.save(train_emb,run/'memory_bank.pt')
    pd.DataFrame(val_clips).assign(anomaly_score=clip_scores).to_csv(run/'clip_scores.csv',index=False); (run/'validation_clips.json').write_text(json.dumps(val_clips),encoding='utf-8')
    (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8'); (run/'threshold.json').write_text(json.dumps({'threshold':threshold,'method':'validation normal frame-score quantile','quantile':cfg['threshold_quantile'],'score_granularity':'frame-level after mean clip coverage aggregation','num_validation_frames':len(frames),'validation_score_mean':float(frames.anomaly_score.mean()),'validation_score_std':float(frames.anomaly_score.std()),'validation_score_min':float(frames.anomaly_score.min()),'validation_score_max':float(frames.anomaly_score.max()),'test_used_for_threshold':False},indent=2),encoding='utf-8')
    if not (run/'training_history.csv').exists():
        pd.DataFrame([{'epoch':best_epoch if best_epoch is not None else np.nan,'train_contrastive_loss':np.nan,'validation_contrastive_loss':best_loss if best_loss is not None else np.nan,'note':'checkpoint metadata only; epoch history unavailable'}]).to_csv(run/'training_history.csv',index=False)
    (run/'run_summary.md').write_text(f"# E4 adapted Video Swin\n\nFinalized from existing checkpoint `{checkpoint_path}` without training. Best checkpoint epoch: {best_epoch}; validation contrastive loss metadata: {best_loss}. Train/validation clips: {len(train_emb)}/{len(val_emb)}. Validation frame threshold q={cfg['threshold_quantile']}: {threshold:.8f}. Test labels read: false.\n")
    print(json.dumps({'checkpoint':str(checkpoint_path),'best_epoch':best_epoch,'best_validation_contrastive_loss':best_loss,'run_dir':str(run),'train_clips':len(train_emb),'validation_clips':len(val_emb),'threshold':threshold,'test_labels_read':False,'training_performed':False},indent=2))

if __name__=='__main__': main()
