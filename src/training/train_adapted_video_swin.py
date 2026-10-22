"""E4 normal-only temporal contrastive adaptation of pretrained Video Swin."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.temporal_pairs import DistinctSequenceBatchSampler, TemporalPositivePairDataset
from src.data.video_loader import R01VideoClipDataset, collate_video_clips, map_clip_scores_to_frames
from src.models.adapted_video_swin import AdaptedVideoSwin, WEIGHTS_NAME, info_nce_loss
from src.models.frame_baseline import knn_scores
from src.training.train_temporal_gru import validate_e1_cache
from src.utils.device import get_device
from src.utils.seed import set_seed


def extract_embeddings(records,model,transform,device,cfg,name):
    ds=R01VideoClipDataset(records,transform,cfg['clip_length'],cfg['stride']); loader=DataLoader(ds,batch_size=int(cfg.get('extract_batch_size',2)),shuffle=False,num_workers=cfg['num_workers'],collate_fn=collate_video_clips)
    vectors=[]; metas=[]; model.eval()
    with torch.inference_mode():
        for videos,meta in loader:
            videos=videos.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=bool(cfg['mixed_precision'] and device.type=='cuda')):
                features,_=model(videos)
            features=torch.nn.functional.normalize(features.float(),dim=1)
            norms=features.norm(dim=1)
            if not torch.isfinite(features).all() or not torch.allclose(norms,torch.ones_like(norms),atol=1e-4): raise RuntimeError(f'{name} embedding check failed')
            vectors.append(features.cpu()); metas.extend(meta)
    if not vectors: raise ValueError(f'No clips in {name}')
    return torch.cat(vectors),metas


def validate_split_manifest(manifest):
    required={'path','sequence','frame_name','frame_index','split'}
    if not required.issubset(manifest.columns): raise ValueError(f'Missing manifest fields: {required-set(manifest.columns)}')
    if manifest[list(required)].isna().any().any() or manifest.duplicated(['split','sequence','frame_name']).any(): raise ValueError('Manifest metadata is incomplete or duplicated')
    splits={k:manifest[manifest.split==k].copy() for k in ('train','val','test')}
    counts={k:len(v.sequence.unique()) for k,v in splits.items()}
    if counts!={'train':27,'val':7,'test':15}: raise ValueError(f'Unexpected sequence split: {counts}')
    if any(not s.startswith('training/') for k in ('train','val') for s in splits[k].sequence.unique()): raise ValueError('Only normal training/ sequences may be adapted')
    if any(not s.startswith('testing/') for s in splits['test'].sequence.unique()): raise ValueError('Test metadata paths are not official testing sequences')
    return splits


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/video_swin_adapted.yaml'); p.add_argument('--manifest'); p.add_argument('--e1-run-dir'); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8')); cfg['manifest']=a.manifest or cfg['manifest']; e1=Path(a.e1_run_dir or cfg['e1_run_dir']); cfg['batch_size']=a.batch_size or cfg['batch_size']
    if cfg['weights_name']!=WEIGHTS_NAME or cfg['model_name']!='swin3d_t' or cfg['pretrained'] is not True: raise ValueError('E4 requires pretrained official KINETICS400 Video Swin Tiny')
    manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split']); splits=validate_split_manifest(manifest); _,_,cached_clips,_=validate_e1_cache(e1,manifest)
    set_seed(cfg['seed']); device=get_device(); model=AdaptedVideoSwin(pretrained=True).to(device); transform=model.make_transform()
    train_ds=TemporalPositivePairDataset(splits['train'],transform,cfg['clip_length'],cfg['stride'],cfg['positive_max_offset'],True,cfg['seed']); val_ds=TemporalPositivePairDataset(splits['val'],transform,cfg['clip_length'],cfg['stride'],cfg['positive_max_offset'],False,cfg['seed'])
    train_sampler=DistinctSequenceBatchSampler(train_ds,cfg['batch_size'],True,cfg['seed']); val_sampler=DistinctSequenceBatchSampler(val_ds,cfg['batch_size'],False,cfg['seed'])
    train_loader=DataLoader(train_ds,batch_sampler=train_sampler,num_workers=cfg['num_workers']); val_loader=DataLoader(val_ds,batch_sampler=val_sampler,num_workers=cfg['num_workers'])
    trainable=[p for p in model.parameters() if p.requires_grad]; frozen=[p for p in model.parameters() if not p.requires_grad]
    backbone_params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('projection_head.')]; projection_params=list(model.projection_head.parameters())
    optimizer=torch.optim.AdamW([{'params':backbone_params,'lr':cfg['backbone_learning_rate']},{'params':projection_params,'lr':cfg['projection_learning_rate']}],weight_decay=cfg['weight_decay'])
    now=datetime.now().astimezone(); run=Path(cfg['output_root'])/f"E4_adapted_swin_{now.strftime('%Y%m%d_%H%M%S')}"; run.mkdir(parents=True,exist_ok=False); (run/'figures').mkdir()
    checkpoint_cfg={**cfg,'e1_run_dir':str(e1.resolve()),'manifest':str(Path(cfg['manifest']).resolve()),'device':str(device),'weights_name':WEIGHTS_NAME,'embedding_dim':768,'projection_dim':128,'trainable_parameters':sum(x.numel() for x in trainable),'frozen_parameters':sum(x.numel() for x in frozen),'trainable_parameter_names':model.trainable_parameter_names(),'train_clip_count':len(train_ds),'validation_clip_count':len(val_ds),'test_labels_read':False}
    (run/'config.json').write_text(json.dumps(checkpoint_cfg,indent=2),encoding='utf-8')
    scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['mixed_precision'] and device.type=='cuda')); best=float('inf'); best_state=None; best_epoch=None; stale=0; history=[]
    for epoch in range(1,(a.epochs or cfg['epochs'])+1):
        train_ds.set_epoch(epoch); train_sampler.set_epoch(epoch); model.train(); losses=[]
        for xa,xb,_seq,_positive_idx in train_loader:
            if len(xa)<2: continue
            # weights transform returns B,T,C,H,W; torchvision Swin expects B,C,T,H,W.
            videos=torch.stack([xa,xb],dim=1).flatten(0,1).to(device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=bool(cfg['mixed_precision'] and device.type=='cuda')):
                _,proj=model(videos); proj=proj.reshape(len(xa),2,-1); loss,logits=info_nce_loss(proj,cfg['temperature'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite E4 training loss')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses=[]
        with torch.inference_mode():
            for xa,xb,_seq,_positive_idx in val_loader:
                if len(xa)<2: continue
                videos=torch.stack([xa,xb],dim=1).flatten(0,1).to(device)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=bool(cfg['mixed_precision'] and device.type=='cuda')):
                    _,proj=model(videos); loss,_=info_nce_loss(proj.reshape(len(xa),2,-1),cfg['temperature'])
                val_losses.append(float(loss.cpu()))
        if not losses or not val_losses: raise ValueError('No valid contrastive batches; increase pair batch size or check split')
        tr=float(np.mean(losses)); va=float(np.mean(val_losses)); history.append({'epoch':epoch,'train_contrastive_loss':tr,'validation_contrastive_loss':va}); print(f'Epoch {epoch:02d}: train InfoNCE={tr:.6f}; validation InfoNCE={va:.6f}')
        if va<best:
            best=va; best_epoch=epoch; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':best_state,'config':checkpoint_cfg,'best_epoch':best_epoch,'best_validation_contrastive_loss':best},run/'best_checkpoint.pt')
        else:
            stale+=1
            if stale>=cfg['patience']: print('Early stop; best validation contrastive loss:',best); break
    if best_state is None: raise RuntimeError('No E4 best checkpoint')
    model.load_state_dict(best_state); model.to(device).eval()
    train_emb,train_clips=extract_embeddings(splits['train'],model,transform,device,cfg,'train'); val_emb,val_clips=extract_embeddings(splits['val'],model,transform,device,cfg,'validation')
    for name,actual,cached in [('train',train_clips,cached_clips['train']),('validation',val_clips,cached_clips['validation'])]:
        keys=('sequence','start_frame','end_frame','frame_names','frame_indices')
        if len(actual)!=len(cached) or any(any(actual[i][k]!=cached[i][k] for k in keys) for i in range(len(actual))): raise ValueError(f'{name} adapted feature clip indexing differs from validated E1 cache')
    val_clip_scores=knn_scores(val_emb,train_emb,cfg['knn_k']).numpy(); val_frame_rows=map_clip_scores_to_frames(val_clips,val_clip_scores,splits['val']); val_frames=pd.DataFrame(val_frame_rows)
    if len(val_frames)!=1612: raise ValueError(f'Expected 1612 validation frames, got {len(val_frames)}')
    threshold=float(np.quantile(val_frames.anomaly_score,cfg['threshold_quantile']))
    outcfg={**cfg,'e1_run_dir':str(e1.resolve()),'manifest':str(Path(cfg['manifest']).resolve()),'device':str(device),'timestamp':now.isoformat(),'best_epoch':best_epoch,'train_sequences':sorted(splits['train'].sequence.unique()),'validation_sequences':sorted(splits['val'].sequence.unique()),'test_sequences':sorted(splits['test'].sequence.unique()),'train_clip_count':len(train_emb),'validation_clip_count':len(val_emb),'test_clip_count':882,'num_test_frames':3685,'embedding_dim':768,'projection_dim':128,'knn_k':cfg['knn_k'],'trainable_parameters':sum(x.numel() for x in trainable),'frozen_parameters':sum(x.numel() for x in frozen),'trainable_parameter_names':model.trainable_parameter_names(),'test_labels_read':False,'e1_cache_metadata_validated':True}
    torch.save({'state_dict':best_state,'config':outcfg,'best_epoch':best_epoch,'best_validation_contrastive_loss':best},run/'model.pt'); torch.save({k:v for k,v in best_state.items() if k.startswith('projection_head.')},run/'projection_head.pt'); torch.save(train_emb,run/'adapted_train_embeddings.pt'); torch.save(val_emb,run/'adapted_validation_embeddings.pt'); torch.save(train_emb,run/'memory_bank.pt'); pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
    pd.DataFrame(val_clips).assign(anomaly_score=val_clip_scores).to_csv(run/'clip_scores.csv',index=False); (run/'validation_clips.json').write_text(json.dumps(val_clips),encoding='utf-8')
    q=cfg['threshold_quantile']; threshold_info={'threshold':threshold,'method':'validation normal frame-score quantile','quantile':q,'score_granularity':'frame-level after mean clip coverage aggregation','num_validation_frames':len(val_frames),'validation_score_mean':float(val_frames.anomaly_score.mean()),'validation_score_std':float(val_frames.anomaly_score.std()),'validation_score_min':float(val_frames.anomaly_score.min()),'validation_score_max':float(val_frames.anomaly_score.max()),'test_used_for_threshold':False}; (run/'threshold.json').write_text(json.dumps(threshold_info,indent=2)); (run/'config.json').write_text(json.dumps(outcfg,indent=2))
    (run/'run_summary.md').write_text(f"# E4 adapted Video Swin\n\n- Pretrained: {WEIGHTS_NAME}; only final stage, final norm and projection head trained.\n- Train clips: {len(train_emb)}; validation clips: {len(val_emb)}.\n- Positive pairs: same sequence, temporal offset 1–{cfg['positive_max_offset']} clips; negatives are other sequences in each batch.\n- Best validation contrastive loss: {best:.8f}.\n- Threshold: validation normal frame scores q={q}, {threshold:.8f}. Test labels read: false.\n")
    print(json.dumps({'run_dir':str(run),'train_clips':len(train_emb),'validation_clips':len(val_emb),'test_clips':882,'trainable_parameters':outcfg['trainable_parameters'],'frozen_parameters':outcfg['frozen_parameters'],'best_validation_contrastive_loss':best,'threshold':threshold,'test_labels_read':False},indent=2))

if __name__=='__main__': main()
