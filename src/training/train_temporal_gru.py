"""Train E2 GRU predictor on normal E1 clip embeddings only."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.temporal_gru import GRUTemporalPredictor, next_embedding_mse
from src.utils.device import get_device
from src.utils.seed import set_seed


class TemporalWindowDataset(Dataset):
    """Build next-step windows within each sequence; never cross boundaries."""
    def __init__(self, embeddings, clips, context_length):
        self.embeddings = embeddings.float().cpu()
        self.clips = clips
        self.context_length = int(context_length)
        grouped = {}
        for index, clip in enumerate(clips):
            grouped.setdefault(clip['sequence'], []).append((index, clip))
        self.sequence_indices = {}
        for seq, items in grouped.items():
            items.sort(key=lambda x: int(x[1]['start_frame']))
            indices = [i for i, _ in items]
            self.sequence_indices[seq] = indices
        self.windows = []
        for seq, indices in sorted(self.sequence_indices.items()):
            for target_pos in range(self.context_length, len(indices)):
                self.windows.append((seq, indices[target_pos-self.context_length:target_pos], indices[target_pos]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        seq, context_ids, target_id = self.windows[index]
        return self.embeddings[context_ids], self.embeddings[target_id], seq, target_id


def load_config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def expected_clip_records(frame_records, clip_length, stride):
    expected=[]
    for sequence, group in frame_records.groupby('sequence', sort=True):
        rows=group.sort_values('frame_index')
        names=rows.frame_name.astype(str).tolist(); indices=rows.frame_index.astype(int).tolist()
        if indices != list(range(len(indices))):
            raise ValueError(f'Frame indices are not contiguous from zero in {sequence}')
        starts=list(range(0,max(0,len(indices)-clip_length+1),stride))
        # E1 appends a final aligned window so each sequence's last frame is covered.
        if len(indices)>=clip_length and (not starts or starts[-1]+clip_length<len(indices)):
            starts.append(len(indices)-clip_length)
        for start in starts:
            expected.append({'sequence':sequence,'start_frame':start,'end_frame':start+clip_length-1,
                             'frame_indices':indices[start:start+clip_length], 'frame_names':names[start:start+clip_length]})
    return expected


def verify_clip_index(frame_records, cached, clip_length, stride, name):
    expected=expected_clip_records(frame_records,clip_length,stride)
    if len(expected)!=len(cached):
        raise ValueError(f'{name} clip count mismatch: cache={len(cached)}, manifest expected={len(expected)}')
    for i,(a,b) in enumerate(zip(expected,cached)):
        for key in ('sequence','start_frame','end_frame','frame_indices','frame_names'):
            if a[key]!=b[key]: raise ValueError(f'{name} clip cache mismatch at row {i}, field={key}')


def validate_e1_cache(e1_dir, manifest):
    cfg=json.loads((e1_dir/'config.json').read_text(encoding='utf-8'))
    required={'model_name':'swin3d_t','weights_name':'Swin3D_T_Weights.KINETICS400_V1','clip_length':16,'stride':4,'embedding_dim':768}
    for key,value in required.items():
        if cfg.get(key)!=value: raise ValueError(f'E1 cache {key} mismatch: expected {value!r}, found {cfg.get(key)!r}')
    if cfg.get('pretrained') is not True or cfg.get('normalize_embeddings') is not True:
        raise ValueError('E1 cache must be pretrained and L2-normalized')
    splits={name:manifest[manifest.split==name].copy() for name in ('train','val','test')}
    seqs={name:sorted(df.sequence.unique().tolist()) for name,df in splits.items()}
    expected_counts={'train':27,'val':7,'test':15}
    for name,expected_count in expected_counts.items():
        if len(seqs[name])!=expected_count: raise ValueError(f'Expected {expected_count} {name} sequences, found {len(seqs[name])}')
    if any(not s.startswith('training/') for name in ('train','val') for s in seqs[name]):
        raise ValueError('Train/validation must contain only normal training sequences')
    if any(not s.startswith('testing/') for s in seqs['test']):
        raise ValueError('Test partition must contain only official testing sequences')
    declared={'train':cfg.get('train_sequences'),'val':cfg.get('validation_sequences'),'test':cfg.get('test_sequences')}
    for name in seqs:
        if seqs[name]!=declared[name]: raise ValueError(f'E1 {name} sequences do not match current manifest split')
    for meta_key,split_name in [('train_cache_metadata','train'),('validation_cache_metadata','val')]:
        meta=cfg.get(meta_key)
        if not isinstance(meta,dict): raise ValueError(f'Missing E1 {meta_key}')
        for key in ('model_name','weights_name','clip_length','stride','input_size'):
            if meta.get(key)!=cfg.get(key): raise ValueError(f'E1 {meta_key} mismatch for {key}')
        expected_meta_sequences=seqs[split_name]
        if sorted(meta.get('sequence_list',[]))!=expected_meta_sequences:
            raise ValueError(f'E1 {meta_key} sequence list mismatch')
    paths={name:e1_dir/'embeddings'/f'{name}_clips.json' for name in ('train','validation','test')}
    clips={name:json.loads(path.read_text(encoding='utf-8')) for name,path in paths.items()}
    verify_clip_index(splits['train'],clips['train'],16,4,'train')
    verify_clip_index(splits['val'],clips['validation'],16,4,'validation')
    verify_clip_index(splits['test'],clips['test'],16,4,'test')
    test_meta=json.loads((e1_dir/'embeddings'/'test_cache_metadata.json').read_text(encoding='utf-8'))
    for key in ('model_name','weights_name','clip_length','stride','input_size'):
        if test_meta.get(key)!=cfg.get(key): raise ValueError(f'Test cache metadata mismatch for {key}')
    if sorted(test_meta.get('sequence_list',[]))!=seqs['test']: raise ValueError('Test cache sequence metadata mismatch')
    tensors={
        'train':torch.load(e1_dir/'embeddings'/'train_embeddings.pt',map_location='cpu',weights_only=True).float(),
        'validation':torch.load(e1_dir/'embeddings'/'validation_embeddings.pt',map_location='cpu',weights_only=True).float(),
    }
    for name,key in [('train','train'),('validation','validation')]:
        z=tensors[key]
        if z.shape!=(len(clips[name]),768) or not torch.isfinite(z).all(): raise ValueError(f'Invalid {name} embedding cache shape/values: {tuple(z.shape)}')
        norms=torch.linalg.vector_norm(z,dim=1)
        if not torch.allclose(norms,torch.ones_like(norms),atol=1e-4): raise ValueError(f'{name} embedding cache not L2 normalized')
    return cfg,splits,clips,tensors


def clip_score_table(model, embeddings, clips, context_length, device, batch_size=128):
    """Produce one MSE score per clip; backfill first context with first valid score."""
    groups={}
    for i,clip in enumerate(clips): groups.setdefault(clip['sequence'],[]).append((i,clip))
    scores=np.full(len(clips),np.nan,dtype=np.float64); model.eval()
    for seq,items in sorted(groups.items()):
        items.sort(key=lambda x:int(x[1]['start_frame'])); ids=[i for i,_ in items]
        if len(ids)<=context_length: raise ValueError(f'{seq} has no prediction beyond context')
        contexts=[]; targets=[]; target_ids=[]
        for pos in range(context_length,len(ids)):
            contexts.append(embeddings[ids[pos-context_length:pos]])
            targets.append(embeddings[ids[pos]]); target_ids.append(ids[pos])
        with torch.inference_mode():
            for start in range(0,len(contexts),batch_size):
                x=torch.stack(contexts[start:start+batch_size]).to(device)
                y=torch.stack(targets[start:start+batch_size]).to(device)
                pred=model(x); err=((pred.float()-y.float())**2).mean(dim=1).cpu().numpy()
                scores[target_ids[start:start+batch_size]]=err
        scores[ids[:context_length]]=scores[ids[context_length]]
    if not np.isfinite(scores).all(): raise ValueError('Missing/nonfinite clip scores after early-score backfill')
    return scores


def map_clip_scores_to_frames(clips,scores):
    agg={}
    for clip,score in zip(clips,scores):
        for idx,name in zip(clip['frame_indices'],clip['frame_names']):
            key=(clip['sequence'],int(idx),str(name)); agg.setdefault(key,[]).append(float(score))
    rows=[]
    for (seq,idx,name),values in sorted(agg.items()):
        split='test' if seq.startswith('testing/') else ('val' if seq.startswith('training/') else 'train')
        rows.append({'split':split,'sequence':seq,'frame_index':idx,'frame_name':name,'anomaly_score':float(np.mean(values)),'num_covering_clips':len(values)})
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/temporal_gru.yaml'); p.add_argument('--e1-run-dir'); p.add_argument('--manifest'); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int)
    a=p.parse_args(); cfg=load_config(a.config); e1_dir=Path(a.e1_run_dir or cfg['e1_run_dir']); manifest_path=Path(a.manifest or cfg['manifest'])
    # Test labels are intentionally excluded from training-time manifest reads.
    manifest=pd.read_csv(manifest_path,usecols=['path','sequence','frame_name','frame_index','split'])
    if manifest.isna().any().any() or manifest.duplicated(['split','sequence','frame_name']).any(): raise ValueError('Manifest metadata invalid or duplicate keys')
    e1cfg,splits,clips,emb=validate_e1_cache(e1_dir,manifest)
    train_ds=TemporalWindowDataset(emb['train'],clips['train'],cfg['context_length'])
    val_ds=TemporalWindowDataset(emb['validation'],clips['validation'],cfg['context_length'])
    test_window_count=sum(max(0,n-int(cfg['context_length'])) for n in pd.Series([c['sequence'] for c in clips['test']]).value_counts())
    if not len(train_ds) or not len(val_ds): raise ValueError('Train/validation temporal windows empty')
    set_seed(int(cfg['seed'])); device=get_device()
    model=GRUTemporalPredictor(768,cfg['hidden_dim'],1,0.0).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    bs=a.batch_size or int(cfg['batch_size'])
    train_loader=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=0)
    max_epochs=a.epochs or int(cfg['epochs']); patience=int(cfg['patience']); best=float('inf'); stale=0; history=[]; best_state=None
    for epoch in range(1,max_epochs+1):
        model.train(); tr=[]
        for x,y,_,_ in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            loss=next_embedding_mse(model(x),y)
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite train loss at epoch {epoch}')
            loss.backward(); optimizer.step(); tr.append(float(loss.detach().cpu()))
        model.eval(); va=[]
        with torch.inference_mode():
            for x,y,_,_ in val_loader:
                loss=next_embedding_mse(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss): raise ValueError(f'Nonfinite validation loss at epoch {epoch}')
                va.append(float(loss.cpu()))
        train_loss=float(np.mean(tr)); val_loss=float(np.mean(va)); history.append({'epoch':epoch,'train_loss':train_loss,'validation_loss':val_loss})
        print(f'Epoch {epoch:02d}: train MSE={train_loss:.8f}; validation MSE={val_loss:.8f}')
        if val_loss<best:
            best=val_loss; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience: print('Early stop; best validation loss:',best); break
    if best_state is None: raise RuntimeError('No best checkpoint')
    model.load_state_dict(best_state); model.to(device).eval()
    val_clip_scores=clip_score_table(model,emb['validation'],clips['validation'],cfg['context_length'],device)
    val_clip_df=pd.DataFrame(clips['validation']); val_clip_df['anomaly_score']=val_clip_scores
    val_frames=map_clip_scores_to_frames(clips['validation'],val_clip_scores)
    expected=splits['val'][['sequence','frame_name','frame_index']]
    merged=expected.merge(val_frames,on=['sequence','frame_name','frame_index'],how='outer',validate='one_to_one',indicator=True)
    if not (merged._merge=='both').all() or len(merged)!=len(expected): raise ValueError('Validation frame mapping incomplete/extra')
    frame_scores=merged.anomaly_score.to_numpy(float); q=float(cfg['threshold_quantile']); threshold=float(np.quantile(frame_scores,q))
    now=datetime.now().astimezone(); run=Path(cfg.get('output_root','models'))/f"E2_gru_{now.strftime('%Y%m%d_%H%M%S')}"; run.mkdir(parents=True,exist_ok=False); (run/'figures').mkdir()
    outcfg={**cfg,'e1_run_dir':str(e1_dir.resolve()),'manifest':str(manifest_path.resolve()),'model':'GRUTemporalPredictor','input_dim':768,'num_layers':1,'dropout':0.0,'device':str(device),'timestamp':now.isoformat(),
            'train_sequences':sorted(splits['train'].sequence.unique()),'validation_sequences':sorted(splits['val'].sequence.unique()),'test_sequences':sorted(splits['test'].sequence.unique()),
            'train_clip_count':len(clips['train']),'validation_clip_count':len(clips['validation']),'test_clip_count':len(clips['test']),
            'train_temporal_windows':len(train_ds),'validation_temporal_windows':len(val_ds),'test_temporal_windows':test_window_count,
            'num_train_frames':len(splits['train']),'num_validation_frames':len(splits['val']),'num_test_frames':len(splits['test']),
            'e1_model_name':e1cfg['model_name'],'e1_weights_name':e1cfg['weights_name'],'e1_clip_length':e1cfg['clip_length'],'e1_stride':e1cfg['stride'],'e1_embedding_dim':e1cfg['embedding_dim'],'e1_embeddings_reused':True,'cache_metadata_validated':True}
    torch.save({'state_dict':best_state,'config':outcfg},run/'model.pt')
    pd.DataFrame(history).to_csv(run/'training_history.csv',index=False)
    val_clip_df[['sequence','start_frame','end_frame','anomaly_score']].to_csv(run/'validation_clip_scores.csv',index=False)
    threshold_info={'threshold':threshold,'method':'quantile','quantile':q,'score_granularity':'validation frame scores after mean aggregation of covering clip prediction errors','num_validation_frames':len(frame_scores),
                    'validation_score_mean':float(frame_scores.mean()),'validation_score_std':float(frame_scores.std()),'validation_score_min':float(frame_scores.min()),'validation_score_max':float(frame_scores.max()),
                    'validation_score_p95':float(np.quantile(frame_scores,.95)),'validation_score_p99':float(np.quantile(frame_scores,.99)),'validation_score_p995':float(np.quantile(frame_scores,.995)),
                    'test_used_for_threshold':False,'early_clip_policy':'backfill first context_length clip scores with first valid next-embedding MSE score'}
    (run/'threshold.json').write_text(json.dumps(threshold_info,indent=2)); (run/'config.json').write_text(json.dumps(outcfg,indent=2))
    (run/'run_summary.md').write_text(f"""# E2 GRU temporal predictor\n\n- Device: {device}\n- Reused E1 frozen embeddings: `{e1_dir}` ({e1cfg['weights_name']})\n- Train: {len(outcfg['train_sequences'])} normal sequences / {len(train_ds)} temporal windows\n- Validation: {len(outcfg['validation_sequences'])} held-out normal sequences / {len(val_ds)} temporal windows\n- Test metadata: {len(clips['test'])} clips / {len(splits['test'])} frames; labels were not read during training\n- GRU: input=768, hidden={cfg['hidden_dim']}, layers=1, context={cfg['context_length']}\n- Objective: predict next normalized clip embedding with MSE\n- Early stop: validation normal next-embedding MSE\n- Frame threshold: validation normal frame scores after mean clip coverage propagation, q={q}\n- Early clip policy: {threshold_info['early_clip_policy']}\n- Best validation loss: {best:.8f}\n\nFinal test evaluation: `python -m src.evaluation.evaluate_temporal_gru --run-dir {run}`\n""")
    import matplotlib.pyplot as plt
    hist=pd.DataFrame(history); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist.epoch,hist.train_loss,label='Train MSE'); ax.plot(hist.epoch,hist.validation_loss,label='Validation MSE'); ax.set(xlabel='Epoch',ylabel='Next-embedding MSE',title='E2 Training'); ax.legend(); fig.tight_layout(); fig.savefig(run/'figures'/'training_curve.png',dpi=160); plt.close(fig)
    print(json.dumps({'run_dir':str(run),'best_validation_loss':best,'train_temporal_windows':len(train_ds),'validation_temporal_windows':len(val_ds),'test_temporal_windows':test_window_count,'threshold':threshold,'test_labels_read':False},indent=2))


if __name__=='__main__': main()
