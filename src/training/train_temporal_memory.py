"""Train E3 temporal-memory predictor on normal E1 embeddings only."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.models.temporal_memory import TemporalMemoryPredictor, memory_losses
from src.training.train_temporal_gru import TemporalWindowDataset, clip_score_table, map_clip_scores_to_frames, validate_e1_cache
from src.utils.device import get_device
from src.utils.seed import set_seed


def windows_per_split(clips, context):
    counts=pd.Series([c['sequence'] for c in clips]).value_counts()
    return int(sum(max(0,int(n)-context) for n in counts))


def e3_clip_scores(model, embeddings, clips, context, device, batch_size=128, diagnostics_path=None):
    grouped={}
    for i,c in enumerate(clips): grouped.setdefault(c['sequence'],[]).append((i,c))
    scores=np.full(len(clips),np.nan); diagnostics=[]; model.eval()
    for seq,items in sorted(grouped.items()):
        items.sort(key=lambda x:int(x[1]['start_frame'])); ids=[i for i,_ in items]
        if len(ids)<=context: raise ValueError(f'{seq} has no prediction beyond context')
        contexts=[]; targets=[]; target_ids=[]
        for pos in range(context,len(ids)):
            contexts.append(embeddings[ids[pos-context:pos]]); targets.append(embeddings[ids[pos]]); target_ids.append(ids[pos])
        with torch.inference_mode():
            for st in range(0,len(contexts),batch_size):
                x=torch.stack(contexts[st:st+batch_size]).to(device); y=torch.stack(targets[st:st+batch_size]).to(device)
                out=model(x); mse=((out['prediction']-y)**2).mean(dim=1); cos=1-torch.nn.functional.cosine_similarity(out['prediction'],y,dim=1)
                ent=-(out['attention']*out['attention'].clamp_min(1e-12).log()).sum(dim=1); maxatt=out['attention'].max(dim=1).values
                sl=slice(st,st+len(x)); scores[target_ids[sl]]=mse.cpu().numpy()
                att=out['attention'].cpu().numpy()
                for j,idx in enumerate(target_ids[sl]):
                    row={'clip_index':idx,'sequence':seq,'start_frame':clips[idx]['start_frame'],'end_frame':clips[idx]['end_frame'],'prediction_mse':float(mse[j]),'cosine_distance':float(cos[j]),'memory_entropy':float(ent[j]),'max_memory_attention':float(maxatt[j])}
                    row.update({f'attention_slot_{k:02d}':float(v) for k,v in enumerate(att[j])}); diagnostics.append(row)
        scores[ids[:context]]=scores[ids[context]]
        # Mirror the documented deterministic early-clip policy in diagnostics too.
        by_id={r['clip_index']:r for r in diagnostics}
        first=by_id[ids[context]]
        for idx in ids[:context]:
            row=dict(first); row.update({'clip_index':idx,'sequence':seq,'start_frame':clips[idx]['start_frame'],'end_frame':clips[idx]['end_frame']}); diagnostics.append(row)
    if not np.isfinite(scores).all(): raise ValueError('E3 produced missing/nonfinite clip scores')
    if diagnostics_path: pd.DataFrame(diagnostics).to_csv(diagnostics_path,index=False)
    return scores


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/temporal_memory.yaml'); p.add_argument('--e1-run-dir'); p.add_argument('--manifest'); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int)
    a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); e1=Path(a.e1_run_dir or cfg['e1_run_dir']); manifest_path=Path(a.manifest or cfg['manifest'])
    manifest=pd.read_csv(manifest_path,usecols=['path','sequence','frame_name','frame_index','split'])
    if manifest.isna().any().any() or manifest.duplicated(['split','sequence','frame_name']).any(): raise ValueError('Manifest metadata invalid or duplicate keys')
    e1cfg,splits,clips,emb=validate_e1_cache(e1,manifest)
    context=int(cfg['context_length']); train_ds=TemporalWindowDataset(emb['train'],clips['train'],context); val_ds=TemporalWindowDataset(emb['validation'],clips['validation'],context)
    counts={k:windows_per_split(clips[c],context) for k,c in [('train','train'),('validation','validation'),('test','test')]}
    if counts!={'train':1263,'validation':329,'test':762}: raise ValueError(f'Unexpected E3 window counts: {counts}')
    set_seed(int(cfg['seed'])); device=get_device(); model=TemporalMemoryPredictor(768,cfg['hidden_dim'],cfg['num_memory_slots'],cfg['memory_dim']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=3,factor=.5)
    train_loader=DataLoader(train_ds,batch_size=a.batch_size or cfg['batch_size'],shuffle=True,num_workers=0); val_loader=DataLoader(val_ds,batch_size=a.batch_size or cfg['batch_size'],shuffle=False,num_workers=0)
    best=float('inf'); best_state=None; stale=0; history=[]
    for epoch in range(1,(a.epochs or cfg['epochs'])+1):
        model.train(); tr=[]
        for x,y,_,_ in train_loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); out=model(x); pred,comp,div=memory_losses(out,y,model.memory_slots)
            loss=pred+cfg['lambda_compact']*comp+cfg['lambda_diversity']*div
            if not torch.isfinite(loss): raise ValueError('Nonfinite E3 training loss')
            loss.backward(); optimizer.step(); tr.append((float(pred.detach()),float(comp.detach()),float(div.detach()),float(loss.detach())))
        model.eval(); va=[]
        with torch.inference_mode():
            for x,y,_,_ in val_loader:
                out=model(x.to(device)); pred,_,_=memory_losses(out,y.to(device),model.memory_slots); va.append(float(pred))
        val_loss=float(np.mean(va)); scheduler.step(val_loss); vals=np.mean(tr,axis=0).tolist()
        row={'epoch':epoch,'train_prediction_mse':vals[0],'train_compactness':vals[1],'train_diversity':vals[2],'train_total_loss':vals[3],'validation_prediction_mse':val_loss,'learning_rate':optimizer.param_groups[0]['lr']}; history.append(row)
        print(f"Epoch {epoch:02d}: train total={vals[3]:.8f}, validation prediction MSE={val_loss:.8f}")
        if val_loss<best:
            best=val_loss; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=cfg['patience']: print('Early stop; best validation prediction loss:',best); break
    if best_state is None: raise RuntimeError('No best E3 checkpoint')
    model.load_state_dict(best_state); model.to(device).eval(); val_scores=e3_clip_scores(model,emb['validation'],clips['validation'],context,device)
    val_frames=map_clip_scores_to_frames(clips['validation'],val_scores); expected=splits['val'][['sequence','frame_name','frame_index']]
    merged=expected.merge(val_frames,on=['sequence','frame_name','frame_index'],how='outer',validate='one_to_one',indicator=True)
    if len(merged)!=1612 or not (merged._merge=='both').all(): raise ValueError('Validation frame mapping mismatch')
    frame_scores=merged.anomaly_score.to_numpy(); threshold=float(np.quantile(frame_scores,cfg['threshold_quantile']))
    now=datetime.now().astimezone(); run=Path(cfg['output_root'])/f"E3_memory_{now.strftime('%Y%m%d_%H%M%S')}"; run.mkdir(parents=True,exist_ok=False); (run/'figures').mkdir()
    outcfg={**cfg,'e1_run_dir':str(e1.resolve()),'manifest':str(manifest_path.resolve()),'model':'TemporalMemoryPredictor','input_dim':768,'num_layers':1,'device':str(device),'timestamp':now.isoformat(),'train_sequences':sorted(splits['train'].sequence.unique()),'validation_sequences':sorted(splits['val'].sequence.unique()),'test_sequences':sorted(splits['test'].sequence.unique()),'train_temporal_windows':counts['train'],'validation_temporal_windows':counts['validation'],'test_temporal_windows':counts['test'],'num_validation_frames':1612,'num_test_frames':len(splits['test']),'e1_embeddings_reused':True,'cache_metadata_validated':True,'best_validation_prediction_mse':best,'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad)}
    torch.save({'state_dict':best_state,'config':outcfg},run/'model.pt'); torch.save(best_state['memory_slots'],run/'memory_slots.pt'); pd.DataFrame(history).to_csv(run/'training_history.csv',index=False); pd.DataFrame(clips['validation']).assign(anomaly_score=val_scores).to_csv(run/'validation_clip_scores.csv',index=False)
    info={'threshold':threshold,'method':'validation frame-score quantile','quantile':cfg['threshold_quantile'],'score_granularity':'frame scores after mean clip coverage propagation','num_validation_frames':len(frame_scores),'validation_score_mean':float(frame_scores.mean()),'validation_score_std':float(frame_scores.std()),'validation_score_min':float(frame_scores.min()),'validation_score_max':float(frame_scores.max()),'test_used_for_threshold':False}; (run/'threshold.json').write_text(json.dumps(info,indent=2)); (run/'config.json').write_text(json.dumps(outcfg,indent=2))
    (run/'run_summary.md').write_text(f"# E3 temporal memory\n\nFrozen E1 embeddings; GRU({cfg['hidden_dim']}) + {cfg['num_memory_slots']} prototypes; context {context}. Train/validation/test windows: {counts}. Best validation prediction MSE: {best:.8g}. Threshold is validation normal frame-score q={cfg['threshold_quantile']}. Test labels are not read during training.\n")
    print(json.dumps({'run_dir':str(run),'parameter_count':outcfg['trainable_parameters'],'threshold':threshold,'window_counts':counts,'test_labels_read':False},indent=2))

if __name__=='__main__': main()
