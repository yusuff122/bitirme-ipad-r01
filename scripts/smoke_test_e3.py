"""E3 structural/gradient smoke test on one cached sequence per split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.models.temporal_memory import TemporalMemoryPredictor, memory_losses
from src.training.train_temporal_gru import TemporalWindowDataset, map_clip_scores_to_frames, validate_e1_cache
from src.training.train_temporal_memory import e3_clip_scores, windows_per_split
from src.utils.device import get_device
from src.utils.seed import set_seed


def main():
    root=Path(__file__).resolve().parents[1]; cfg=yaml.safe_load((root/'configs/temporal_memory.yaml').read_text()); manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split'])
    e1=root/'outputs/reports/e1_diagnostics/source/E1_video_swin16_20261022_090120'; e1=e1 if e1.is_dir() else Path(cfg['e1_run_dir'])
    e1cfg,splits,clips,emb=validate_e1_cache(e1,manifest); print('Reused E1 cache:',e1,'weights:',e1cfg['weights_name'])
    counts={k:windows_per_split(clips[c],8) for k,c in [('train','train'),('validation','validation'),('test','test')]}; print('Full windows:',counts); assert counts=={'train':1263,'validation':329,'test':762}
    one={}
    for split,manifest_split,key in [('train','train','train'),('validation','val','validation'),('test','test','test')]:
        seq=sorted(splits[manifest_split].sequence.unique())[0]; ids=[i for i,c in enumerate(clips[key]) if c['sequence']==seq]; one[split]=(seq,ids)
    seq,ids=one['train']; ds=TemporalWindowDataset(emb['train'][ids],[clips['train'][i] for i in ids],8); print('One-sequence windows:',{k:max(0,len(v[1])-8) for k,v in one.items()})
    x=torch.stack([ds[i][0] for i in range(min(32,len(ds)))]); y=torch.stack([ds[i][1] for i in range(min(32,len(ds)))])
    set_seed(cfg['seed']); device=get_device(); model=TemporalMemoryPredictor(768,256,64,256).to(device); x=x.to(device); y=y.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
    out=model(x); pred,comp,div=memory_losses(out,y,model.memory_slots); total=pred+.01*comp+.001*div
    att=out['attention']; print('Shapes:',{'input':tuple(x.shape),'GRU hidden':tuple(out['hidden'].shape),'memory slots':tuple(out['memory_slots'].shape),'attention':tuple(att.shape),'retrieved memory':tuple(out['memory'].shape),'prediction':tuple(out['prediction'].shape)})
    print('Attention min/max:',float(att.detach().min()),float(att.detach().max()),'sum min/max:',float(att.detach().sum(1).min()),float(att.detach().sum(1).max())); print('Losses:',{'prediction':float(pred.detach()),'compactness':float(comp.detach()),'diversity':float(div.detach()),'total':float(total.detach())})
    assert torch.isfinite(total) and torch.allclose(att.sum(1),torch.ones(len(att),device=device),atol=1e-6)
    initial=float(total.detach()); opt_losses=[]
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True); out=model(x); p,c,d=memory_losses(out,y,model.memory_slots); loss=p+.01*c+.001*d; loss.backward()
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),f'No finite gradient: {name}'
        assert model.memory_slots.grad.abs().sum()>0
        optimizer.step(); opt_losses.append(float(loss.detach()))
    with torch.no_grad(): out=model(x); p,c,d=memory_losses(out,y,model.memory_slots); final=float((p+.01*c+.001*d))
    print('Smoke training total loss:',initial,'->',final,'(12 steps)'); assert final<initial
    for split,key,embedding_key in [('train','train','train'),('validation','validation','validation')]:
        q,subids=one[split]; subclips=[clips[key][i] for i in subids]; dsx=TemporalWindowDataset(emb[embedding_key][subids],subclips,8)
        assert all(all(dsx.clips[j]['sequence']==s for j in cids+[target]) for s,cids,target in dsx.windows)
    val_scores=e3_clip_scores(model,emb['validation'],clips['validation'],8,device); val_frames=map_clip_scores_to_frames(clips['validation'],val_scores); assert len(val_frames)==1612
    val_threshold=float(np.quantile(val_frames.anomaly_score,.995)); test_z=torch.load(e1/'embeddings/test_clip_embeddings.pt',map_location='cpu',weights_only=True).float(); test_scores=e3_clip_scores(model,test_z,clips['test'],8,device); test_frames=map_clip_scores_to_frames(clips['test'],test_scores)
    assert len(test_frames)==3685 and not test_frames.duplicated(['split','sequence','frame_name']).any() and test_frames.anomaly_score.notna().all()
    result={'cache_reused':True,'window_counts':counts,'one_sequence_windows':{k:max(0,len(v[1])-8) for k,v in one.items()},'parameter_count':sum(p.numel() for p in model.parameters()),'input_shape':list(x.shape),'hidden_shape':list(out['hidden'].shape),'memory_slots_shape':list(out['memory_slots'].shape),'attention_shape':list(att.shape),'prediction_shape':list(out['prediction'].shape),'attention_sum_min':float(att.detach().sum(1).min()),'attention_sum_max':float(att.detach().sum(1).max()),'prediction_loss':float(p.detach()),'compactness_loss':float(c.detach()),'diversity_loss':float(d.detach()),'total_loss_initial':initial,'total_loss_final':final,'all_parameter_gradients_finite':True,'sequence_boundaries_preserved':True,'validation_frames':len(val_frames),'validation_threshold_smoke':val_threshold,'test_frames_label_free':len(test_frames),'test_labels_loaded':False,'full_training_started':False}
    outpath=root/'outputs/reports/e3_smoke_test.json'; outpath.parent.mkdir(parents=True,exist_ok=True); outpath.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2)); print('E3 smoke test PASSED; no full run and no test labels loaded.')

if __name__=='__main__': main()
