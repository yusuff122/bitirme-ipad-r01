"""Small E4 pair/partial-fine-tuning gradient smoke test; no test labels read."""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.data.temporal_pairs import DistinctSequenceBatchSampler,TemporalPositivePairDataset
from src.models.adapted_video_swin import AdaptedVideoSwin,info_nce_loss
from src.training.train_adapted_video_swin import validate_split_manifest
from src.training.train_temporal_gru import validate_e1_cache
from src.utils.device import get_device
from src.utils.seed import set_seed


def main():
    cfg=yaml.safe_load((ROOT/'configs/video_swin_adapted.yaml').read_text()); manifest=pd.read_csv(cfg['manifest'],usecols=['path','sequence','frame_name','frame_index','split']); splits=validate_split_manifest(manifest); e1=ROOT/'outputs/reports/e1_diagnostics/source/E1_video_swin16_20261022_090120'; e1=e1 if e1.is_dir() else Path(cfg['e1_run_dir']); e1cfg,_,e1clips,_=validate_e1_cache(e1,manifest); print('E1 cache metadata validated and reused:',e1); print('Train/validation clip counts:',len(e1clips['train']),len(e1clips['validation']))
    set_seed(cfg['seed']); device=get_device(); model=AdaptedVideoSwin(True).to(device); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print('Trainable parameters:',trainable,'Frozen parameters:',frozen); print('Trainable layer names:',model.trainable_parameter_names())
    # Inspect pairs on one train and one validation sequence, strictly locally.
    train_seq=sorted(splits['train'].sequence.unique())[0]; val_seq=sorted(splits['val'].sequence.unique())[0]
    tr=TemporalPositivePairDataset(splits['train'][splits['train'].sequence==train_seq],model.make_transform(),16,4,3,True,cfg['seed']); va=TemporalPositivePairDataset(splits['val'][splits['val'].sequence==val_seq],model.make_transform(),16,4,3,False,cfg['seed'])
    xa,xb,seq,pos=tr[0]; assert seq==train_seq and tr.clips[pos]['sequence']==seq and 0<abs(tr.clips[pos]['start_frame']-tr.clips[0]['start_frame'])<=12
    vxa,vxb,vseq,vpos=va[0]; assert vseq==val_seq and va.clips[vpos]['sequence']==vseq
    print('Positive pair rule verified:',seq,tr.clips[0]['start_frame'],'->',tr.clips[pos]['start_frame'],'; validation sequence:',vseq)
    full=TemporalPositivePairDataset(splits['train'],model.make_transform(),16,4,3,True,cfg['seed']); sampler=DistinctSequenceBatchSampler(full,2,True,cfg['seed']); batch=next(iter(DataLoader(full,batch_sampler=sampler,num_workers=0)))
    ax,bx,seqs,positive_ids=batch; assert len(set(seqs))==2 and all(full.clips[int(i)]['sequence']==seqs[j] for j,i in enumerate(positive_ids)); print('Negative samples exist: distinct train sequences in batch:',list(seqs))
    videos=torch.stack([ax,bx],dim=1).flatten(0,1).to(device); assert videos.shape==(4,3,16,224,224); print('Backbone input B,C,T,H,W:',tuple(videos.shape))
    before={n:p.detach().clone() for n,p in model.named_parameters()}; allowed=[]; optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('projection_head.')],'lr':1e-5},{'params':model.projection_head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    losses=[]; model.eval()  # deterministic dropout/drop-path for measuring optimization on a fixed smoke batch
    for step in range(10):
        optimizer.zero_grad(set_to_none=True); features,proj=model(videos); pair_proj=proj.reshape(2,2,128); loss,logits=info_nce_loss(pair_proj,.07)
        if step==0:
            normalized_features=torch.nn.functional.normalize(features,dim=1); print('Feature/projection/logits shapes:',tuple(features.shape),tuple(proj.shape),tuple(logits.shape)); print('Adapted feature norm:',normalized_features[0].norm().item(),'projection norm:',proj[0].norm().item())
        assert features.shape==(4,768) and proj.shape==(4,128) and logits.shape==(2,2) and torch.isfinite(loss)
        loss.backward()
        for n,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),f'missing/nonfinite gradient {n}'; allowed.append(n)
            else: assert p.grad is None,f'frozen layer got gradient: {n}'
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    for n,p in model.named_parameters():
        if not p.requires_grad: assert torch.equal(p.detach(),before[n]),f'frozen layer changed: {n}'
    print('Ten-step InfoNCE loss:',losses,'; finite gradients only on allowed layers: OK; frozen weights unchanged: OK')
    vram=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else None; print('Peak CUDA VRAM GiB:',vram if vram is not None else 'not available on '+str(device))
    result={'cache_reused':True,'train_clip_count':len(e1clips['train']),'validation_clip_count':len(e1clips['validation']),'trainable_parameters':trainable,'frozen_parameters':frozen,'trainable_parameter_names':model.trainable_parameter_names(),'positive_pair_sequence':seq,'positive_pair_start_frames':[tr.clips[0]['start_frame'],tr.clips[pos]['start_frame']],'negative_sequences':list(seqs),'input_shape':list(videos.shape),'feature_dim':768,'projection_dim':128,'adapted_feature_norm':float(normalized_features[0].detach().norm()),'contrastive_logits_shape':[2,2],'smoke_losses':losses,'all_trainable_gradients_finite':True,'frozen_gradients_absent':True,'frozen_weights_unchanged':True,'peak_cuda_vram_gib':vram,'test_labels_read':False,'full_training_started':False}
    path=ROOT/'outputs/reports/e4_smoke_test.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)); print('E4 smoke test PASSED; no full training, test labels not read.')

if __name__=='__main__': main()
