"""Final E4 test evaluation; official labels are accessed only here."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.video_loader import R01VideoClipDataset, collate_video_clips, map_clip_scores_to_frames
from src.evaluation.evaluate_frame import exact_test_alignment
from src.evaluation.metrics import calculate_metrics
from src.models.adapted_video_swin import AdaptedVideoSwin
from src.models.frame_baseline import knn_scores
from src.training.train_temporal_gru import validate_e1_cache
from src.utils.device import get_device


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--manifest'); p.add_argument('--batch-size',type=int); a=p.parse_args(); run=Path(a.run_dir); cfg=json.loads((run/'config.json').read_text()); manifest_path=a.manifest or cfg['manifest']
    # This is the sole E4 command which reads test labels.
    manifest=pd.read_csv(manifest_path); test=manifest[manifest.split=='test'].copy()
    if len(test)!=3685 or test.label.isna().any() or not set(test.label.astype(int).unique()).issubset({0,1}): raise ValueError('Expected 3685 labeled test frames')
    validate_e1_cache(Path(cfg['e1_run_dir']),manifest[['path','sequence','frame_name','frame_index','split']])
    device=get_device(); model=AdaptedVideoSwin(pretrained=True).to(device); ckpt=torch.load(run/'model.pt',map_location='cpu',weights_only=True); model.load_state_dict(ckpt['state_dict']); model.eval()
    if cfg['weights_name']!='Swin3D_T_Weights.KINETICS400_V1': raise ValueError('E4 checkpoint did not start from expected Kinetics weights')
    records=test.drop(columns=['label']).copy(); ds=R01VideoClipDataset(records,model.make_transform(),cfg['clip_length'],cfg['stride']); loader=DataLoader(ds,batch_size=a.batch_size or 2,shuffle=False,num_workers=cfg['num_workers'],collate_fn=collate_video_clips); vectors=[]; clips=[]
    with torch.inference_mode():
        for videos,meta in loader:
            videos=videos.to(device)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=bool(cfg['mixed_precision'] and device.type=='cuda')): feats,_=model(videos)
            feats=torch.nn.functional.normalize(feats.float(),dim=1)
            norms=feats.norm(dim=1)
            if not torch.isfinite(feats).all() or not torch.allclose(norms,torch.ones_like(norms),atol=1e-4): raise ValueError('Test adapted embedding validation failed')
            vectors.append(feats.cpu()); clips.extend(meta)
    z=torch.cat(vectors)
    if len(z)!=882 or len(clips)!=882: raise ValueError(f'Expected 882 test clips, got features={len(z)}, metadata={len(clips)}')
    torch.save(z,run/'adapted_test_embeddings.pt'); memory=torch.load(run/'memory_bank.pt',map_location='cpu',weights_only=True).float(); scores=knn_scores(z,memory,cfg['knn_k']).numpy(); clip_df=pd.DataFrame(clips); clip_df['anomaly_score']=scores; clip_df.to_csv(run/'clip_scores.csv',index=False)
    frame=pd.DataFrame(map_clip_scores_to_frames(clips,scores,records)); pred=frame[['split','sequence','frame_name','frame_index','label','anomaly_score','num_covering_clips']].rename(columns={'label':'ground_truth'}); threshold=float(json.loads((run/'threshold.json').read_text())['threshold']); pred['threshold']=threshold; pred['prediction']=(pred.anomaly_score>=threshold).astype(int)
    aligned=exact_test_alignment(pred,test[['split','sequence','frame_name','frame_index','label']].rename(columns={'label':'ground_truth'}))
    if len(aligned)!=3685 or aligned.anomaly_score.isna().any(): raise ValueError(f'Frame mapping expected 3685 unique scored frames; got {len(aligned)}')
    aligned=aligned[['sequence','frame_name','frame_index','ground_truth','anomaly_score','threshold','prediction','num_covering_clips']]; aligned.to_csv(run/'predictions.csv',index=False); metrics=calculate_metrics(aligned.ground_truth,aligned.anomaly_score,threshold); metrics.update({'threshold':threshold,'threshold_quantile':cfg['threshold_quantile'],'test_used_for_threshold':False,'num_test_frames':len(aligned),'num_test_clips':len(z)}); (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
    figdir=run/'figures'; hist=pd.read_csv(run/'training_history.csv'); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist.epoch,hist.train_contrastive_loss,label='Train InfoNCE'); ax.plot(hist.epoch,hist.validation_contrastive_loss,label='Validation InfoNCE'); ax.set(xlabel='Epoch',ylabel='Loss',title='E4 Training'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for y,name in [(0,'Normal'),(1,'Anomaly')]: ax.hist(aligned.loc[aligned.ground_truth==y,'anomaly_score'],bins=40,density=True,alpha=.5,label=name)
    ax.axvline(threshold,color='black',linestyle='--',label='Validation threshold'); ax.set(xlabel='kNN distance',ylabel='Density',title='E4 test score distribution'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'score_distribution.png',dpi=160); plt.close(fig)
    cm=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],title='E4 confusion matrix')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figdir/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for seq,ax in zip(['testing/01','testing/03','testing/06','testing/08'],axes.flat):
        g=aligned[aligned.sequence==seq].sort_values('frame_index'); x=g.frame_index.to_numpy(); y=g.ground_truth.to_numpy(); ax.plot(x,g.anomaly_score); ax.axhline(threshold,color='orange',linestyle='--'); starts=np.where((y==1)&np.r_[True,y[:-1]==0])[0]; ends=np.where((y==1)&np.r_[y[1:]==0,True])[0]
        for lo,hi in zip(starts,ends): ax.axvspan(x[lo],x[hi],color='red',alpha=.15)
        ax.set(title=seq,xlabel='Frame',ylabel='kNN distance')
    fig.tight_layout(); fig.savefig(figdir/'timeline_examples.png',dpi=160); plt.close(fig)
    base={'E1':{'AUROC':.5932755208,'AUPR':.3846233176,'F1':.2723900371},'E4':{'AUROC':metrics['auroc'],'AUPR':metrics['aupr'],'F1':metrics['f1']}}
    report=Path('outputs/reports'); report.mkdir(parents=True,exist_ok=True); (report/'E1_vs_E4.json').write_text(json.dumps(base,indent=2)); (report/'E1_vs_E4.md').write_text('# E1 vs E4\n\n| Metric | E1 generic frozen Swin | E4 adapted Swin |\n|---|---:|---:|\n'+f"| AUROC | {base['E1']['AUROC']:.6f} | {metrics['auroc']:.6f} |\n| AUPR | {base['E1']['AUPR']:.6f} | {metrics['aupr']:.6f} |\n| F1 | {base['E1']['F1']:.6f} | {metrics['f1']:.6f} |\n")
    print(json.dumps({'metrics':metrics,'test_clips':len(z),'test_frames':len(aligned),'embedding_shape':list(z.shape)},indent=2))

if __name__=='__main__': main()
