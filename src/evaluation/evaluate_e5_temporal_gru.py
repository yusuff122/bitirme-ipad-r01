"""Final E5 evaluation on official test labels; labels are used only here."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluate_frame import exact_test_alignment
from src.evaluation.metrics import calculate_metrics
from src.models.temporal_gru import GRUTemporalPredictor
from src.training.e5_artifacts import load_e4_metadata_and_embeddings
from src.training.train_temporal_gru import clip_score_table, map_clip_scores_to_frames
from src.utils.device import get_device


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--e4-run-dir'); p.add_argument('--manifest'); a=p.parse_args()
    run=Path(a.run_dir); cfg=json.loads((run/'config.json').read_text(encoding='utf-8')); e4=a.e4_run_dir or cfg['e4_run_dir']; manifest_path=a.manifest or cfg['manifest']
    # Final test labels are read in this evaluator only, never by E5 training/threshold code.
    manifest=pd.read_csv(manifest_path); test=manifest[manifest.split=='test'].copy()
    if len(test)!=3685 or test.label.isna().any() or not set(test.label.astype(int).unique()).issubset({0,1}): raise ValueError('Expected 3685 labeled official test frames')
    cache=load_e4_metadata_and_embeddings(e4,manifest[['path','sequence','frame_name','frame_index','split']],include_test=True)
    clips=cache['clips']['test']; test_z=cache['test_embeddings']
    if len(clips)!=882: raise ValueError(f'Expected 882 test clip records, found {len(clips)}')
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True); device=get_device(); model=GRUTemporalPredictor(cfg['input_dim'],cfg['hidden_dim'],cfg['num_layers'],cfg['dropout']).to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    scores=clip_score_table(model,test_z,clips,cfg['context_length'],device); pd.DataFrame(clips).assign(anomaly_score=scores).to_csv(run/'clip_scores.csv',index=False)
    mapped=map_clip_scores_to_frames(clips,scores); pred=mapped[['split','sequence','frame_name','frame_index','anomaly_score','num_covering_clips']].copy(); threshold_info=json.loads((run/'threshold.json').read_text()); threshold=float(threshold_info['threshold'])
    pred['ground_truth']=pred.merge(test[['split','sequence','frame_name','label']],on=['split','sequence','frame_name'],how='left',validate='one_to_one').label.astype(int).to_numpy(); pred['threshold']=threshold; pred['prediction']=(pred.anomaly_score>=threshold).astype(int)
    expected=test[['split','sequence','frame_name','frame_index','label']].rename(columns={'label':'ground_truth'}); aligned=exact_test_alignment(pred,expected)
    if len(aligned)!=3685 or aligned.anomaly_score.isna().any(): raise ValueError('E5 test score alignment must contain exactly 3685 frames')
    aligned=aligned[['sequence','frame_name','frame_index','ground_truth','anomaly_score','threshold','prediction','num_covering_clips']]; aligned.to_csv(run/'predictions.csv',index=False)
    metrics=calculate_metrics(aligned.ground_truth,aligned.anomaly_score,threshold); metrics.update({'threshold':threshold,'threshold_method':'validation_normal_frame_q995','threshold_quantile':cfg['threshold_quantile'],'test_used_for_threshold':False,'num_test_frames':len(aligned),'num_test_clips':len(clips)}); (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    figdir=run/'figures'; figdir.mkdir(exist_ok=True); hist=pd.read_csv(run/'training_history.csv'); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist.epoch,hist.train_loss,label='Train MSE'); ax.plot(hist.epoch,hist.validation_loss,label='Validation MSE'); ax.set(xlabel='Epoch',ylabel='Next-embedding MSE',title='E5 Training'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,name,color in [(0,'Normal test','tab:blue'),(1,'Anomaly test','tab:red')]: ax.hist(aligned.loc[aligned.ground_truth==label,'anomaly_score'],bins=40,density=True,alpha=.55,label=name,color=color)
    ax.axvline(threshold,color='black',linestyle='--',label='Validation frame threshold'); ax.set(xlabel='Frame anomaly score (MSE)',ylabel='Density',title='E5 Test Score Distribution'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'score_distribution.png',dpi=160); plt.close(fig)
    cm=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],title='E5 Confusion Matrix')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figdir/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for seq,ax in zip(['testing/01','testing/03','testing/06','testing/08'],axes.flat):
        g=aligned[aligned.sequence==seq].sort_values('frame_index'); x=g.frame_index.to_numpy(); y=g.ground_truth.to_numpy(); ax.plot(x,g.anomaly_score,color='#244E8A'); ax.axhline(threshold,color='#D08000',linestyle='--'); starts=np.where((y==1)&np.r_[True,y[:-1]==0])[0]; ends=np.where((y==1)&np.r_[y[1:]==0,True])[0]
        for lo,hi in zip(starts,ends): ax.axvspan(x[lo],x[hi],color='#E76F51',alpha=.18)
        ax.set(title=seq,xlabel='Frame index',ylabel='Prediction MSE'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figdir/'timeline_examples.png',dpi=160); plt.close(fig)
    summary=(run/'run_summary.md').read_text(encoding='utf-8'); summary+='\n## Final test metrics\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n'; (run/'run_summary.md').write_text(summary,encoding='utf-8')
    print(json.dumps({'metrics':metrics,'run_dir':str(run),'test_clips':len(clips),'test_frames':len(aligned),'test_labels_read_here_only':True},indent=2))

if __name__=='__main__': main()
