"""Final E3 test evaluation; test labels are read only here."""
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
from src.models.temporal_memory import TemporalMemoryPredictor
from src.training.train_temporal_gru import map_clip_scores_to_frames, validate_e1_cache
from src.training.train_temporal_memory import e3_clip_scores
from src.utils.device import get_device


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--e1-run-dir'); p.add_argument('--manifest'); a=p.parse_args()
    run=Path(a.run_dir); cfg=json.loads((run/'config.json').read_text()); e1=Path(a.e1_run_dir or cfg['e1_run_dir']); manifest_path=Path(a.manifest or cfg['manifest'])
    manifest=pd.read_csv(manifest_path); test=manifest[manifest.split=='test'].copy()
    if len(test)!=3685 or test.label.isna().any() or not set(test.label.astype(int).unique()).issubset({0,1}): raise ValueError('Expected 3685 labeled test frames')
    _,_,clips,_=validate_e1_cache(e1,manifest[['path','sequence','frame_name','frame_index','split']]); z=torch.load(e1/'embeddings'/'test_clip_embeddings.pt',map_location='cpu',weights_only=True).float()
    if z.shape!=(882,768) or not torch.isfinite(z).all() or not torch.allclose(torch.linalg.vector_norm(z,dim=1),torch.ones(882),atol=1e-4): raise ValueError('Invalid E1 test embedding cache')
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True); device=get_device(); model=TemporalMemoryPredictor(768,cfg['hidden_dim'],cfg['num_memory_slots'],cfg['memory_dim']).to(device); model.load_state_dict(checkpoint['state_dict']); model.eval()
    clip_scores=e3_clip_scores(model,z,clips['test'],cfg['context_length'],device,diagnostics_path=run/'memory_attention.csv'); clipdf=pd.DataFrame(clips['test']).copy(); clipdf['anomaly_score']=clip_scores; clipdf.to_csv(run/'clip_scores.csv',index=False)
    fr=map_clip_scores_to_frames(clips['test'],clip_scores); pred=fr[['split','sequence','frame_name','frame_index','anomaly_score','num_covering_clips']].copy(); threshold=float(json.loads((run/'threshold.json').read_text())['threshold'])
    pred['ground_truth']=pred.merge(test[['split','sequence','frame_name','label']],on=['split','sequence','frame_name'],how='left',validate='one_to_one').label.astype(int).to_numpy(); pred['threshold']=threshold; pred['prediction']=(pred.anomaly_score>=threshold).astype(int)
    aligned=exact_test_alignment(pred,test[['split','sequence','frame_name','frame_index','label']].rename(columns={'label':'ground_truth'}))
    if len(aligned)!=3685 or aligned.anomaly_score.isna().any(): raise ValueError('Expected exactly 3685 scored test frames')
    aligned=aligned[['sequence','frame_name','frame_index','ground_truth','anomaly_score','threshold','prediction','num_covering_clips']]; aligned.to_csv(run/'predictions.csv',index=False); metrics=calculate_metrics(aligned.ground_truth,aligned.anomaly_score,threshold); metrics.update({'threshold':threshold,'threshold_quantile':.995,'test_used_for_threshold':False,'num_test_frames':len(aligned),'num_test_clips':len(clips['test'])}); (run/'metrics.json').write_text(json.dumps(metrics,indent=2))
    figdir=run/'figures'; hist=pd.read_csv(run/'training_history.csv'); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist.epoch,hist.train_total_loss,label='Train total'); ax.plot(hist.epoch,hist.validation_prediction_mse,label='Validation prediction MSE'); ax.set(xlabel='Epoch',ylabel='Loss',title='E3 Training'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for y,name in [(0,'Normal'),(1,'Anomaly')]: ax.hist(aligned.loc[aligned.ground_truth==y,'anomaly_score'],bins=40,density=True,alpha=.5,label=name)
    ax.axvline(threshold,color='black',linestyle='--',label='Validation threshold'); ax.set(xlabel='Prediction MSE',ylabel='Density',title='E3 test score distribution'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'score_distribution.png',dpi=160); plt.close(fig)
    cm=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]]); fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],title='E3 confusion matrix')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figdir/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for seq,ax in zip(['testing/01','testing/03','testing/06','testing/08'],axes.flat):
        g=aligned[aligned.sequence==seq].sort_values('frame_index'); ax.plot(g.frame_index,g.anomaly_score); ax.axhline(threshold,color='orange',linestyle='--'); y=g.ground_truth.to_numpy(); xx=g.frame_index.to_numpy(); starts=np.where((y==1)&np.r_[True,y[:-1]==0])[0]; ends=np.where((y==1)&np.r_[y[1:]==0,True])[0]
        for lo,hi in zip(starts,ends): ax.axvspan(xx[lo],xx[hi],color='red',alpha=.15)
        ax.set(title=seq,xlabel='Frame',ylabel='Prediction MSE')
    fig.tight_layout(); fig.savefig(figdir/'timeline_examples.png',dpi=160); plt.close(fig)
    att=pd.read_csv(run/'memory_attention.csv'); cols=[f'attention_slot_{i:02d}' for i in range(cfg['num_memory_slots'])]; usage=att[cols].mean(axis=0).to_numpy()
    fig,ax=plt.subplots(figsize=(9,4)); ax.bar(np.arange(len(usage)),usage); ax.set(xlabel='Memory slot',ylabel='Mean attention weight',title='E3 memory usage'); fig.tight_layout(); fig.savefig(figdir/'memory_usage.png',dpi=160); plt.close(fig)
    import subprocess,sys
    subprocess.run([sys.executable,'scripts/compare_e0_e1_e2_e3.py','--e3-metrics',str(run/'metrics.json')],check=True)
    print(json.dumps({'metrics':metrics,'run_dir':str(run)},indent=2))

if __name__=='__main__': main()
