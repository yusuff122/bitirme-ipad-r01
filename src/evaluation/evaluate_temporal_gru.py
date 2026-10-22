"""Final E2 test evaluation; test labels are read only in this command."""
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
from src.training.train_temporal_gru import clip_score_table, map_clip_scores_to_frames, validate_e1_cache
from src.utils.device import get_device


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-dir',required=True); parser.add_argument('--e1-run-dir'); parser.add_argument('--manifest'); args=parser.parse_args()
    run=Path(args.run_dir); cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    e1=Path(args.e1_run_dir or cfg['e1_run_dir']); manifest_path=Path(args.manifest or cfg['manifest'])
    # Labels are accessed here, after training/threshold files are already fixed.
    manifest=pd.read_csv(manifest_path)
    test=manifest[manifest.split=='test'].copy()
    if len(test)!=int(cfg['num_test_frames']) or test.label.isna().any(): raise ValueError('Test frame count/labels differ from saved run metadata')
    if not set(test.label.astype(int).unique()).issubset({0,1}): raise ValueError('Test labels must be binary')
    e1_cfg,splits,clips,_=validate_e1_cache(e1,manifest[['path','sequence','frame_name','frame_index','split']])
    test_emb=torch.load(e1/'embeddings'/'test_clip_embeddings.pt',map_location='cpu',weights_only=True).float()
    if test_emb.shape!=(len(clips['test']),768) or not torch.isfinite(test_emb).all(): raise ValueError('Invalid test embedding cache shape/values')
    norms=torch.linalg.vector_norm(test_emb,dim=1)
    if not torch.allclose(norms,torch.ones_like(norms),atol=1e-4): raise ValueError('Test embeddings are not L2-normalized')
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    device=get_device(); model=GRUTemporalPredictor(768,cfg['hidden_dim'],cfg['num_layers'],cfg['dropout']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    clip_scores=clip_score_table(model,test_emb,clips['test'],cfg['context_length'],device)
    clip_df=pd.DataFrame(clips['test']); clip_df['anomaly_score']=clip_scores
    clip_df[['sequence','start_frame','end_frame','anomaly_score']].to_csv(run/'clip_scores.csv',index=False)
    frame_scores=map_clip_scores_to_frames(clips['test'],clip_scores)
    predictions=frame_scores[['split','sequence','frame_name','frame_index','anomaly_score','num_covering_clips']].copy()
    threshold_info=json.loads((run/'threshold.json').read_text(encoding='utf-8')); threshold=float(threshold_info['threshold'])
    predictions['ground_truth'] = predictions.merge(test[['split','sequence','frame_name','label']],on=['split','sequence','frame_name'],how='left',validate='one_to_one').label.astype(int).to_numpy()
    predictions['threshold']=threshold; predictions['prediction']=(predictions.anomaly_score>=threshold).astype(int)
    expected=test[['split','sequence','frame_name','frame_index','label']].rename(columns={'label':'ground_truth'})
    aligned=exact_test_alignment(predictions,expected)
    if len(aligned)!=3685: raise ValueError(f'Expected exactly 3685 test frame scores, got {len(aligned)}')
    aligned=aligned[['sequence','frame_name','frame_index','ground_truth','anomaly_score','threshold','prediction','num_covering_clips']]
    aligned.to_csv(run/'predictions.csv',index=False)
    metrics=calculate_metrics(aligned.ground_truth,aligned.anomaly_score,threshold)
    metrics.update({'threshold':threshold,'threshold_method':threshold_info['method'],'threshold_quantile':threshold_info['quantile'],'num_test_frames':len(aligned),'num_test_clips':len(clips['test']),'test_used_for_threshold':False,
                    'fpr':float(metrics['fp']/(metrics['fp']+metrics['tn'])) if metrics['fp']+metrics['tn'] else 0.0,
                    'fnr':float(metrics['fn']/(metrics['fn']+metrics['tp'])) if metrics['fn']+metrics['tp'] else 0.0})
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    figdir=run/'figures'; figdir.mkdir(exist_ok=True)
    hist=pd.read_csv(run/'training_history.csv'); fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist.epoch,hist.train_loss,label='Train MSE'); ax.plot(hist.epoch,hist.validation_loss,label='Validation MSE'); ax.set(xlabel='Epoch',ylabel='Next-embedding MSE',title='E2 Training Curve'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curve.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for label,name,color in [(0,'Normal test','tab:blue'),(1,'Anomaly test','tab:red')]:
        values=aligned.loc[aligned.ground_truth==label,'anomaly_score']; ax.hist(values,bins=40,density=True,alpha=.55,label=name,color=color)
    ax.axvline(threshold,color='black',linestyle='--',label=f'Validation frame threshold={threshold:.4g}'); ax.set(xlabel='Frame anomaly score (MSE)',ylabel='Density',title='E2 Test Score Distribution'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'score_distribution.png',dpi=160); plt.close(fig)
    cm=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Normal','Anomaly'],yticklabels=['Normal','Anomaly'],xlabel='Predicted',ylabel='Ground truth',title='E2 Confusion Matrix')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(figdir/'confusion_matrix.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for seq,ax in zip(['testing/01','testing/03','testing/06','testing/08'],axes.flat):
        g=aligned[aligned.sequence==seq].sort_values('frame_index'); x=g.frame_index.to_numpy(); y=g.ground_truth.to_numpy()
        ax.plot(x,g.anomaly_score,color='#244E8A'); ax.axhline(threshold,color='#D08000',linestyle='--')
        starts=np.where((y==1)&np.r_[True,y[:-1]==0])[0]; ends=np.where((y==1)&np.r_[y[1:]==0,True])[0]
        for a,b in zip(starts,ends): ax.axvspan(x[a],x[b],color='#E76F51',alpha=.18)
        ax.set(title=seq,xlabel='Frame index',ylabel='MSE anomaly score'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figdir/'timeline_examples.png',dpi=160); plt.close(fig)
    summary=(run/'run_summary.md').read_text(encoding='utf-8')
    summary+='\n## Final test metrics\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n'
    (run/'run_summary.md').write_text(summary,encoding='utf-8')
    print(json.dumps({'metrics':metrics,'test_clips':len(clips['test']),'test_frames':len(aligned),'run_dir':str(run)},indent=2))


if __name__=='__main__': main()
