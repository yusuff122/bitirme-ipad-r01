import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def calculate_metrics(y_true, scores, threshold):
    y=np.asarray(y_true,dtype=int); s=np.asarray(scores,dtype=float); p=(s>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'aupr':float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,'f1':float(f1_score(y,p,zero_division=0)),'precision':float(precision_score(y,p,zero_division=0)),'recall':float(recall_score(y,p,zero_division=0)),'accuracy':float(accuracy_score(y,p)),'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'fpr':float(fp/(fp+tn)) if fp+tn else 0.0,'fnr':float(fn/(fn+tp)) if fn+tp else 0.0}
