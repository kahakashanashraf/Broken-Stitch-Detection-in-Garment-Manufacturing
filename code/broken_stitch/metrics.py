"""Metrics: keep class-specific rates and class imbalance explicit."""
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support,roc_auc_score,average_precision_score


def compute_metrics(y_true,prob_faulty):
    yt=np.asarray(y_true,dtype=int)
    p=np.asarray(prob_faulty,dtype=float)
    if yt.size==0 or yt.size!=p.size:raise ValueError('Empty or misaligned predictions')
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid class probabilities')
    pred=(p>=.5).astype(int)
    pr,rc,f1,support=precision_recall_fscore_support(yt,pred,labels=[0,1],zero_division=0)
    ans={'n':int(yt.size), 'accuracy':float(accuracy_score(yt,pred)),
         'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),
         'macro_f1':float(np.mean(f1)),
         'confusion_matrix_actual_faulty':confusion_matrix(yt,pred,labels=[0,1]).tolist(),
         'actual':{'precision':float(pr[0]),'recall':float(rc[0]),'f1':float(f1[0]),'support':int(support[0])},
         'faulty':{'precision':float(pr[1]),'recall':float(rc[1]),'f1':float(f1[1]),'support':int(support[1])}}
    if set(yt)=={0,1}:
        ans['roc_auc_faulty']=float(roc_auc_score(yt,p))
        ans['average_precision_faulty']=float(average_precision_score(yt,p))
    else:ans['roc_auc_faulty']=ans['average_precision_faulty']=None
    return ans