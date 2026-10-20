"""Exploratory no-refit prior correction with TRAIN-FABRIC class prevalence only.

CAUTION: Planned after preliminary heldout-fabric results were seen, so adjustment
is post-hoc exploratory even though each numerical threshold uses only that
fold's TRAINING fabric class counts, not its validation labels.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix,accuracy_score
import compare_fabric_adaptation as base

ARMS=base.ARMS


def train_prior_threshold(report):
    counts=report['training_label_counts']
    if set(counts)!={'0','1'}:raise ValueError('STOP: missing train classes')
    n0=int(counts['0']);n1=int(counts['1'])
    if n0<100 or n1<100 or n0+n1!=int(report['train_samples']):raise ValueError('STOP: bad training-only counts')
    # Trained BCE was balanced to 50% per class. To adjust to observed TRAIN
    # source prevalence, weighted p>=1-prior is equivalent to corrected p>=.5.
    return n0/(n0+n1)


def process(inputs,reports,output):
    pred={a:base.read_oof(inputs[a]) for a in ARMS}
    order=sorted(pred[ARMS[0]])
    if set(order)!=set(pred[ARMS[1]]):raise ValueError('STOP: models do not align')
    for path in order:
        if pred[ARMS[0]][path][:3]!=pred[ARMS[1]][path][:3]:raise ValueError('STOP: group fold/label mismatch')
    y=np.array([int(pred[ARMS[0]][path][1]=='broken') for path in order])
    fold=np.array([pred[ARMS[0]][path][2] for path in order])
    group=np.array([pred[ARMS[0]][path][0] for path in order])
    scores={}
    details={}
    for arm in ARMS:
        thresholds={}
        for k,repfile in enumerate(reports[arm]):
            rep=json.loads(Path(repfile).read_text())
            if rep['arm']!=arm or rep['fold']!=k or rep['val_fabric_overlap']!=0:
                raise ValueError('STOP: report identity or group separation mismatch')
            thresholds[k]=train_prior_threshold(rep)
        p=np.array([pred[arm][path][3] for path in order])
        fold_thr=np.array([thresholds[int(k)] for k in fold])
        at05=(p>=.5).astype(int)
        prior_pred=(p>=fold_thr).astype(int)
        scores[arm]={
          'threshold_0_5':measures(y,at05),
          'train_fabric_prior_adjusted':measures(y,prior_pred),
          'threshold_by_fold_train_only':{str(k):float(v) for k,v in thresholds.items()},
        }
        details[arm]={'at05':at05,'prior':prior_pred}
    result={'status':'POSTHOC_EXPLORATORY_PREVALENCE_CORRECTION_ON_EARLIER_INSPECTED_STITCHINGNET',
      'method':'Each heldout fold threshold equals count(normal in TRAIN fabric groups) / count(all in TRAIN fabric groups). This corrects balanced binary cross entropy from equal class prior to TRAIN fabric prevalence.',
      'n':len(y),'source_fabrics':11,'per_arm_metrics':scores,
      'posthoc_design':True,'heldout_fabric_labels_used_to_choose_thresholds':False,
      'prohibitions':['Threshold recipe was proposed after first CV results were inspected. These adjusted OOF scores are exploratory; cannot be presented as new independent confirmation.','A probability ranking ROC-AUC is unchanged by monotonic prior correction inside each fold, but pooling different fold thresholds may affect global ranking.','Source StitchingNet had already been inspected; this is NOT thesis-only zero-shot generalization.']}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'train_prior_adjustment.json').write_text(json.dumps(result,indent=2))
    lines=['# Posthoc train-fabric prevalence threshold correction','',
      '**Exploratory / post-hoc: NOT a fresh independent test.**',
      'Threshold is based ONLY on class counts from each fold TRAINING fabrics; **no heldout labels are used to choose a numerical threshold**.','',
      '| Initialization | Fixed 0.5 macro-F1 | Train-prior macro-F1 | Prior broken recall | Prior false alarms |',
      '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        x=scores[arm];a=x['threshold_0_5'];b=x['train_fabric_prior_adjusted']
        lines.append(f"| {arm} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {b['broken_recall']:.4f} | {b['fp']} |")
    lines+=['','This is an operating-point sensitivity analysis: macro-F1 improvements here are not model-weight improvements.','']
    (out/'TRAIN_PRIOR_ADJUSTMENT.md').write_text('\n'.join(lines))
    print('PRIOR_CORRECTION_RESULT',json.dumps(scores),flush=True)


def measures(y,pred):
    tn,fp,fn,tp=[int(x) for x in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    return {'macro_f1':float(f1_score(y,pred,average='macro')),
      'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
      'accuracy':float(accuracy_score(y,pred)),'broken_recall':tp/(tp+fn),
      'normal_recall':tn/(tn+fp),'tn':tn,'fp':fp,'fn':fn,'tp':tp}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arm in ARMS:
        for k in range(5):
            parser.add_argument('--'+arm+str(k),required=True)
            parser.add_argument('--'+arm+str(k)+'report',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    process({a:[getattr(args,a+str(k)) for k in range(5)] for a in ARMS},
            {a:[getattr(args,a+str(k)+'report') for k in range(5)] for a in ARMS},args.out)