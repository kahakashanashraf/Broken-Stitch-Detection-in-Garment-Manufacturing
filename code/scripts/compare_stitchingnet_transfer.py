"""Compare six immutable prediction tables on already inspected StitchingNet.

NO evaluation of new models, training, calibration, threshold selection or data download
beyond prediction CSVs. Fabric-cluster bootstrap is descriptive with only 11 clusters.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score

MODELS=('class_only','parent_class','parent_class_aug','mobilenetv3','densenet121','vgg16')
BOOTSTRAP_DRAWS=1000
SEED=20261020


def load_predictions(path):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    if len(rows)!=3637:raise ValueError('STOP: expected full StitchingNet binary subset')
    results={}
    for r in rows:
        name=r['source_file']
        if name in results:raise ValueError('STOP: duplicate image path')
        label=r['actual_class']
        if label not in ('normal','broken'):raise ValueError('STOP: invalid label')
        fabric=r['fabric']
        if fabric not in 'ABCDEFGHIJK' or len(fabric)!=1:
            raise ValueError('STOP: invalid fabric group')
        if r['prediction'] not in ('normal','broken'):
            raise ValueError('STOP: invalid prediction label')
        p=float(r['p_broken'])
        if not np.isfinite(p) or p<0 or p>1 or r['prediction']!=('broken' if p>=0.5 else 'normal'):
            raise ValueError('STOP: threshold/score mismatch')
        results[name]=(label,fabric,p)
    if Counter(x[0] for x in results.values())!=Counter({'normal':2436,'broken':1201}):
        raise ValueError('STOP: class label totals mismatch')
    if len(set(x[1] for x in results.values()))!=11:
        raise ValueError('STOP: expected 11 fabric groups')
    return results


def evaluate(y,p):
    pred=(p>=0.5).astype(int)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'normal_recall':float(recall_score(y,pred,pos_label=0)),
        'broken_recall':float(recall_score(y,pred,pos_label=1)),
        'broken_auroc':float(roc_auc_score(y,p))}


def compare(inputs,out):
    tables={m:load_predictions(inputs[m]) for m in MODELS}
    keys=sorted(tables[MODELS[0]])
    for m in MODELS[1:]:
        if set(tables[m])!=set(keys):raise ValueError('STOP: different external examples per model')
        for name in keys:
            if tables[m][name][:2]!=tables[MODELS[0]][name][:2]:
                raise ValueError('STOP: labels/fabrics not paired across models')
    yy=np.array([int(tables[MODELS[0]][name][0]=='broken') for name in keys],dtype=int)
    fabric=np.array([tables[MODELS[0]][name][1] for name in keys]);fabids=sorted(set(fabric))
    probs={m:np.array([tables[m][name][2] for name in keys],dtype=float) for m in MODELS}
    scores={m:evaluate(yy,p) for m,p in probs.items()}
    rng=np.random.default_rng(SEED)
    idx={g:np.where(fabric==g)[0] for g in fabids}
    f1boots={m:[] for m in MODELS};recallboots={m:[] for m in MODELS}
    for _ in range(BOOTSTRAP_DRAWS):
        chosen=rng.choice(fabids,len(fabids),replace=True)
        ix=np.concatenate([idx[g] for g in chosen]);ys=yy[ix]
        if set(ys)!={0,1}:continue
        for m in MODELS:
            pp=probs[m][ix];pr=pp>=0.5
            f1boots[m].append(f1_score(ys,pr,average='macro'))
            recallboots[m].append(recall_score(ys,pr,pos_label=1))
    for m in MODELS:
        for k,bs in [('fabric_bootstrap_macro_f1_95pct',f1boots[m]),('fabric_bootstrap_broken_recall_95pct',recallboots[m])]:
            scores[m][k]=np.percentile(bs,[2.5,97.5]).tolist()
    paired={}
    for a,b in [('parent_class','class_only'),('parent_class_aug','parent_class'),('mobilenetv3','class_only'),('mobilenetv3','densenet121')]:
        ds=np.asarray(f1boots[a])-np.asarray(f1boots[b]);dr=np.asarray(recallboots[a])-np.asarray(recallboots[b])
        paired[f'{a}_minus_{b}']={
            'observed_delta_macro_f1':scores[a]['macro_f1']-scores[b]['macro_f1'],
            'descriptive_fabric_bootstrap_delta_macro_f1_95pct':np.percentile(ds,[2.5,97.5]).tolist(),
            'observed_delta_broken_recall':scores[a]['broken_recall']-scores[b]['broken_recall'],
            'descriptive_fabric_bootstrap_delta_broken_recall_95pct':np.percentile(dr,[2.5,97.5]).tolist()}
    obj={'status':'DESCRIPTIVE_CROSS_DATASET_POST_HOC_NON_BLIND_COMPARISON',
         'source':'https://zenodo.org/records/20352813',
         'stitchingnet_previously_inspected':True,
         'new_blind_test_claim':False,'model_threshold_calibration_or_training':False,
         'evaluation_samples':len(keys),'fabrics':fabids,
         'bootstrap_unit':'11 fabric-thread combinations, not 3637 independent camera sources',
         'bootstrap_draws':BOOTSTRAP_DRAWS,'seed':SEED,
         'models':scores,'paired_differences':paired,
         'scientific_note':'Results are descriptive domain-transfer on an already examined dataset. Selection among these models using external results consumes the benchmark; do not claim a fresh untouched test or real deployment readiness.'}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'stitchingnet_transfer_comparison.json').write_text(json.dumps(obj,indent=2))
    rows=['# Frozen model transfer to previously inspected StitchingNet','',
          '**Non-blind descriptive evaluation**, no external training or calibration; 2436 normal + 1201 broken; 11 fabrics.','',
          '| Model | Macro-F1 | Balanced accuracy | Broken recall | AUROC | Fabric bootstrap 95% interval (descriptive) |',
          '|---|---:|---:|---:|---:|---|']
    for m in MODELS:
        x=scores[m];l,h=x['fabric_bootstrap_macro_f1_95pct']
        rows.append(f"| {m} | {x['macro_f1']:.4f} | {x['balanced_accuracy']:.4f} | {x['broken_recall']:.4f} | {x['broken_auroc']:.4f} | [{l:.4f}, {h:.4f}] |")
    rows+=['','Unseen source-photograph ancestry of original thesis data is filename-inferred, not manually authenticated.',
           'Fabric bootstrap uses 11 acquisition-condition groups and is an uncertainty *description*, not a significance guarantee.',
           'The StitchingNet dataset has already been used by the project in a previous model evaluation, so these are **not** blind new-test results.',
           'No model architecture or threshold was tuned on StitchingNet within this workflow.','']
    (out/'STITCHINGNET_TRANSFER_COMPARISON.md').write_text('\n'.join(rows))
    print('TRANSFER_COMPARISON '+json.dumps({m:{'macro_f1':x['macro_f1'],'broken_recall':x['broken_recall'],'balanced_accuracy':x['balanced_accuracy']} for m,x in scores.items()},sort_keys=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for m in MODELS:ap.add_argument('--'+m,required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    compare({m:Path(getattr(args,m)) for m in MODELS},args.out)