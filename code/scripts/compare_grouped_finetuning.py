"""Compare previously frozen 5fold development OOF model predictions, cluster bootstrap.

This script DOES NOT train, evaluate held-out test, or read StitchingNet. It cannot
turn development-only scores into independent evaluation claims.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score

MODELS=('mobilenetv3','densenet121','vgg16')
SEED=20261020
B=2000


def read_oof(path):
    rows=list(csv.DictReader(Path(path).read_text().splitlines()))
    if len(rows)!=6278:raise ValueError('STOP: 6278 grouped dev OOF images required')
    by={}
    for r in rows:
        key=r['path']
        if key in by:raise ValueError('STOP: repeated development sample '+key)
        if r['label'] not in ('actual','faulty') or not r['inferred_parent'].startswith('photo_candidate:'):
            raise ValueError('STOP: invalid class/source')
        p=float(r['p_faulty']);fold=int(r['fold'])
        if not np.isfinite(p) or not 0<=p<=1 or not 0<=fold<5:
            raise ValueError('STOP: invalid probability or fold')
        by[key]=(r['label'],r['inferred_parent'],fold,p)
    if len({r[1] for r in by.values()})!=26:raise ValueError('STOP: unexpected parent count')
    folds={}
    for row in by.values():
        old=folds.setdefault(row[1],row[2])
        if old!=row[2]:raise ValueError('STOP: one parent assigned to multiple validation folds')
    if len(set(folds.values()))!=5:raise ValueError('STOP: missing CV folds')
    return by


def metrics(y,p):
    pred=(np.asarray(p)>=0.5).astype(int)
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if set(y)!={0,1}:raise ValueError('STOP: comparison sample lacks a class')
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return {'n':len(y),'macro_f1':float(f1_score(y,pred,average='macro')),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),
            'faulty_recall':float(tp/(tp+fn)), 'normal_recall':float(tn/(tn+fp)),
            'faulty_auc':float(roc_auc_score(y,p)),
            'confusion_matrix_normal_faulty':cm.tolist()}


def summarize(files,out):
    inputs={model:read_oof(files[model]) for model in MODELS}
    canonical=inputs[MODELS[0]]
    order=sorted(canonical)
    for name in MODELS[1:]:
        if set(inputs[name])!=set(order):raise ValueError('STOP: different OOF keys')
        for path in order:
            if inputs[name][path][:3]!=canonical[path][:3]:
                raise ValueError('STOP: label/parent/fold mismatch across architectures')
    labels=np.asarray([0 if canonical[p][0]=='actual' else 1 for p in order],dtype=int)
    roots=np.asarray([canonical[p][1] for p in order])
    preds={m:np.asarray([inputs[m][p][3] for p in order],dtype=float) for m in MODELS}
    raw={m:metrics(labels,preds[m]) for m in MODELS}
    rng=np.random.default_rng(SEED)
    parents=np.asarray(sorted(set(roots)))
    buckets={p:np.where(roots==p)[0] for p in parents}
    boot={m:[] for m in MODELS};diffs={}
    for _ in range(B):
        drawn=rng.choice(parents,size=len(parents),replace=True)
        sample=np.concatenate([buckets[p] for p in drawn])
        yy=labels[sample]
        if len(set(yy))<2:continue
        for m in MODELS:
            boot[m].append(f1_score(yy,preds[m][sample]>=0.5,average='macro'))
    for m in MODELS:
        bs=np.asarray(boot[m])
        raw[m]['parent_cluster_bootstrap_macro_f1_95pct']=np.percentile(bs,[2.5,97.5]).tolist()
        raw[m]['bootstrap_n']=len(bs)
    for a,b in [('mobilenetv3','densenet121'),('mobilenetv3','vgg16'),('densenet121','vgg16')]:
        delta=np.asarray(boot[a])-np.asarray(boot[b])
        diffs[f'{a}_minus_{b}']={
            'delta_pooled_macro_f1':raw[a]['macro_f1']-raw[b]['macro_f1'],
            'descriptive_paired_parent_bootstrap_95pct':np.percentile(delta,[2.5,97.5]).tolist(),
        }
    output={
       'status':'EXPLORATORY_COMPARISON_POST_HOC_SELECTION_NOT_INDEPENDENT_TEST',
       'methods':'Same fixed 5-fold source-grouped CV with 26 filename-inferred parent roots; 128px; fixed 2 epochs; train cap 40 per parent/class; genuine final-layer fine-tuning; per-fold OOF predictions; final refits are not OOF.',
       'num_dev_images':len(order),'inferred_parent_count':len(parents),
       'num_bootstrap_draws':B,'bootstrap_parent_unit':'26 inferred photographic roots, not 6278 independently captured photos',
       'bootstrap_seed':SEED,'model_metrics':raw,'pairwise_differences':diffs,
       'test_and_stitchingnet_data_access':False,
       'cautions':['The five-parent prior holdout was already inspected in earlier work; not a new independent model-comparison test.',
                   'Original Photoshop source-photo root mapping was inferred, not human-verified.',
                   'A single 5fold seed is a fragile estimate of model ranking.',
                   'All models fitted partly on correlated synthetic/augmented records.',
                   'Architecture comparison is exploratory (three candidates) and may overfit development protocol.',
                   'Cluster-bootstrap describes conditional score variability, not generalization beyond unseen source domains.']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'architecture_groupedcv_comparison.json').write_text(json.dumps(output,indent=2))
    lines=['# Pretrained last-layer fine-tuning: development-only grouped 5-fold comparison','',
           '**NOT an independent, untouched test.** All scores use 26 filename-inferred source parents (6,278 augmented images).','',
           '| Architecture | OOF macro-F1 | balanced accuracy | faulty recall | parent-bootstrap 95% (descriptive) |',
           '|---|---:|---:|---:|---|']
    for m in MODELS:
        z=raw[m];lo,hi=z['parent_cluster_bootstrap_macro_f1_95pct']
        lines.append(f"| {m} | {z['macro_f1']:.4f} | {z['balanced_accuracy']:.4f} | {z['faulty_recall']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','**Same 5-fold assignments, no held-out prior test or StitchingNet data.**',
            'Each model checkpoint is final refit on the entire development pool; listed score arises only from 5-fold OOF.',
            'The 26 roots are inferred from filename links, and do not prove independent source photographs.',
            'A strong leaderboard claim or external deployment claim is not supported.','']
    (out/'GROUPED_CV_COMPARISON.md').write_text('\n'.join(lines))
    print('GROUPED_CV_COMPARISON',json.dumps({'model_scores':{k:{'macro_f1':v['macro_f1'],'balanced_acc':v['balanced_accuracy']} for k,v in raw.items()},'parent_n':26}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for m in MODELS:p.add_argument('--'+m,required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args();summarize({m:getattr(a,m) for m in MODELS},a.out)