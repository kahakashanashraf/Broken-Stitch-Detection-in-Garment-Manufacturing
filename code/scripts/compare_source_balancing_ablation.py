"""Paired 26-inferred-parent resampling for locked development-only A/B/C CV.

All models must predict the SAME images, true labels, source group & validation fold.
Bootstrap resamples inferred photo candidates, not 6,278 independent photographs.
Results are DESCRIPTIVE because hypotheses were selected after prior explorations.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score,confusion_matrix,f1_score,roc_auc_score

ARMS=('class_only','parent_class','parent_class_aug')
COMPARISONS=(('parent_class','class_only'),('parent_class_aug','parent_class'),('parent_class_aug','class_only'))
SEED=20261020
BOOTSTRAP=2000


def read_oof(path):
    rows=list(csv.DictReader(Path(path).read_text().splitlines()))
    if len(rows)!=6278:raise ValueError('STOP: expected 6278 development-only out-of-fold rows')
    store={};parents={}
    for row in rows:
        key=row['path'];parent=row['inferred_parent'];lab=row['label'];fold=int(row['fold']);p=float(row['p_faulty'])
        if key in store or lab not in ('actual','faulty') or not parent.startswith('photo_candidate:') or not 0<=fold<5 or not np.isfinite(p) or not 0<=p<=1:
            raise ValueError('STOP: duplicate/malformed OOF record')
        if parent in parents and parents[parent]!=fold:
            raise ValueError('STOP: source candidate crosses OOF validation folds')
        parents[parent]=fold
        store[key]=(lab,parent,fold,p)
    if len(parents)!=26 or set(parents.values())!=set(range(5)):
        raise ValueError('STOP: 26 inferred parent candidates and five folds required')
    return store


def metric(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    pred=(p>=0.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    return {'n':len(y),'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'normal_recall':float(tn/(tn+fp)) if tn+fp else None,
            'faulty_recall':float(tp/(tp+fn)) if tp+fn else None,
            'faulty_roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'confusion_matrix_normal_faulty':cm.tolist()}


def compare(paths,out,report_paths=None):
    data={arm:read_oof(paths[arm]) for arm in ARMS}
    keys=sorted(data[ARMS[0]])
    canonical=data[ARMS[0]]
    for arm in ARMS[1:]:
        if set(data[arm])!=set(keys):raise ValueError('STOP: OOF sample paths differ across arms')
        for key in keys:
            if data[arm][key][:3]!=canonical[key][:3]:
                raise ValueError('STOP: labels, parent groups, or folds differ across arms')
    if report_paths:
        reports={arm:json.loads(Path(report_paths[arm]).read_text()) for arm in ARMS}
        for arm,rep in reports.items():
            if rep['arm']!=arm or rep['dev_image_n']!=len(keys) or rep['inferred_photo_parent_n']!=26:
                raise ValueError('STOP: arm metadata mismatch')
            if rep['source_balance_intervention_without_parent_capped_sampling'] is not True:
                raise ValueError('STOP: globally sampled training control missing')
            if rep['same_holdout_not_opened'] is not True or rep['stitchingnet_not_opened'] is not True:
                raise ValueError('STOP: previously inspected test was accessed')
            if any(row['parent_overlap']!=0 for row in rep['folds']):
                raise ValueError('STOP: fold leaked')
        for k in range(5):
            sample_sha={reports[a]['folds'][k]['selected_image_paths_sha256'] for a in ARMS}
            if len(sample_sha)!=1:raise ValueError('STOP: training samples differ across arms in fold '+str(k))
        manifest_shas={reports[a]['manifest_sha256'] for a in ARMS}
        if len(manifest_shas)!=1:raise ValueError('STOP: manifest differs')
    y=np.asarray([0 if canonical[k][0]=='actual' else 1 for k in keys],dtype=int)
    parents=np.asarray([canonical[k][1] for k in keys])
    probs={arm:np.asarray([data[arm][k][3] for k in keys]) for arm in ARMS}
    raw={arm:metric(y,probs[arm]) for arm in ARMS}
    unique=sorted(set(parents));idx={parent:np.where(parents==parent)[0] for parent in unique}
    rng=np.random.default_rng(SEED)
    boot={arm:[] for arm in ARMS}
    for _ in range(BOOTSTRAP):
        draws=rng.choice(unique,len(unique),replace=True)
        sample=np.concatenate([idx[k] for k in draws])
        yy=y[sample]
        if len(set(yy))!=2:continue
        for arm in ARMS:
            boot[arm].append(f1_score(yy,probs[arm][sample]>=0.5,labels=[0,1],average='macro',zero_division=0))
    for arm in ARMS:
        vals=np.asarray(boot[arm]);raw[arm]['bootstrap_parent_cluster_95pct_macro_f1']=np.percentile(vals,[2.5,97.5]).tolist()
    changes={}
    for a,b in COMPARISONS:
        delta=np.asarray(boot[a])-np.asarray(boot[b])
        changes[f'{a}_minus_{b}']={'pooled_macro_f1_delta':float(raw[a]['macro_f1']-raw[b]['macro_f1']),
            'paired_parent_bootstrap_descriptive_95pct':np.percentile(delta,[2.5,97.5]).tolist(),
            'bootstrap_fraction_positive_descriptive':float(np.mean(delta>0)),
            'hypothesis':'Additional source reweighting within identical per-parent-capped training pool' if b=='class_only' and a=='parent_class' else 'Additional augmentation / combined changes under identical folds'}
    obj={'status':'DESCRIPTIVE_POST_HOC_DEVELOPMENT_ONLY_ABLATION_NO_INDEPENDENT_TEST',
         'scientific_question':'Does additional inferred-source class-balanced loss reweighting, and separately training-only augmentation, change grouped-CV performance versus class-weighting-only?',
         'arms':{'class_only':'Class-balanced loss, same parent cap, no online augmentation',
                 'parent_class':'Class-and-inferred-parent-balanced loss, same cap, no online augmentation',
                 'parent_class_aug':'Same as parent_class plus train-only rotation/color jitter'},
         'fair_controls':['identical 26 inferred parent CV groups and five folds','identical training images within each fold, SHA seeded global class-stratified 800 images per class (1600 total), no per-parent sampling cap','identical ImageNet MobileNetV3-Small, convolutional tail, optimizer, learning rate, epochs, resize, seed','identical 0.5 threshold and no validation tuning'],
         'n_dev_images':len(keys),'parent_candidates':len(unique),'bootstrap_parent_candidates':len(unique),
         'bootstrap_draws':BOOTSTRAP,'seed':SEED,'model_scores':raw,'paired_effects':changes,
         'not_a_new_independent_test':True,'test_and_stitchingnet_opened':False,
         'important_limitations':['All three arms train on the same per-parent/class capped subset; this tests incremental source weighting, not source balancing vs unrestricted image sampling.',
                                  'The parent candidate identities are inferred from filenames and Photoshop derivative linking, not independently verified raw photography.',
                                  'All labels include historical synthetic Photoshop fault images and augmentation variants.',
                                  'Only 26 candidate source units; resampling CIs are descriptive conditional on development pool.',
                                  'Hypothesis selected after multiple prior model results were seen; cannot claim confirmatory causal generalization to untouched real-world sources.']}
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    (output/'controlled_source_aware_ablation.json').write_text(json.dumps(obj,indent=2))
    lines=['# Controlled source-weighting and train-only augmentation ablation',
        '', '**Exploratory development-only 5-fold grouped-CV results**; 26 inferred photo parents, 6,278 augmented image records.',
        '', '| Arm | Grouped OOF macro-F1 | Balanced accuracy | Faulty recall | Parent bootstrap descriptive 95% |',
        '|---|---:|---:|---:|---|']
    for arm in ARMS:
        m=raw[arm];lo,hi=m['bootstrap_parent_cluster_95pct_macro_f1']
        lines.append(f"| {arm} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['faulty_recall']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    lines+=['','## Paired differences (macro-F1, same OOF images and candidate parents)']
    for name,z in changes.items():
        lo,hi=z['paired_parent_bootstrap_descriptive_95pct']
        lines.append(f"- **{name}**: {z['pooled_macro_f1_delta']:+.4f}, descriptive interval [{lo:+.4f}, {hi:+.4f}]")
    lines+=['','**Caveat:** Within identical global class-stratified sampling, this ablation tests **additional loss weighting by parent**, not the benefit of parent-grouped splits in general. Grouped splits were held fixed throughout.',
             'The previously inspected five-parent holdout and StitchingNet were NOT accessed. This is not a new external result.',
             'Inferred photographed-source identities have not been human verified.','']
    (output/'CONTROLLED_ABLATION_RESULTS.md').write_text('\n'.join(lines))
    print('ABLATION_COMPARISON_FINAL',json.dumps({'macro_f1':{a:raw[a]['macro_f1'] for a in ARMS},'paired_deltas':{k:v['pooled_macro_f1_delta'] for k,v in changes.items()}}),flush=True)
    return obj

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for arm in ARMS:
        parser.add_argument('--'+arm.replace('_','-'),dest=arm,required=True)
        parser.add_argument('--'+arm.replace('_','-')+'-report',dest=arm+'_report',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    compare({a:getattr(args,a) for a in ARMS},args.out,{a:getattr(args,a+'_report') for a in ARMS})