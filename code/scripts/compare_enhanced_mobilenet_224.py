"""Compare frozen previous 128px/2epoch OOF with new 224px epochs 2 and 12.

Results are development-only on 26 filename-inferred source-root groups.
Predeclared milestones; previously inspected test and StitchingNet never accessed.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix

SLOTS=('baseline128e2','new224e2','new224e12')


def read_old(path):
    data={}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            key=row['path']
            if key in data:raise ValueError('STOP duplicate old OOF record')
            p=float(row['p_faulty']);fold=int(row['fold'])
            if not np.isfinite(p) or not 0<=p<=1 or not 0<=fold<5:raise ValueError('STOP invalid previous probability')
            data[key]={'label':row['label'],'parent':row['inferred_parent'],'fold':fold,'baseline128e2':p}
    if len(data)!=6278:raise ValueError('STOP previous OOF size != 6278')
    return data


def read_new(files,previous):
    data={}
    for k,fold_path in enumerate(files):
        with Path(fold_path).open(newline='') as handle:
            for r in csv.DictReader(handle):
                key=r['path']
                if key in data:raise ValueError('STOP duplicated enhanced OOF sample')
                if key not in previous:raise ValueError('STOP enhanced OOF path not in old baseline')
                old=previous[key]
                if r['label']!=old['label'] or r['inferred_parent']!=old['parent'] or int(r['fold'])!=k or old['fold']!=k:
                    raise ValueError('STOP enhanced fold/label/group mismatch')
                p2=float(r['p_faulty_epoch2']);p12=float(r['p_faulty_epoch12'])
                if not np.isfinite([p2,p12]).all() or not (0<=p2<=1 and 0<=p12<=1):
                    raise ValueError('STOP invalid enhanced OOF probabilities')
                data[key]={'new224e2':p2,'new224e12':p12}
    if len(data)!=len(previous):raise ValueError('STOP enhanced OOF incomplete')
    return data


def score(y,p):
    pred=(p>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    return {'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'faulty_recall':float(tp/(tp+fn)),
            'normal_recall':float(tn/(tn+fp)),
            'roc_auc_faulty':float(roc_auc_score(y,p)),
            'faulty_missed':fn,'false_faulty_alarms':fp,
            'confusion_matrix_normal_faulty':cm.tolist()}


def summarize(old_path, new_files, out, draws=2000):
    old=read_old(old_path)
    new=read_new(new_files,old)
    keys=sorted(old)
    y=np.asarray([int(old[k]['label']=='faulty') for k in keys])
    parents=np.asarray([old[k]['parent'] for k in keys])
    fold=np.asarray([old[k]['fold'] for k in keys])
    if len(set(parents))!=26 or len(set(fold))!=5 or set(y)!={0,1}:
        raise ValueError('STOP baseline partition mismatch')
    parents_by_fold={}
    for pa,fo in zip(parents,fold):
        if pa in parents_by_fold and parents_by_fold[pa]!=fo:raise ValueError('STOP baseline parent crossing folds')
        parents_by_fold[pa]=fo
    predictions={k:np.asarray([old[x][k] if k=='baseline128e2' else new[x][k] for x in keys],dtype=float) for k in SLOTS}
    summary={k:score(y,predictions[k]) for k in SLOTS}
    group_list=np.asarray(sorted(set(parents)))
    by_group={k:np.flatnonzero(parents==k) for k in group_list}
    rng=np.random.default_rng(20261020)
    boot={k:[] for k in SLOTS}
    for i in range(draws):
        g=rng.choice(group_list,size=len(group_list),replace=True)
        ids=np.concatenate([by_group[x] for x in g])
        yy=y[ids]
        if len(set(yy))<2:continue
        for slot in SLOTS:
            boot[slot].append(float(f1_score(yy,predictions[slot][ids]>=.5,average='macro')))
    for slot in SLOTS:
        v=np.asarray(boot[slot]);summary[slot]['descriptive_parent_bootstrap_macro_f1_95pct']=np.percentile(v,[2.5,97.5]).tolist()
    pair={}
    for lhs,rhs in [('new224e2','baseline128e2'),('new224e12','new224e2'),('new224e12','baseline128e2')]:
        dd=np.asarray(boot[lhs])-np.asarray(boot[rhs])
        pair[f'{lhs}_minus_{rhs}']={'macro_f1_difference':summary[lhs]['macro_f1']-summary[rhs]['macro_f1'],
                                    'descriptive_parent_bootstrap_95pct':np.percentile(dd,[2.5,97.5]).tolist(),
                                    'bootstrap_valid_draws':len(dd)}
    result={'status':'EXPLORATORY_26_FILENAME_INFERRED_SOURCE_PARENT_DEVELOPMENT_CV_ONLY',
            'prespecified_comparisons':'128px 2epoch vs 224px 2epoch (resolution); 224px 2epoch vs 224px 12epoch (more updates)',
            'same_model':'MobileNetV3-Small, ImageNet pretrained tail fine tuned',
            'same_train_sampling':'Class_only global 800 actual and 800 faulty SHA-ranked records per fold; same hashes, same seed, no online augmentation',
            'n':len(keys),'source_parent_n':len(group_list),'fold_n':5,'scores':summary,
            'paired_bootstrap_differences':pair,'num_bootstrap_draws':draws,
            'prior_holdout_accessed':False,'stitchingnet_accessed':False,
            'caveats':['All groups are filename-inferred; physical source authenticity not confirmed.',
                       'OOF is development-only, earlier external test already inspected.',
                       'Pairwise clustered intervals are descriptive, not inferential guarantees.',
                       'Predeclared 2 and 12 epoch evaluations were not chosen by outer-fold performance.',
                       'No new independent photographs or external test evaluated for the enhanced checkpoint.']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'enhanced_mobilenet_224cv_comparison.json').write_text(json.dumps(result,indent=2))
    with (out/'enhanced_paired_predictions.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['path','true_label','inferred_parent','fold']+list(SLOTS))
        for i,key in enumerate(keys):w.writerow([key,old[key]['label'],parents[i],int(fold[i])]+[predictions[z][i] for z in SLOTS])
    md=['# MobileNetV3 224px 12epoch development grouped CV — comparison','',
        'All measurements are development OOF on 26 filename-inferred photograph groups, NOT fresh external tests.',
        'Five identical grouped folds, fixed class_only 800 normal+800 faulty sampling, no online augmentation.',
        '', '| Training recipe | macro-F1 | balanced accuracy | broken recall | ROC-AUC |', '|---|---:|---:|---:|---:|']
    for k in SLOTS:
        v=summary[k];md.append(f"| {k} | {v['macro_f1']:.4f} | {v['balanced_accuracy']:.4f} | {v['faulty_recall']:.4f} | {v['roc_auc_faulty']:.4f} |")
    md.extend(['','Paired 26-parent-cluster bootstrap intervals (descriptive):',''])
    for k,v in pair.items():md.append(f"- {k}: Δ macro-F1 {v['macro_f1_difference']:+.4f} 95% interval [{v['descriptive_parent_bootstrap_95pct'][0]:+.4f}, {v['descriptive_parent_bootstrap_95pct'][1]:+.4f}]")
    md+=['','The original test and external StitchingNet were NOT accessed. A final refit checkpoint is distinct from fold-specific OOF models.',
          'Changing training epochs may overfit synthetic/Photoshop derivatives; new independent images needed for deployment claims.']
    (out/'ENHANCED_MOBILENET_224_COMPARISON.md').write_text('\n'.join(md)+'\n')
    print('ENHANCED_COMPARISON',json.dumps({'scores':{k:{'macro_f1':v['macro_f1'],'balanced_acc':v['balanced_accuracy'],'faulty_recall':v['faulty_recall']} for k,v in summary.items()},'pairs':pair}),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True)
    for k in range(5):ap.add_argument(f'--fold{k}',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    summarize(a.baseline,[getattr(a,f'fold{k}') for k in range(5)],a.out)