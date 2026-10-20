"""Audit and compare 5-fold unseen-fabric StitchingNet target-supervised adaptation.

This analysis is NOT zero-shot or untouched external validation. Source-grouped
fabric holdout is prospective at fold level but dataset previously inspected.
"""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score,roc_auc_score,confusion_matrix

ARMS=('imagenet','synthetic')
N=3637
FABRICS=set('ABCDEFGHIJK')
SEED=20261020


def read_oof(files):
    prediction={}
    group_folds={}
    for fold,file in enumerate(files):
        rows=list(csv.DictReader(Path(file).read_text().splitlines()))
        if not rows:raise ValueError('STOP: empty held-out fold')
        for r in rows:
            key=r['source_file']
            if key in prediction:raise ValueError('STOP: duplicate OOF sample')
            fab=r['fabric'];lab=r['actual_class'];k=int(r['fold']);p=float(r['probability_broken'])
            if fab not in FABRICS or lab not in ('normal','broken') or k!=fold or not np.isfinite(p) or not 0<=p<=1:
                raise ValueError('STOP: invalid sample/fabric/fold/class/score')
            if fab in group_folds and group_folds[fab]!=fold:raise ValueError('STOP: same fabric held out in multiple folds')
            group_folds[fab]=fold
            prediction[key]=(fab,lab,fold,p)
    if len(prediction)!=N or set(group_folds)!=FABRICS or len(set(group_folds.values()))!=5:
        raise ValueError('STOP: all 3637 images and 11 fabrics/5 folds required')
    cnt=Counter(v[1] for v in prediction.values())
    if cnt!={'normal':2436,'broken':1201}:raise ValueError('STOP: class counts wrong')
    return prediction


def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if set(y)!={0,1}:raise ValueError('STOP: metrics require both classes')
    pred=(p>=.5).astype(int)
    tn,fp,fn,tp=[int(x) for x in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    return {'macro_f1':float(f1_score(y,pred,average='macro')),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'accuracy':float(accuracy_score(y,pred)),
            'broken_recall':tp/(tp+fn),
            'normal_recall':tn/(tn+fp),
            'broken_auc':float(roc_auc_score(y,p)),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}


def compare(inp,out,zero_shot=None):
    data={arm:read_oof(inp[arm]) for arm in ARMS}
    a=data[ARMS[0]];b=data[ARMS[1]]
    keys=sorted(a)
    if set(keys)!=set(b):raise ValueError('STOP: arm OOF samples mismatch')
    for key in keys:
        if a[key][:3]!=b[key][:3]:raise ValueError('STOP: arms have unmatched label/fabric/fold')
    y=np.array([int(a[k][1]=='broken') for k in keys]);group=np.array([a[k][0] for k in keys]);p={arm:np.array([data[arm][k][3] for k in keys]) for arm in ARMS}
    result={arm:metrics(y,p[arm]) for arm in ARMS}
    baseline=None
    if zero_shot is not None:
        rows=list(csv.DictReader(Path(zero_shot).read_text().splitlines()))
        zs={r['source_file']:(r['fabric'],r['actual_class'],float(r['p_broken'])) for r in rows}
        if set(zs)!=set(keys):raise ValueError('STOP: zero-shot reference cannot be aligned')
        if any(zs[k][0]!=a[k][0] or zs[k][1]!=a[k][1] for k in keys):raise ValueError('STOP: zero-shot baseline class/fabric mismatch')
        z=np.array([zs[k][2] for k in keys]);baseline=metrics(y,z)
    by_fabric={fab:{arm:metrics(y[group==fab],p[arm][group==fab]) for arm in ARMS} for fab in sorted(set(group))}
    rng=np.random.default_rng(SEED);fabric=list(sorted(set(group)))
    idx={fab:np.where(group==fab)[0] for fab in fabric}
    diffs=[]
    for _ in range(2000):
        draw=rng.choice(fabric,len(fabric),replace=True)
        ix=np.concatenate([idx[fab] for fab in draw])
        yy=y[ix]
        if set(yy)!={0,1}:continue
        diffs.append(f1_score(yy,p['synthetic'][ix]>=.5,average='macro')-f1_score(yy,p['imagenet'][ix]>=.5,average='macro'))
    if len(diffs)<1900:raise ValueError('STOP: too many invalid cluster-bootstrap replicates')
    d={'synthetic_minus_imagenet_macro_f1':result['synthetic']['macro_f1']-result['imagenet']['macro_f1'],
       'descriptive_11_fabric_cluster_bootstrap_95pct':np.percentile(diffs,[2.5,97.5]).tolist(),
       'bootstrap_draws':len(diffs)}
    result_json={'status':'DESCRIPTIVE_SUPERVISED_STITCHINGNET_ADAPTATION_HELDOUT_FABRIC_NOT_BLIND',
      'training_dataset':'StitchingNet normal/broken subset: this was previously used as an external evaluation dataset in project',
      'test_protocol':'Five grouped CV folds hold out entire fabric categories, train on other StitchingNet fabrics; model initialized from ImageNet vs ImageNet+thesis-synthetic-pretraining',
      'n_images':len(y),'n_fabrics':len(fabric),'metric':'pooled fabric-held-out OOF macro-F1; 0.5 threshold',
      'models':result,'older_zero_shot_reference':baseline,
      'paired_11fabric_uncertainty':d,'by_fabric':by_fabric,
      'caution':['This method trains with labeled images of StitchingNet. Do NOT call results zero-shot transfer, an independent new dataset or test against StitchingNet original paper protocols.',
       'Fabric-category exclusivity does not guarantee that all machine/session correlation is absent.',
       'Comparison is retrospective; StitchingNet had been used for earlier model inspection.',
       '11 fabrics are only 11 group units; uncertainty descriptive not a significance guarantee.',
       'Grad-CAM heatmaps are qualitative interpretations, not spatial ground-truth validation.']}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'fabric_group_adaptation_comparison.json').write_text(json.dumps(result_json,indent=2))
    line=['# StitchingNet fabric-held-out adaptation with Grad-CAM','',
      '**These models were trained on StitchingNet; this is NOT zero-shot external transfer.**',
      'Original StitchingNet had been inspected earlier in this project; descriptive grouped-CV experiment only.','',
      '| Model initialization | Pooled OOF macro-F1 | Balanced accuracy | Broken recall | Broken misses |',
      '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        s=result[arm];line.append(f"| {arm} | {s['macro_f1']:.4f} | {s['balanced_accuracy']:.4f} | {s['broken_recall']:.4f} | {s['fn']} |")
    if baseline:
        line.append(f"| Thesis-only zero-shot MobileNet (different evaluation: no StitchingNet training) | {baseline['macro_f1']:.4f} | {baseline['balanced_accuracy']:.4f} | {baseline['broken_recall']:.4f} | {baseline['fn']} |")
    line+=['',f"Paired (synthetic minus ImageNet) OOF macro-F1: **{d['synthetic_minus_imagenet_macro_f1']:+.4f}**; descriptive 11-fabric bootstrap interval {d['descriptive_11_fabric_cluster_bootstrap_95pct']}",
      '','**Scientific interpretation:** source-data adaptation and initialization effects are distinct. A high StitchingNet-trained score is not evidence that original thesis-only model transfers at that accuracy.',
      '**XAI:** Each per-fold release includes Grad-CAM grayscale maps for broken-logit errors/hits. These are actual derivative saliency maps, but not pixel-validated defect localization.','']
    (out/'FABRIC_ADAPTATION_COMPARISON.md').write_text('\n'.join(line))
    print('FABRIC_COMPARISON',json.dumps({'scores':result,'paired':d,'old_zero_shot':baseline}),flush=True)
    return result_json

if __name__=='__main__':
    q=argparse.ArgumentParser()
    for arm in ARMS:
        for k in range(5):q.add_argument('--'+arm+str(k),required=True)
    q.add_argument('--zero-shot')
    q.add_argument('--out',required=True)
    a=q.parse_args()
    compare({arm:[getattr(a,arm+str(k)) for k in range(5)] for arm in ARMS},a.out,a.zero_shot)