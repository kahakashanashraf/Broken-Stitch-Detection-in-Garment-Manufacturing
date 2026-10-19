"""One-time frozen inferred-photo-parent internal holdout evaluation.

This does NOT verify photographed-source ancestry and does NOT use StitchingNet.
Never retrain or recalibrate in this evaluation. No data or threshold selection.
"""
import argparse,csv,hashlib,json,zipfile
from collections import Counter
from pathlib import Path
import torch
from broken_stitch.metrics import compute_metrics
from frozen_stitchingnet_eval import predict_zip,sha

BAD='dataset/train/actual/actual_original_AB.png_579370ba-65d0-4793-a3c0-0f73fbf6c367.png'
EXPECTED_CHECKPOINT_SHA='02fa682d0648aa2521d50bb5f402ba4c8452ee71fd079333d135af9178435253'
EXPECTED_ARCHIVE_BYTES=2146784353

def main(a):
    if Path(a.archive).stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError('STOP: wrong original ZIP size')
    source=json.loads(Path(a.archive_metadata).read_text())
    if sha(a.archive)!=source['sha256']:
        raise RuntimeError('STOP: original ZIP digest mismatch')
    report=json.loads(Path(a.training_report).read_text())
    ck_sha=sha(a.weights)
    if ck_sha!=EXPECTED_CHECKPOINT_SHA or ck_sha!=report['weights_sha256']:
        raise RuntimeError('STOP: checkpoint changed since the fixed training run')
    ck=torch.load(a.weights,map_location='cpu',weights_only=True)
    manifest_bytes=Path(a.manifest).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=ck['manifest_sha256']:
        raise RuntimeError('STOP: train/test manifest differs from checkpoint')
    if ck['arch']!='mobilenet_v3_small' or ck['fixed_threshold']!=0.5:
        raise RuntimeError('STOP: model or threshold changed')
    if ck['training_grouping']!='31_INFERRED_PHOTO_PARENTS_FROM_FILENAME_AND_IMAGE_SIMILARITY_NOT_HUMAN_VERIFIED':
        raise RuntimeError('STOP: wrong grouping')
    if ck['class_map']!={'actual':0,'faulty':1}:
        raise RuntimeError('STOP: class mapping changed')
    rows=list(csv.DictReader(manifest_bytes.decode('utf-8-sig').splitlines()))
    if len(rows)!=7365: raise RuntimeError('STOP: unexpected manifest count')
    if sum(r['relative_path']==BAD for r in rows)!=1:
        raise RuntimeError('STOP: unexpected image integrity exclusions')
    rows=[r for r in rows if r['relative_path']!=BAD]
    by_group={}
    for r in rows:
        s=r['split']; g=r['inferred_photographed_source']
        if s not in ('train','val','test') or r['label'] not in ('actual','faulty'):
            raise RuntimeError('STOP: invalid split/label')
        if g in by_group and by_group[g]!=s:raise RuntimeError('STOP: group leaked across splits')
        by_group[g]=s
    if Counter(by_group.values())!=Counter({'train':21,'val':5,'test':5}):
        raise RuntimeError('STOP: unexpected photo-root split')
    test=[r for r in rows if r['split']=='test']
    if len(test)!=1086 or Counter(r['label'] for r in test)!=Counter({'actual':422,'faulty':664}):
        raise RuntimeError('STOP: expected untouched test not found')
    # Test data are accessed for predictions only, after all frozen checks pass.
    items=[(r['relative_path'],0 if r['label']=='actual' else 1) for r in test]
    with zipfile.ZipFile(a.archive) as z:
        y,pr=predict_zip(z,items,ck,'mobilenet')
    scores=compute_metrics(y,pr)
    if len(y)!=1086 or not all(0<=x<=1 for x in pr):
        raise RuntimeError('STOP: missing or invalid predictions')
    by_parent={}
    for i,row in enumerate(test):
        g=row['inferred_photographed_source']
        by_parent.setdefault(g,[]).append(i)
    if len(by_parent)!=5:raise RuntimeError('STOP: expected exactly 5 holdout parent groups')
    group_out=[]
    for parent,ix in sorted(by_parent.items()):
        a0=[y[i] for i in ix]; b0=[pr[i]>=0.5 for i in ix]
        tn=sum(t==0 and not p for t,p in zip(a0,b0))
        fp=sum(t==0 and p for t,p in zip(a0,b0))
        fn=sum(t==1 and not p for t,p in zip(a0,b0))
        tp=sum(t==1 and p for t,p in zip(a0,b0))
        group_out.append({'inferred_parent':parent,'n':len(ix),'normal_n':tn+fp,'faulty_n':tp+fn,'tn':tn,'fp':fp,'fn':fn,'tp':tp})
    out={
        'status':'FIRST_AND_ONLY_FROZEN_INFERRED_PHOTO_PARENT_INTERNAL_HOLDOUT',
        'publication_grade':False,
        'source_photo_lineage_human_verified':False,
        'test_is_from_original_thesis_augmented_archive':True,
        'checkpoint_sha256':ck_sha,
        'split_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'original_archive_sha256':source['sha256'],
        'fixed_threshold':0.5,
        'no_training_no_selection_no_calibration':True,
        'test_image_count':len(test),
        'test_inferred_parent_count':len(by_parent),
        'metrics':scores,
        'per_inferred_parent':group_out,
        'caveat':'Only five parent candidates, underlying original photographed sources not manually proved; augmented data; inspect label/domain/generalization validity before submission.'
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('archive','archive-metadata','weights','training-report','manifest','out'):
        p.add_argument('--'+key,required=True)
    main(p.parse_args())
