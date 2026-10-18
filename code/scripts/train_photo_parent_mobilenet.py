"""Pre-registered photo-PARENT inferred grouped MobileNet experiment.
No use of StitchingNet OR the internal test partition for training/model selection.
This is distinct from earlier 182-UUID-filename-family MobileNet baseline.
"""
import argparse,csv,json,hashlib,zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from broken_stitch.metrics import compute_metrics
import mobilenet_feature_baseline as b

BAD='dataset/train/actual/actual_original_AB.png_579370ba-65d0-4793-a3c0-0f73fbf6c367.png'

def main(archive,source_manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261018);np.random.seed(20261018)
    raw=Path(source_manifest).read_bytes()
    rows=list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(rows)!=7365 or not rows:raise ValueError('Unexpected parent-manifest size')
    imgs=[r for r in rows if r['relative_path']!=BAD]
    if len(imgs)!=7364 or sum(x['relative_path']==BAD for x in rows)!=1:
        raise ValueError('Expected exactly one corrupt image removed by known SHA-bearing pathname')
    groups={}
    for r in imgs:
        grp=r['inferred_photographed_source']; s=r['split']
        assert r['label'] in ('actual','faulty') and s in ('train','val','test')
        if grp in groups and groups[grp]!=s:raise ValueError('Source-group leakage')
        groups[grp]=s
    if Counter(groups.values())!=Counter({'train':21,'val':5,'test':5}):raise ValueError('Unexpected group partition')
    parts={sp:[r for r in imgs if r['split']==sp] for sp in ('train','val','test')}
    for v in parts.values():
        if {r['label'] for r in v}!={'actual','faulty'}:raise ValueError('Missing class')
    model=b.extractor(pretrained=True)
    with zipfile.ZipFile(archive) as z:
        X,y,_=b.features_from_zip(model,z,parts['train'])
        V,yv,_=b.features_from_zip(model,z,parts['val'])
    scaler=StandardScaler().fit(X)
    clf=LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=350,random_state=20261018)
    clf.fit(scaler.transform(X),y)
    probs=clf.predict_proba(scaler.transform(V))[:,1]
    scores=compute_metrics(yv,probs)
    ck={'arch':'mobilenet_v3_small','torchvision_pretrained':b.WEIGHTS_ID,
        'feature_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'head':{'coef':torch.from_numpy(clf.coef_.astype(np.float32)),
                'intercept':torch.from_numpy(clf.intercept_.astype(np.float32)),
                'mean':torch.from_numpy(scaler.mean_.astype(np.float32)),
                'scale':torch.from_numpy(scaler.scale_.astype(np.float32))},
        'class_map':{'actual':0,'faulty':1},'fixed_threshold':0.5,
        'manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'training_grouping':'31_INFERRED_PHOTO_PARENTS_FROM_FILENAME_AND_IMAGE_SIMILARITY_NOT_HUMAN_VERIFIED',
        'selection_policy':'Fixed ImageNet feature extractor, C=1.0 balanced logistic head; no tuning; external unaccessed',
        'preprocess':{'resize':[224,224],'image_net_normalization':b.IMNET}}
    weights=out/'mobilenet_v3_small_photo_parent_inferred.pt';torch.save(ck,weights)
    report={'status':'EXPLORATORY_INFERRED_PHOTO_PARENTS_ONLY','training_count':len(X),'validation_count':len(V),
            'test_count_UNTOUCHED_BY_THIS_MODEL':len(parts['test']),
            'groups_by_split':dict(Counter(groups.values())),
            'labels_by_split':{sp:dict(Counter(r['label'] for r in ls)) for sp,ls in parts.items()},
            'removed_corrupt_file':BAD,'validation_metrics':scores,
            'weights':weights.name,'weights_sha256':hashlib.file_digest(open(weights,'rb'),'sha256').hexdigest(),
            'model_selection_stitchingnet_used':False,'stitchingnet_evaluation_performed':False,
            'caveat':'This is NOT verified photographed-source independence; raw Photoshop original edit provenance missing',
            'comparison':'Cannot compare validation metrics directly with older 182-group partition since group composition differs.'}
    (out/'photo_parent_mobilenet_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();main(a.zip,a.manifest,a.out)