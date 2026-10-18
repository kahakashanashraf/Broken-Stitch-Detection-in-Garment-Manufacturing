"""Precommitted MobileNetV3-Small ImageNet-pretrained frozen-feature binary baseline.
Never accesses internal TEST, StitchingNet, or photo-source lineages during training.
"""
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from broken_stitch.metrics import compute_metrics

IMNET={'mean':(.485,.456,.406),'std':(.229,.224,.225)}
TFRM=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize(**IMNET)])
WEIGHTS_ID='torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1'

def extractor(device='cpu',pretrained=False):
    if pretrained:
        weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
        net=mobilenet_v3_small(weights=weights).features
    else:
        net=mobilenet_v3_small(weights=None).features
    for p in net.parameters():p.requires_grad_(False)
    return net.eval().to(device)

def features_from_zip(net,z,rows,batch_size=24,device='cpu'):
    allfeat=[]; labels=[];fails=[]
    if not rows:return np.empty((0,576),np.float32),np.empty((0,),np.int64),[]
    with torch.no_grad():
        for pos in range(0,len(rows),batch_size):
            batch=rows[pos:pos+batch_size]
            xs=[];ys=[]
            for r in batch:
                try:
                    data=z.read(r['relative_path'])
                    with Image.open(io.BytesIO(data)) as im:
                        x=TFRM(ImageOps.exif_transpose(im).convert('RGB'))
                    xs.append(x);ys.append(0 if r['label']=='actual' else 1)
                except (KeyError,OSError,ValueError) as e:
                    fails.append([r['relative_path'],type(e).__name__]);continue
            if xs:
                out=net(torch.stack(xs).to(device)).mean((2,3)).cpu().numpy()
                allfeat.append(out);labels.extend(ys)
            if pos % (batch_size*30)==0:print(f'features {pos}/{len(rows)}',flush=True)
    if fails:raise ValueError('STOP: images unreadable: '+repr(fails[:3]))
    return np.concatenate(allfeat),np.asarray(labels,dtype=np.int64),fails

def read_manifest(path):
    raw=Path(path).read_bytes()
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if not rows:raise ValueError('Empty split manifest')
    if {r['split'] for r in rows}!={'train','val','test'}:raise ValueError('Must have train/val/test')
    bygroup={}
    for r in rows:
        assert r['label'] in ('actual','faulty')
        k=r['source_id'];s=r['split']
        if k in bygroup and bygroup[k]!=s:raise ValueError('Candidate source groups overlap')
        bygroup[k]=s
    return rows,hashlib.sha256(raw).hexdigest()

def train(zip_path,manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(4)
    rows,sha=read_manifest(manifest)
    subsets={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    if len(subsets['train'])!=5167 or len(subsets['val'])!=1120 or len(subsets['test'])!=1077:
        raise ValueError('Unexpected split size: original pre-registered candidate split must be unchanged')
    net=extractor(pretrained=True)
    with zipfile.ZipFile(zip_path) as z:
        ft,yt,_=features_from_zip(net,z,subsets['train'])
        fv,yv,_=features_from_zip(net,z,subsets['val'])
    scaler=StandardScaler().fit(ft)
    clf=LogisticRegression(C=1.0,class_weight='balanced',solver='lbfgs',max_iter=350,random_state=42)
    clf.fit(scaler.transform(ft),yt)
    pr=clf.predict_proba(scaler.transform(fv))[:,1]
    val=compute_metrics(yv,pr)
    # Torch feature backbone + a fully specified head: strict frozen external inference.
    ck={'arch':'mobilenet_v3_small','torchvision_pretrained':WEIGHTS_ID,
        'feature_state':{k:v.detach().cpu() for k,v in net.state_dict().items()},
        'head':{'coef':torch.from_numpy(clf.coef_.astype(np.float32)),
                'intercept':torch.from_numpy(clf.intercept_.astype(np.float32)),
                'mean':torch.from_numpy(scaler.mean_.astype(np.float32)),
                'scale':torch.from_numpy(scaler.scale_.astype(np.float32))},
        'class_map':{'actual':0,'faulty':1},'fixed_threshold':0.5,
        'manifest_sha256':sha,'training_grouping':'UNVERIFIED_UUID_FILENAME_FAMILY',
        'selection_policy':'One fixed ImageNet feature extractor, C=1.0 balanced logistic head; no hyperparameter selection',
        'preprocess':{'resize':[224,224],'image_net_normalization':IMNET}}
    path=out/'mobilenet_v3_small_frozen_features.pt'
    torch.save(ck,path)
    meta={'checkpoint':path.name,'bytes':path.stat().st_size,
          'checkpoint_sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
          'training_count':len(yt),'validation_count':len(yv), 'test_count_UNTOUCHED':len(subsets['test']),
          'internal_candidate_manifest_sha256':sha,
          'feature_network':'MobileNetV3-Small pretrained ImageNet1K V1 FROZEN',
          'trained_classifier':'balanced logistic regression C=1.0 fixed random_state 42',
          'external_stitchingnet_used':False,'internal_test_used':False,
          'publication_ready':False,'lineage':'candidate groups from UUID filenames, NOT verified photo parents',
          'validation_metrics':val}
    (out/'mobilenet_training_report.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='validation_metrics'},indent=2),flush=True)
    print('VAL_METRICS',json.dumps(val,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset-zip',required=True);p.add_argument('--portable-manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();train(a.dataset_zip,a.portable_manifest,a.out)