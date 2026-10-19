"""Fit an EfficientNet-B0 (ImageNet1K-V1) frozen encoder with a supervised
photo-parent/class-balanced logistic classification head on the cleaned v2 ZIPs.

Validation only is assessed. Previously inspected test.zip and StitchingNet are
NEVER opened. Parent groups from Photoshop filenames are not human-proven.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from broken_stitch.metrics import compute_metrics

SEED = 20261019
EXTRACTOR = 'torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1'
DATA_VERSION = 'v2_inferred_photo_parent_exclusive_NOT_HUMAN_VERIFIED'
EXPECTED = {'train': {'actual':1878, 'faulty':3141},
            'val': {'actual':433,'faulty':826},
            'test': {'actual':422,'faulty':664}}
MEAN = (.485,.456,.406)
STD = (.229,.224,.225)
SIZE = 224
PRE = T.Compose([T.Resize((SIZE,SIZE),interpolation=T.InterpolationMode.BILINEAR),
                 T.ToTensor(),T.Normalize(mean=MEAN,std=STD)])


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):
            h.update(b)
    return h.hexdigest()


def validate_inputs(report, manifest, paths):
    if report['version']!=DATA_VERSION:
        raise ValueError('STOP: unexpected cleaned dataset version')
    if report['inferred_photo_parent_cross_split_overlaps']!=0 or report['exact_cross_split_byte_duplicates']!=0:
        raise ValueError('STOP: audit shows leakage')
    rows=list(csv.DictReader(io.StringIO(Path(manifest).read_text(encoding='utf-8-sig'))))
    if len(rows)!=7364:raise ValueError('STOP: unexpected manifest count')
    manifest_sha=digest(manifest)
    parents={}
    buckets=defaultdict(list)
    sets={k:set() for k in ('train','val','test')}
    for i,r in enumerate(rows):
        sp=r['split'];label=r['label'];group=r['inferred_photographed_source'];p=r['clean_path']
        if sp not in sets or label not in ('actual','faulty'):
            raise ValueError('STOP: invalid split/label in manifest')
        if not group.startswith('photo_candidate:'):
            raise ValueError('STOP: unexpected parent identifier')
        if p in sets[sp] or not p.startswith(f'{sp}/{label}/') or '..' in Path(p).parts:
            raise ValueError('STOP: invalid/colliding relative path')
        if group in parents and parents[group]!=sp:
            raise ValueError('STOP: parent crossing splits')
        parents[group]=sp
        sets[sp].add(p)
        buckets[(sp,group,label)].append(i)
    if Counter(parents.values())!=Counter({'train':21,'val':5,'test':5}):
        raise ValueError('STOP: expected 21/5/5 parent partition')
    for sp in sets:
        subset=[r for r in rows if r['split']==sp]
        if Counter(x['label'] for x in subset)!=Counter(EXPECTED[sp]):
            raise ValueError('STOP: dataset class sizes differ')
        if report['images_by_split_label'][sp]!=EXPECTED[sp]:
            raise ValueError('STOP: quality report partition sizes differ')
        rec=report['archives'][sp]
        if rec['name']!=sp+'.zip':raise ValueError('STOP: archive reference differs')
        if sp!='test':
            file=paths[sp]
            if file.stat().st_size!=rec['bytes'] or digest(file)!=rec['sha256']:
                raise ValueError('STOP: '+sp+' ZIP hash/size mismatch')
            with zipfile.ZipFile(file) as z:
                if len(z.namelist())!=len(sets[sp]) or set(z.namelist())!=sets[sp]:
                    raise ValueError('STOP: zip members do not match clean manifest')
                # Hash each image upon reading in features_from_zip.
    return rows,manifest_sha,buckets,parents


def weights_by_photo_and_class(rows):
    """Class-balanced, per-photo-class balanced weights (only train records)."""
    b=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']!='train':
            raise ValueError('Weights can use train images only')
        b[(r['label'],r['inferred_photographed_source'])].append(i)
    groups_by_label=Counter(label for label,source in b)
    if set(groups_by_label)!={'actual','faulty'}:
        raise ValueError('Missing train class')
    w=np.zeros(len(rows),dtype=np.float64)
    for (label,parent),positions in b.items():
        for i in positions:
            w[i]=0.5/(groups_by_label[label]*len(positions))
    if not np.isclose(w.sum(),1.0):
        raise ValueError('Train weights do not total one')
    if any(np.sum(w[np.array([r['label']==lab for r in rows])])<.4999 for lab in ('actual','faulty')):
        raise ValueError('Class weights not balanced')
    return w*len(rows)


def features_from_zip(encoder,zip_path,rows,batch_size=20):
    result=[]; labels=[]; names=[]
    with zipfile.ZipFile(zip_path) as z, torch.no_grad():
        for off in range(0,len(rows),batch_size):
            block=rows[off:off+batch_size]
            tensors=[]
            for r in block:
                payload=z.read(r['clean_path'])
                if hashlib.sha256(payload).hexdigest()!=r['sha256']:
                    raise ValueError('STOP: image differs from clean manifest')
                with Image.open(io.BytesIO(payload)) as im:
                    tensors.append(PRE(ImageOps.exif_transpose(im).convert('RGB')))
            outputs=encoder(torch.stack(tensors)).mean((2,3)).cpu().numpy()
            result.append(outputs)
            labels.extend(0 if r['label']=='actual' else 1 for r in block)
            names.extend(r['clean_path'] for r in block)
            if off%(batch_size*40)==0:print(f'Encoder: {Path(zip_path).name} {off}/{len(rows)}',flush=True)
    return np.concatenate(result).astype(np.float32),np.array(labels,dtype=np.int64),names


def logistic_predict_numpy(feats,head):
    ft=np.asarray(feats,dtype=np.float64)
    mu=np.asarray(head['mean'],dtype=np.float64)
    sc=np.asarray(head['scale'],dtype=np.float64)
    coef=np.asarray(head['coef'],dtype=np.float64)
    bias=float(head['intercept'])
    logits=((ft-mu)/sc)@coef+bias
    return 1/(1+np.exp(-np.clip(logits,-50,50)))


def run(args):
    np.random.seed(SEED);random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4)
    paths={'train':Path(args.train_zip),'val':Path(args.val_zip)}
    report=json.loads(Path(args.quality_report).read_text())
    rows,manifest_sha,buckets,parents=validate_inputs(report,args.manifest,paths)
    subsets={s:sorted((r for r in rows if r['split']==s),key=lambda r:r['clean_path']) for s in ('train','val')}
    # Clear protection: this process only receives train.zip and val.zip; the test archive is absent.
    encoder=efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
    encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    Xt,yt,tnames=features_from_zip(encoder,paths['train'],subsets['train'])
    Xv,yv,vnames=features_from_zip(encoder,paths['val'],subsets['val'])
    if Xt.shape!=(5019,1280) or Xv.shape!=(1259,1280):
        raise ValueError('STOP: unexpected encoder feature shape')
    ww=weights_by_photo_and_class(subsets['train'])
    scaler=StandardScaler().fit(Xt,sample_weight=ww)
    classifier=LogisticRegression(C=1.0,max_iter=500,solver='lbfgs',random_state=SEED)
    classifier.fit(scaler.transform(Xt),yt,sample_weight=ww)
    head={'coef':classifier.coef_[0].astype(np.float64),
          'intercept':np.float64(classifier.intercept_[0]),
          'mean':scaler.mean_.astype(np.float64),
          'scale':scaler.scale_.astype(np.float64)}
    val_prob=logistic_predict_numpy(Xv,head)
    val_score=compute_metrics(yv,val_prob)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ck={
      'arch':'efficientnet_b0_frozen_encoder',
      'pretrained_weights':EXTRACTOR,
      'feature_state':{k:v.detach().cpu() for k,v in encoder.state_dict().items()},
      'head':{k:torch.as_tensor(v).clone() for k,v in head.items()},
      'class_map':{'actual':0,'faulty':1},
      'fixed_threshold':0.5,
      'seed':SEED,
      'preprocess':{'resize':[SIZE,SIZE],'mean':MEAN,'std':STD},
      'manifest_sha256':manifest_sha,
      'original_data_sha256':report['source_archive_sha256'],
      'dataset_quality_report':report['version'],
      'training_grouping':'31_FILENAME_INFERRED_PHOTO_PARENTS_NOT_INDEPENDENTLY_VERIFIED',
      'selection_policy':'fixed ImageNet-pretrained EfficientNet-B0 frozen features; train-only weighted StandardScaler; parent/class-balanced logistic regression C=1.0; threshold .5; validation reporting only',
      'test_accessed':False,'external_stitchingnet_accessed':False,
    }
    cp=out/'efficientnet_b0_clean_v2.pt'
    torch.save(ck,cp)
    prediction=out/'validation_predictions.csv'
    with open(prediction,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['clean_path','true_class','prob_faulty','predicted_class'])
        for row,p in zip(subsets['val'],val_prob):
            w.writerow([row['clean_path'],row['label'],float(p),'faulty' if p>=.5 else 'actual'])
    result={
      'name':'EfficientNet-B0 ImageNet-frozen feature extractor + supervised group-balanced logistic head',
      'status':'TRAINED_CLEANED_V2_TRAIN_ONLY_VALIDATED_ON_VAL_TEST_UNTOUCHED_BY_THIS_RUN',
      'validations_from_development_only':True,
      'train_images':len(Xt),'validation_images':len(Xv),
      'test_images_NOT_OPENED':report['images_by_split_label']['test'],
      'train_groups':len({r['inferred_photographed_source'] for r in subsets['train']}),
      'validation_groups':len({r['inferred_photographed_source'] for r in subsets['val']}),
      'weights_train_per_photo_class':'Each candidate photo contributes equally within each present class, and classes have equal aggregate weight',
      'training_params':{'image_size':SIZE,'pretrained':EXTRACTOR,'encoder_trainable':False,'head':'weighted logistic regression','C':1.0,'fixed_threshold':0.5,'seed':SEED},
      'sha256_clean_manifest':manifest_sha,
      'sha256_train_zip':report['archives']['train']['sha256'],
      'sha256_val_zip':report['archives']['val']['sha256'],
      'sha256_checkpoint':digest(cp),
      'checkpoint_bytes':cp.stat().st_size,
      'validation_metrics':val_score,
      'reference_model':'Earlier MobileNet pretrained on same 31-root inferred split; its validation score was macro-F1 0.6419 / balanced accuracy 0.6725 (compare only cautiously; preprocessing/backbone and class/sample weighting differ).',
      'caveats':['All images in val are historical augmented/Photoshop-edited, not raw unedited independent photographs',
                 '31 photographed parent roots are inferred from filename, not independently confirmed',
                 'Only five validation parents; pooled image metrics may hide source-level heterogeneity',
                 'Prior old internal holdout and StitchingNet already inspected; neither was used in this run',
                 'Do not claim novel independent test generalization or deployment readiness'],
    }
    (out/'training_report.json').write_text(json.dumps(result,indent=2))
    print('TRAINING_REPORT',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--train-zip',required=True)
    a.add_argument('--val-zip',required=True)
    a.add_argument('--manifest',required=True)
    a.add_argument('--quality-report',required=True)
    a.add_argument('--out',required=True)
    run(a.parse_args())