"""Descriptive, previously-inspected external StitchingNet model transfer evaluation.

No training, calibration, tuning or parameter-selection logic. No threshold scanning.
The StitchingNet dataset has previously been evaluated on a *different* model.
Thus these data are NOT a fresh untouched benchmark and must never be called so.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grouped_finetune_cv import FineTuneBinary, VAL_TF, IMAGE_SIZE

EXTERNAL_MD5='51b228d9572b925035e9e9af01c96569'
SHA={
    'class_only':'851dc9a675b4c149e618e7d1e75a33bb3bfbdd9e0afd8962f6ef43eae123076b',
    'parent_class':'6dbccee3bc47c924d8e73c4008f3665dbb3c1bfd45155eda1839417b047cd6c3',
    'parent_class_aug':'f35c88e0793bb1ae92ed5b49f93bb66b0ebf376feecd8d79ee0c22d88852e73d',
    'mobilenetv3':'5d7d46898175e8c2daed2f32a15f2a5f7462ea81783ced1ca6df658696e89f77',
    'densenet121':'2493c8cc5833106ca5cce826a83ae10afbda593bac804bb995fa4b1f1dca1202',
    'vgg16':'12cb835e5ec7ad4684e2fedae3c7b20b3a4dbd26c4beded0f66a542b65c998af',
}
BASENAME=re.compile(r'^([A-K])_(00|02)_(\d+)\.(?:jpg|jpeg|png)$', re.I)


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def list_binary_members(archive):
    rows=[]
    with zipfile.ZipFile(archive) as z:
        for zi in z.infolist():
            if zi.is_dir(): continue
            m=BASENAME.fullmatch(Path(zi.filename).name)
            if m:
                fab,code,number=m.groups()
                rows.append((zi.filename,0 if code=='00' else 1,fab.upper()))
    rows.sort()
    if Counter(y for _,y,_ in rows)!=Counter({0:2436,1:1201}):
        raise ValueError('STOP: StitchingNet normal/broken class counts differ')
    if len({n for n,_,_ in rows})!=3637 or len(set(f for *_,f in rows))!=11:
        raise ValueError('STOP: duplicate paths or missing fabrics in StitchingNet')
    return rows


class ExternalArchiveImages(Dataset):
    def __init__(self,path,rows):
        self.archive=Path(path);self.rows=rows;self._z=None
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        if self._z is None:self._z=zipfile.ZipFile(self.archive,'r')
        name,label,fabric=self.rows[i]
        raw=self._z.read(name)
        with Image.open(io.BytesIO(raw)) as im:
            img=VAL_TF(ImageOps.exif_transpose(im).convert('RGB'))
        return img, int(label), name, fabric


def load_checkpoint(file,arm):
    if digest(file)!=SHA[arm]:
        raise ValueError('STOP: frozen checkpoint digest differs for '+arm)
    ck=torch.load(file,map_location='cpu',weights_only=True)
    arch='mobilenetv3' if arm in ('class_only','parent_class','parent_class_aug') else arm
    if ck.get('architecture')!=arch or ck.get('image_size')!=IMAGE_SIZE:
        raise ValueError('STOP: architecture/image size differ')
    if ck.get('label_map')!={'actual':0,'faulty':1}:
        raise ValueError('STOP: unexpected label map')
    if arm in ('class_only','parent_class','parent_class_aug'):
        if ck.get('ablation_arm')!=arm or ck.get('threshold')!=0.5:
            raise ValueError('STOP: ablation metadata/threshold changed')
        if ck.get('stitchingnet_used') is not False or ck.get('prior_test_used') is not False:
            raise ValueError('STOP: checkpoint provenance mismatch')
    else:
        if ck.get('fixed_threshold')!=0.5 or ck.get('dev_parent_count')!=26:
            raise ValueError('STOP: fine-tuned checkpoint metadata/threshold changed')
        if ck.get('not_untouched_external') is not True:
            raise ValueError('STOP: checkpoint provenance missing')
    net=FineTuneBinary(arch,pretrained=False)
    net.features.load_state_dict(ck['features'],strict=True)
    net.head.load_state_dict(ck['head'],strict=True)
    net.eval()
    return net


def summary(y,p):
    y=np.asarray(y,dtype=np.int32);p=np.asarray(p,dtype=np.float64)
    if len(y)==0 or set(y)!={0,1} or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('STOP: empty, single-class or invalid scores')
    pred=(p>=0.5).astype(np.int32)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=[int(v) for v in cm.ravel()]
    return {'n':len(y),'normal_n':int((y==0).sum()),'broken_n':int((y==1).sum()),
        'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro')),
        'normal_recall':float(recall_score(y,pred,pos_label=0)),
        'broken_recall':float(recall_score(y,pred,pos_label=1)),
        'broken_precision':float(tp/(tp+fp)) if tp+fp else 0.,
        'roc_auc_broken':float(roc_auc_score(y,p)),
        'average_precision_broken':float(average_precision_score(y,p)),
        'tn':tn,'fp':fp,'fn':fn,'tp':tp,'confusion_matrix_normal_broken':cm.tolist()}


def run(args):
    torch.set_num_threads(4)
    if digest(args.external_zip,'md5')!=EXTERNAL_MD5:
        raise ValueError('STOP: wrong StitchingNet archive MD5')
    rows=list_binary_members(args.external_zip)
    model=load_checkpoint(args.weights,args.model)
    data=ExternalArchiveImages(args.external_zip,rows)
    dl=DataLoader(data,batch_size=20,shuffle=False,num_workers=0)
    labels=[];scores=[];names=[];fabrics=[]
    with torch.no_grad():
        for i,(x,y,name,fabric) in enumerate(dl):
            scores.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
            names.extend(name);fabrics.extend(fabric)
            if i%30==0:print(f'external {args.model}: {len(scores)}/{len(rows)}',flush=True)
    if len(scores)!=3637 or len(labels)!=len(rows):
        raise ValueError('STOP: incomplete evaluation')
    if any(a!=b or c!=d for (a,b,c,d) in zip(names,[r[0] for r in rows],fabrics,[r[2] for r in rows])):
        raise ValueError('STOP: external images misaligned')
    metrics=summary(labels,scores)
    breakdown={}
    for fabric in sorted(set(fabrics)):
        idx=[i for i,x in enumerate(fabrics) if x==fabric]
        breakdown[fabric]=summary([labels[i] for i in idx],[scores[i] for i in idx])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'external_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_file','fabric','actual_class','p_broken','prediction'])
        for name,f,y,p in zip(names,fabrics,labels,scores):
            w.writerow([name,f,'normal' if y==0 else 'broken',float(p),'broken' if p>=.5 else 'normal'])
    record={
        'status':'DESCRIPTIVE_EXTERNAL_TRANSFER_ON_PREVIOUSLY_INSPECTED_STITCHINGNET_NO_TUNING',
        'model':args.model,'checkpoint_sha256':SHA[args.model],
        'external_archive_md5':EXTERNAL_MD5,
        'dataset_source':'https://zenodo.org/records/20352813',
        'external_data_previously_seen_in_different_model_experiment':True,
        'evaluation_untouched_or_blind':False,
        'test_time_training_or_calibration':False,
        'model_and_threshold_frozen_before_this_run':True,
        'threshold':0.5,
        'included_codes':['00','02'],'excluded_codes':['01','03','04','05','06','07','08','09','10'],
        'metrics':metrics,'by_fabric':breakdown,
        'always_predict_normal_accuracy_baseline':2436/3637,
        'caveat': 'Transfer across the historical Photoshop-derived thesis dataset and StitchingNet. StitchingNet was previously evaluated on a different model; these are non-blind descriptive generalization diagnostics, NOT fresh untouched test evidence. Original photographic parent IDs still inferred. No deployment claims.'
    }
    (out/'external_metrics.json').write_text(json.dumps(record,indent=2))
    print('EXTERNAL_RESULT '+json.dumps({'model':args.model,'metrics':metrics},sort_keys=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=sorted(SHA),required=True)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--external-zip',required=True)
    ap.add_argument('--out',required=True)
    run(ap.parse_args())