"""Actually fine-tune pretrained final convolution stages under 5-fold inferred-photo-parent CV.

No use of previously inspected holdout or StitchingNet; validation is pooled OOF on 26
inferred development sources. Each source/class has a fixed 40-image cap in training,
independent of fold validation outcomes. Refit afterwards on all 26 development roots.
Source ancestry is filename-inferred, not verified independent raw photography.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    densenet121, DenseNet121_Weights,
    vgg16, VGG16_Weights,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support
)

SEED = 20261020
SPLITS = ('train', 'val')
EXPECTED_ZIP_SHA = {
    'train': '9ffecc2e2c25cdf948e4b6483f8d6cc3109a045bdc18b653005557d07b23bb69',
    'val': '9f7df5beffc92d7247cc8b96999233ca1337f3e4e6fe2042d57d08de89c4aae6',
}
EXPECTED_MANIFEST_SHA = 'a873092d19c7b47987c7311a561d35cac3ef254d5aa0110e9a62fd1434a0e2c5'
MAX_PER_PARENT_CLASS = 40
N_SPLITS = 5
IMAGE_SIZE = 128
BATCH = 12
EPOCHS = 2

TRAIN_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.RandomRotation(8),
    T.ColorJitter(brightness=0.08, contrast=0.08),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])
VAL_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2<<20),b''):h.update(chunk)
    return h.hexdigest()


def verify_dev_inputs(manifest, archives):
    raw=Path(manifest).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_MANIFEST_SHA:
        raise ValueError('STOP: manifest is not the audited cleaned-v2 version')
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    if len(rows)!=7364:raise ValueError('STOP: manifest length differs')
    parent_split={}; counts=collections.Counter(); parents={}
    dev=[]; seen=set()
    for row in rows:
        path=row['clean_path'];split=row['split'];label=row['label'];parent=row['inferred_photographed_source']
        if split not in ('train','val','test') or label not in ('actual','faulty') or not parent.startswith('photo_candidate:'):
            raise ValueError('STOP: malformed manifest class/parent/split')
        if row['sha256'] in seen: # exact identical across labels or splits is suspicious
            raise ValueError('STOP: repeated exact image SHA')
        seen.add(row['sha256'])
        if parent in parent_split and parent_split[parent]!=split:
            raise ValueError('STOP: candidate root leakage in v2')
        parent_split[parent]=split
        if not path.startswith(f'{split}/{label}/'):
            raise ValueError('STOP: invalid clean file path')
        counts[(split,label)]+=1
        if split in SPLITS:
            dev.append(row)
    expected=collections.Counter({('train','actual'):1878,('train','faulty'):3141,
                                  ('val','actual'):433,('val','faulty'):826,
                                  ('test','actual'):422,('test','faulty'):664})
    if counts!=expected:raise ValueError('STOP: counts mismatch')
    if collections.Counter(parent_split.values())!=collections.Counter({'train':21,'val':5,'test':5}):
        raise ValueError('STOP: parent counts mismatch')
    if len(dev)!=6278 or len({r['inferred_photographed_source'] for r in dev})!=26:
        raise ValueError('STOP: development pool mismatch')
    for s in SPLITS:
        if file_sha(archives[s])!=EXPECTED_ZIP_SHA[s]:
            raise ValueError('STOP: cleaned-v2 zip digest mismatch '+s)
        with zipfile.ZipFile(archives[s]) as z:
            a=set(z.namelist()); b={r['clean_path'] for r in dev if r['split']==s}
            if a!=b or z.testzip() is not None:
                raise ValueError('STOP: archive members or CRC differ from manifest')
    return sorted(dev,key=lambda r:r['clean_path'])


def balanced_parent_cap(rows, indices, fold_seed, cap=MAX_PER_PARENT_CLASS):
    """Deterministic per parent/class sample cap. No validation images are read here."""
    bins=collections.defaultdict(list)
    for j in indices:
        r=rows[int(j)]
        bins[(r['inferred_photographed_source'],r['label'])].append(int(j))
    chosen=[]
    for key, items in sorted(bins.items()):
        # Stable content-based ordering avoids global RNG side effects and repeated sampling.
        ordered=sorted(items,key=lambda j:hashlib.sha256(f"{fold_seed}:{rows[j]['clean_path']}".encode()).hexdigest())
        chosen.extend(ordered[:cap])
    return sorted(chosen)


def photo_class_weights(rows, selected):
    by=collections.defaultdict(list)
    for j in selected:
        r=rows[j];by[(r['label'],r['inferred_photographed_source'])].append(j)
    parents_per_class=collections.Counter(k[0] for k in by)
    if set(parents_per_class)!={'actual','faulty'}:raise ValueError('STOP: missing training class')
    weights={}
    for (lab,parent),ids in by.items():
        for j in ids:
            weights[j]=0.5/(parents_per_class[lab]*len(ids))
    z=sum(weights.values())
    if abs(z-1)>1e-7:raise ValueError('STOP: parent/class weights do not sum to one')
    scale=len(selected)
    return {j:v*scale for j,v in weights.items()}


class ZipRows(Dataset):
    def __init__(self,rows,ids,archives,train=False,weights=None):
        self.rows=rows;self.ids=list(ids);self.archives=archives;self.train=train
        self.weights=weights or {};self.handles={}
    def __len__(self):return len(self.ids)
    def __getitem__(self,k):
        row_id=self.ids[k];r=self.rows[row_id]
        sp=r['split']
        if sp not in self.handles:self.handles[sp]=zipfile.ZipFile(self.archives[sp],'r')
        raw=self.handles[sp].read(r['clean_path'])
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:
            raise ValueError('STOP: a materialized image differs from signed manifest')
        with Image.open(io.BytesIO(raw)) as im:
            image=(TRAIN_TF if self.train else VAL_TF)(ImageOps.exif_transpose(im).convert('RGB'))
        return image,torch.tensor(0 if r['label']=='actual' else 1,dtype=torch.float32),\
            torch.tensor(self.weights.get(row_id,1.0),dtype=torch.float32)
    def close(self):
        for z in self.handles.values():z.close()
        self.handles={}


class FineTuneBinary(nn.Module):
    def __init__(self,model_name,pretrained=True):
        super().__init__()
        self.model_name=model_name
        if model_name=='mobilenetv3':
            base=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=base.features; in_ch=576
            tail=list(self.features[11].parameters())+list(self.features[12].parameters())
        elif model_name=='densenet121':
            base=densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=base.features;in_ch=1024
            tail=list(self.features.denseblock4.denselayer16.parameters())+list(self.features.norm5.parameters())
        elif model_name=='vgg16':
            base=vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features=base.features;in_ch=512
            tail=list(self.features[26].parameters())+list(self.features[28].parameters())
        else:raise ValueError('Unknown model '+model_name)
        for p in self.features.parameters():p.requires_grad_(False)
        for p in tail:p.requires_grad_(True)
        self.head=nn.Linear(in_ch,1)
        self.tail_parameter_n=sum(x.numel() for x in tail)
        if self.tail_parameter_n<=0:raise ValueError('STOP: no pretrained convolutional layer to fine-tune')
        del base
    def train(self,mode=True):
        super().train(mode)
        # BatchNorm running stats are frozen to prevent drift on tiny photo groups;
        # convolutional tail weights retain gradient and are updated by the optimizer.
        for m in self.features.modules():
            if isinstance(m,(nn.BatchNorm2d,nn.SyncBatchNorm)):m.eval()
        return self
    def forward(self,x):
        feats=self.features(x)
        if self.model_name=='densenet121':feats=F.relu(feats,inplace=False)
        return self.head(F.adaptive_avg_pool2d(feats,1).flatten(1)).flatten()


def evaluate(y,p):
    yt=np.asarray(y,dtype=np.int64);probs=np.asarray(p,dtype=np.float64)
    if len(yt)==0 or set(yt)!={0,1} or not np.isfinite(probs).all():
        raise ValueError('STOP: validation labels/probs invalid')
    pred=(probs>=.5).astype(np.int64)
    _,recall,f1,support=precision_recall_fscore_support(yt,pred,labels=[0,1],zero_division=0)
    return {'n':int(len(y)), 'accuracy':float(accuracy_score(yt,pred)),
            'macro_f1':float(f1_score(yt,pred,average='macro')),
            'balanced_accuracy':float(balanced_accuracy_score(yt,pred)),
            'normal_recall':float(recall[0]),'faulty_recall':float(recall[1]),
            'normal_n':int(support[0]),'faulty_n':int(support[1]),
            'auc_faulty':float(roc_auc_score(yt,probs)),
            'confusion_matrix_normal_faulty':confusion_matrix(yt,pred,labels=[0,1]).tolist()}


def fit_once(model_name,rows,tr_indices,va_indices,archives,fold_id,epochs,device='cpu',batch_size=BATCH,
             cap=MAX_PER_PARENT_CLASS,refit=False):
    chosen=balanced_parent_cap(rows,tr_indices,fold_id,cap=cap)
    trainweights=photo_class_weights(rows,chosen)
    train_groups={rows[x]['inferred_photographed_source'] for x in chosen}
    val_groups={rows[int(x)]['inferred_photographed_source'] for x in va_indices}
    if train_groups & val_groups:raise ValueError('STOP: source-photo group leakage at fit')
    ds=ZipRows(rows,chosen,archives,train=True,weights=trainweights)
    torch.manual_seed(SEED+fold_id);np.random.seed(SEED+fold_id);random.seed(SEED+fold_id)
    model=FineTuneBinary(model_name,pretrained=True).to(device)
    model.train(True)
    opt=torch.optim.AdamW([{'params':list(model.head.parameters()),'lr':0.001},
                           {'params':[p for p in model.features.parameters() if p.requires_grad],'lr':0.0002}],
                          weight_decay=0.0001)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False)
    history=[]
    for epoch in range(epochs):
        model.train(True)
        loss_sum=0.;count=0
        for x,y,w in loader:
            x=x.to(device);y=y.to(device);w=w.to(device)
            logits=model(x)
            loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()/w.sum().clamp(min=1e-5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum+=float(loss.item())*len(y);count+=len(y)
        history.append({'epoch':epoch+1,'train_loss_mean':loss_sum/count})
        print(json.dumps({'model':model_name,'fold':fold_id,'epoch':epoch+1,'train_loss':loss_sum/count,'train_n':len(chosen)}),flush=True)
    predictions=[];labels=[]
    if not refit:
        vds=ZipRows(rows,va_indices,archives,train=False)
        vl=DataLoader(vds,batch_size=batch_size,shuffle=False,num_workers=0)
        model.eval()
        with torch.no_grad():
            for x,y,_ in vl:
                pp=torch.sigmoid(model(x.to(device))).cpu().numpy().tolist()
                predictions.extend(pp);labels.extend(y.numpy().astype(int).tolist())
        vds.close()
    ds.close()
    return model,{'train_n':len(chosen),'train_parent_n':len(train_groups),
                  'tail_trainable_parameters':model.tail_parameter_n,
                  'head_trainable_parameters':sum(p.numel() for p in model.head.parameters()),
                  'epoch_history':history},labels,predictions


def run(args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    print('COMPUTE_DEVICE',device,flush=True)
    archives={'train':Path(args.train_zip),'val':Path(args.val_zip)}
    dev=verify_dev_inputs(args.manifest,archives)
    x=np.arange(len(dev));y=np.asarray([0 if r['label']=='actual' else 1 for r in dev])
    groups=np.asarray([r['inferred_photographed_source'] for r in dev])
    cv=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=20261019)
    probs=np.full(len(dev),np.nan);assign=np.full(len(dev),-1);fold_reports=[]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for k,(tr,va) in enumerate(cv.split(x,y,groups)):
        if len(set(groups[tr])&set(groups[va])):raise ValueError('STOP: fold groups leaked')
        if len(set(y[tr]))!=2 or len(set(y[va]))!=2:raise ValueError('STOP: fold missing class')
        model,info,yt,yp=fit_once(args.model,dev,tr,va,archives,k,epochs=args.epochs,
                                  device=device,batch_size=args.batch_size,cap=args.cap)
        if yt!=y[va].tolist():raise ValueError('STOP: sample/pred alignment failed')
        probs[va]=yp;assign[va]=k
        this=evaluate(yt,yp)
        fold_reports.append({'fold':k,**info,'val_n':len(va),'val_parent_n':len(set(groups[va])),
                             'cross_split_parent_n':0,'metrics':this})
        print('FOLD_RESULT',json.dumps({'model':args.model,'fold':k,'score':this}),flush=True)
        del model
    if not np.isfinite(probs).all() or (assign<0).any():raise ValueError('STOP: missing OOF')
    pooled=evaluate(y.tolist(),probs.tolist())
    # OOF results are from five disjoint validation roots, but post-hoc comparison of
    # these three architectures is still developmental selection, not a new test.
    refit_indices=np.arange(len(dev));empty=np.asarray([],dtype=int)
    final,info,_,_=fit_once(args.model,dev,refit_indices,empty,archives,99,epochs=args.epochs,
                             device=device,batch_size=args.batch_size,cap=args.cap,refit=True)
    save={'architecture':args.model,'head':final.head.state_dict(),'features':final.features.state_dict(),
          'label_map':{'actual':0,'faulty':1},'pretrained':True,
          'trainable_tail_params':final.tail_parameter_n,
          'image_size':IMAGE_SIZE,'fixed_threshold':0.5,
          'dev_manifest_sha256':EXPECTED_MANIFEST_SHA,'dev_parent_count':26,
          'trained_on':'all 6278 development images available, capped max 40 per parent-class in refit',
          'not_untouched_external':True,'epochs':args.epochs,'train_aug_only':True}
    ck=out/f'{args.model}_fine_tuned_dev_v1.pt';torch.save(save,ck)
    with (out/'oof_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty'])
        for i,r in enumerate(dev):w.writerow([r['clean_path'],r['label'],r['inferred_photographed_source'],int(assign[i]),float(probs[i])])
    report={'status':'FINE_TUNED_CONV_LAYERS_GROUPED_5FOLD_DEVELOPMENT_ONLY',
            'architecture':args.model,'model_checkpoint':ck.name,
            'compute_device_used':device,
            'checkpoint_sha256':file_sha(ck),'checkpoint_bytes':ck.stat().st_size,
            'pretrained_imagenet':True,'fully_frozen_encoder':False,
            'trainable_tail_parameters':info['tail_trainable_parameters'],
            'total_development_images':len(dev),'inferred_photo_parents_development':26,
            'folds':N_SPLITS,'fixed_epochs':args.epochs,'training_cap_per_parent_class':args.cap,
            'augmentation_training_only':'RandomRotation(8), ColorJitter(0.08,0.08)',
            'image_size':IMAGE_SIZE,'batch_size':args.batch_size,
            'model_selection_on_heldout_or_stitchingnet':False,
            'already_inspected_test_not_read':True,'stitchingnet_not_read':True,
            'manifest_sha256':EXPECTED_MANIFEST_SHA,
            'train_archive_sha256':EXPECTED_ZIP_SHA['train'],
            'val_archive_sha256':EXPECTED_ZIP_SHA['val'],
            'oof_metrics':pooled,'folds_report':fold_reports,
            'final_refit':info,
            'limitations':['Photo parents inferred from filename, no independently authenticated raw photos.',
                           'Training data consists of correlated augmented/Photoshop-edited derivatives.',
                           'Image-weighted pooled OOF overstates number of independent source photographs (26 only).',
                           'Performance chosen after investigating previous holds is development-only, not a fresh external test.',
                           'No heldout/internal previously inspected test or StitchingNet re-evaluation.']}
    (out/'training_cv_report.json').write_text(json.dumps(report,indent=2))
    print('CV_FINAL',json.dumps({'model':args.model,'pooled':pooled,'checkpoint':ck.name,'tail_trainable_parameters':info['tail_trainable_parameters']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['mobilenetv3','densenet121','vgg16'],required=True)
    ap.add_argument('--train-zip',required=True);ap.add_argument('--val-zip',required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=EPOCHS);ap.add_argument('--batch-size',type=int,default=BATCH)
    ap.add_argument('--cap',type=int,default=MAX_PER_PARENT_CLASS)
    a=ap.parse_args()
    if a.epochs!=EPOCHS or a.cap!=MAX_PER_PARENT_CLASS or a.batch_size!=BATCH:
        raise ValueError('STOP: do not change fixed design without new experiment version')
    run(a)