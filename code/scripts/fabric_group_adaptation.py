"""Fabric-held-out REAL-DATA adaptation, exploratory (StitchingNet already inspected).

Compare two initializations with the *same* target-domain 5-fold grouped CV:
  imageNet: torchvision ImageNet MobileNetV3-Small features, new head.
  synthetic: archived 224px/12-epoch thesis-trained MobileNetV3 features & head.

Only normal(00) and broken(02) classes, 11 entire fabric groups. Evaluation
fabric is never used for training in that fold. This is NOT zero-shot transfer,
not a new independent blind test, and no physical factory deploy claim.
"""
import argparse, csv, hashlib, io, json, random, os, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image,ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score,accuracy_score,confusion_matrix,roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parent))
from grouped_finetune_cv import FineTuneBinary
import evaluate_stitchingnet_finetuned as prior_external

NAME='stitchingnet_fabric_held_out_adaptation_v1'
FABRICS=tuple('ABCDEFGHIJK')
SEED=20261020
FOLDS=5
EPOCHS=6
BATCH=16
SIZE=224
EXT_MD5='51b228d9572b925035e9e9af01c96569'
SYNTH_SHA='0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81'
ARMS=('imagenet','synthetic')
EVAL_TF=T.Compose([T.Resize((SIZE,SIZE)),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
TRAIN_TF=T.Compose([T.Resize((SIZE,SIZE)),T.RandomRotation(8),T.ColorJitter(brightness=.08,contrast=.08),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])


def sha(path,hashname='sha256'):
    d=hashlib.new(hashname)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):d.update(b)
    return d.hexdigest()


def extract_rows(archive):
    if sha(archive,'md5')!=EXT_MD5:raise ValueError('STOP: StitchingNet original archive digest differs')
    rows=prior_external.list_binary_members(archive)
    assert len(rows)==3637
    assert set(r[2] for r in rows)==set(FABRICS)
    return rows


def grouped_folds(rows):
    y=np.array([r[1] for r in rows],dtype=np.int64)
    groups=np.array([r[2] for r in rows])
    cv=StratifiedGroupKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
    assignments=np.full(len(rows),-1,dtype=int)
    folds=[]
    for k,(tr,va) in enumerate(cv.split(np.arange(len(rows)),y,groups)):
        seen=set(groups[tr]);held=set(groups[va])
        if seen&held or set(y[tr])!={0,1} or set(y[va])!={0,1}:
            raise ValueError('STOP: invalid fabric-held-out CV group/class split')
        if set(assignments[va])!={-1}:raise ValueError('STOP: sample assigned twice')
        assignments[va]=k
        folds.append((tr,va,sorted(seen),sorted(held)))
    if (assignments<0).any() or set(assignments)!=set(range(FOLDS)):
        raise ValueError('STOP: incomplete OOF group folds')
    for fab in FABRICS:
        ii=np.where(groups==fab)[0]
        if len(set(assignments[ii]))!=1:raise ValueError('STOP: fabric group crosses OOF folds')
    return folds,assignments


class FabricZip(Dataset):
    def __init__(self,archive,rows,idx,training):
        self.archive=str(archive);self.rows=rows;self.idx=[int(i) for i in idx]
        self.training=training;self.handle=None
    def __len__(self):return len(self.idx)
    def __getitem__(self,k):
        if self.handle is None:self.handle=zipfile.ZipFile(self.archive,'r')
        name,label,fabric=self.rows[self.idx[k]]
        data=self.handle.read(name)
        with Image.open(io.BytesIO(data)) as im:
            x=(TRAIN_TF if self.training else EVAL_TF)(ImageOps.exif_transpose(im).convert('RGB'))
        return x,torch.tensor(float(label),dtype=torch.float32),self.idx[k]
    def close(self):
        if self.handle:self.handle.close();self.handle=None


def build_model(arm,synthetic_weights):
    if arm not in ARMS:raise ValueError('bad model arm')
    model=FineTuneBinary('mobilenetv3',pretrained=(arm=='imagenet'))
    if arm=='synthetic':
        if sha(synthetic_weights)!=SYNTH_SHA:raise ValueError('STOP: synthetic checkpoint digest mismatch')
        ck=torch.load(synthetic_weights,weights_only=True,map_location='cpu')
        if ck.get('architecture')!='mobilenetv3' or ck.get('image_size')!=224 or ck.get('epochs')!=12 or ck.get('dev_parent_count')!=26:
            raise ValueError('STOP: wrong synthetic checkpoint provenance')
        model.features.load_state_dict(ck['features'],strict=True)
        model.head.load_state_dict(ck['head'],strict=True)
    return model


def scores(y,p):
    labels=np.asarray(y,dtype=int);probs=np.asarray(p,dtype=float)
    if set(labels)!={0,1} or len(labels)!=len(probs) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any():
        raise ValueError('invalid OOF evaluation inputs')
    pred=(probs>=.5).astype(int)
    tn,fp,fn,tp=[int(x) for x in confusion_matrix(labels,pred,labels=[0,1]).ravel()]
    return dict(n=len(labels),macro_f1=float(f1_score(labels,pred,average='macro')),
      balanced_accuracy=float(balanced_accuracy_score(labels,pred)),
      accuracy=float(accuracy_score(labels,pred)),broken_recall=tp/(tp+fn),
      normal_recall=tn/(tn+fp),broken_auc=float(roc_auc_score(labels,probs)),
      tn=tn,fp=fp,fn=fn,tp=tp)


def cam_for_broken_logit(model,x):
    """True Grad-CAM with activation/gradient of the *broken* logit, not a label map."""
    acts=[]
    handle=model.features.register_forward_hook(lambda _module,_inputs,output: acts.append(output))
    try:
        with torch.enable_grad():
            model.eval()
            logits=model(x)
            if len(acts)!=1:raise RuntimeError('Grad-CAM activation missing')
            grad=torch.autograd.grad(logits.sum(),acts[0],retain_graph=False)[0]
            w=grad.mean(dim=(2,3),keepdim=True)
            cam=torch.relu((acts[0].detach()*w).sum(1,keepdim=True))
            cam=F.interpolate(cam,size=x.shape[-2:],mode='bilinear',align_corners=False)
            arr=cam[0,0].cpu().detach().numpy()
            arr=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
            return arr
    finally:
        handle.remove()


def xai_examples(model,archive,rows,idx,probs,out,arm,fold):
    """Export 2 heatmaps-only (no third party original image pixels) + class error metadata.
    Grad-CAM value is *not* ground-truth defect-localization accuracy.
    """
    y=[rows[int(i)][1] for i in idx]
    choices={}
    for k,(true,p) in enumerate(zip(y,probs)):
        pred=int(p>=.5)
        kind='false_negative' if true==1 and pred==0 else ('true_positive' if true==1 and pred==1 else None)
        if kind and kind not in choices:choices[kind]=k
        if len(choices)==2:break
    if not choices:return []
    ds=FabricZip(archive,rows,idx,training=False)
    reports=[]
    try:
        for kind,k in choices.items():
            x,truth,global_i=ds[k]
            heat=cam_for_broken_logit(model,x.unsqueeze(0))
            fn=f'{arm}_fabricfold{fold}_{kind}_gradcam.png'
            # Grayscale is an actual gradient saliency map, not a modified source image.
            Image.fromarray(np.uint8(np.round(255*heat))).save(out/fn)
            reports.append({'image_identifier':rows[int(global_i)][0],
                'fabric':rows[int(global_i)][2],'category':kind,'probability_broken':float(probs[k]),
                'xai_png':fn,'meaning':'Positive broken-class Grad-CAM; visualization only, no ground-truth localization validation'})
    finally:ds.close()
    return reports


def fit_one(args,rows,folds):
    if args.fold not in tuple(str(k) for k in range(FOLDS)):
        raise ValueError('STOP: must specify a held-out fabric fold 0-4')
    if args.arm not in ARMS:raise ValueError('STOP: must choose pretrained initialization arm')
    k=int(args.fold);tr,va,tr_fabrics,va_fabrics=folds[k]
    setseed=SEED+100*k
    random.seed(setseed);np.random.seed(setseed);torch.manual_seed(setseed)
    torch.set_num_threads(4)
    model=build_model(args.arm,args.synthetic_weights)
    model.train(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    # Identical optimizer, folds, classes and sampling per arm.
    optimizer=torch.optim.AdamW([
        {'params':model.head.parameters(),'lr':5e-4},
        {'params':[p for p in model.features.parameters() if p.requires_grad],'lr':1e-4},
    ],weight_decay=1e-4)
    trainlabels=[rows[int(i)][1] for i in tr]
    cnt=Counter(trainlabels)
    if set(cnt)!={0,1}:raise ValueError('Missing train class')
    train_weights={class_id:len(tr)/(2*count) for class_id,count in cnt.items()}
    td=FabricZip(args.stitchingnet_zip,rows,tr,training=True)
    loader=DataLoader(td,batch_size=BATCH,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(setseed+999))
    history=[]
    for ep in range(1,EPOCHS+1):
        model.train(True)
        loss_acc=0.0
        for xx,yy,_ in loader:
            xx,yy=xx.to(device),yy.to(device)
            ww=torch.where(yy==0,torch.full_like(yy,train_weights[0]),torch.full_like(yy,train_weights[1]))
            logits=model(xx)
            loss=(F.binary_cross_entropy_with_logits(logits,yy,reduction='none')*ww).sum()/ww.sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            loss_acc+=loss.item()*len(yy)
        history.append({'epoch':ep,'train_loss_mean':loss_acc/len(tr)})
        print('FABRIC_EPOCH',json.dumps({'arm':args.arm,'fold':k,'epoch':ep,'loss':history[-1]['train_loss_mean']}),flush=True)
    td.close()
    vd=FabricZip(args.stitchingnet_zip,rows,va,training=False)
    probs=[];labels=[];names=[]
    model.eval()
    try:
        with torch.inference_mode():
            for xx,yy,i in DataLoader(vd,batch_size=BATCH,shuffle=False,num_workers=0):
                probs.extend(torch.sigmoid(model(xx.to(device))).detach().cpu().tolist())
                labels.extend(yy.int().tolist());names.extend(i.tolist())
    finally:vd.close()
    if names!=[int(i) for i in va] or labels!=[rows[int(i)][1] for i in va]:
        raise ValueError('STOP: held-out predictions misaligned')
    stat=scores(labels,probs)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    csvpath=out/'fabric_heldout_oof_predictions.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_file','fabric','actual_class','fold','probability_broken'])
        for i,p in zip(va,probs):
            name,label,fab=rows[int(i)];w.writerow([name,fab,'broken' if label else 'normal',k,float(p)])
    # CPU gradients to prevent duplicate graph residency on constrained runners.
    model=model.cpu()
    gradcam=xai_examples(model,args.stitchingnet_zip,rows,va,probs,out,args.arm,k)
    report={'status':'EXPLORATORY_STITCHINGNET_FABRIC_HELD_OUT_ADAPTATION_NOT_BLIND',
      'arm':args.arm,'fold':k,'training_data':'StitchingNet normal/broken only',
      'train_samples':len(tr),'validation_samples':len(va),
      'training_fabrics':tr_fabrics,'held_out_fabrics':va_fabrics,
      'training_label_counts':{str(a):int(b) for a,b in cnt.items()},
      'val_fabric_overlap':0,'epochs':EPOCHS,'resolution':SIZE,'learning_rates':{'head':5e-4,'tail':1e-4},
      'online_aug_train_only':'rotation8,brightness8%,contrast8%',
      'original_thesis_images_in_this_training':0,
      'pretrained_source':'ImageNet1K-V1 only' if args.arm=='imagenet' else 'ImageNet1K-V1 then thesis augmented-source 224px/12epoch refit',
      'quality_metric':stat,'epoch_loss':history,'xai_heatmaps':gradcam,
      'threshold':.5,'dataset_md5':EXT_MD5,
      'note':'StitchingNet was used in earlier project evaluation, so fold scores are exploratory; folds hold out ENTIRE fabric categories but capture/session correlation may remain. No independent untouched dataset. Grad-CAM is qualitative, cannot prove true defect localization.'}
    (out/'fabric_heldout_report.json').write_text(json.dumps(report,indent=2))
    print('FABRIC_FOLD_FINAL',json.dumps({'arm':args.arm,'fold':k,'score':stat,'held_fabrics':va_fabrics}),flush=True)


def main(args):
    if args.epochs!=EPOCHS or args.batch_size!=BATCH:
        raise ValueError('STOP: preregistered fixed adaptation recipe only')
    rows=extract_rows(args.stitchingnet_zip)
    folds,_=grouped_folds(rows)
    fit_one(args,rows,folds)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--stitchingnet-zip',required=True)
    ap.add_argument('--synthetic-weights',required=True)
    ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--fold',choices=[str(x) for x in range(FOLDS)],required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=EPOCHS)
    ap.add_argument('--batch-size',type=int,default=BATCH)
    main(ap.parse_args())