"""Development-only, matched-sample high-resolution MobileNetV3 experiment v1.

Comparisons prespecified: old 128px/2epoch class_only OOF (existing frozen baseline),
224px/2epoch and 224px/12epoch with EXACTLY the class_only sampled images per fold.
Zero prior holdout / StitchingNet data opened or used for selection. Source parents
are filename-inferred, not human-authenticated photographed-source identities.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_finetune_cv as prior
from source_balancing_ablation import fixed_class_stratified_sample, class_only_weights

SIZE = 224
EPOCHS = 12
MILESTONES = (2, 12)
SEED = prior.SEED
PER_CLASS = 800
BATCH_SIZE = 12
MODEL = 'mobilenetv3'
RUN_ID = 'enhanced_mobilenet_v3_224px_12epochs_development_groupedcv_v1'
DATA_SPLITS = ('train', 'val')
# Source archive digests are from audited and released cleaned-v2 dataset.


def set_preprocess():
    # No online augmentation: identical geometry and normalization in train/val;
    # only image resolution differs from the frozen class_only baseline.
    pre = T.Compose([T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BILINEAR),
                     T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    prior.TRAIN_TF = pre
    prior.VAL_TF = pre


def file_sha(path):
    return prior.file_sha(path)


def fold_members(rows):
    y = np.asarray([0 if r['label']=='actual' else 1 for r in rows], dtype=np.int64)
    groups = np.asarray([r['inferred_photographed_source'] for r in rows])
    if len(y)!=6278 or len(set(groups))!=26:
        raise ValueError('STOP: unexpected development image/group count')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261019)
    folds = list(splitter.split(np.arange(len(rows)), y, groups))
    if len(folds)!=5:
        raise ValueError('STOP: five folds required')
    for k,(tr,va) in enumerate(folds):
        if set(groups[tr]) & set(groups[va]):
            raise ValueError(f'STOP: group leakage in fold {k}')
        if set(y[tr])!={0,1} or set(y[va])!={0,1}:
            raise ValueError(f'STOP: class missing in fold {k}')
    return y,groups,folds


def train_model(rows, train_idx, validation_idx, fold, archives, output, device):
    train_groups = {rows[int(i)]['inferred_photographed_source'] for i in train_idx}
    val_groups = {rows[int(i)]['inferred_photographed_source'] for i in validation_idx}
    if train_groups & val_groups:
        raise ValueError('STOP: fold source overlap')
    sample = fixed_class_stratified_sample(rows,train_idx,fold,n_per_class=PER_CLASS)
    weights = class_only_weights(rows,sample)
    sample_names = [rows[i]['clean_path'] for i in sample]
    selected_hash = hashlib.sha256('\n'.join(sample_names).encode()).hexdigest()
    torch.manual_seed(SEED+fold); np.random.seed(SEED+fold); random.seed(SEED+fold)
    if device=='cuda':torch.cuda.manual_seed_all(SEED+fold)
    model = prior.FineTuneBinary(MODEL,pretrained=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW([
        {'params':list(model.head.parameters()),'lr':0.001},
        {'params':[p for p in model.features.parameters() if p.requires_grad],'lr':0.0002}
    ],weight_decay=0.0001)
    train_ds=prior.ZipRows(rows,sample,archives,train=False,weights=weights)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,
                            generator=torch.Generator().manual_seed(SEED+fold+999))
    history=[]; scores={}
    for epoch in range(1,EPOCHS+1):
        model.train(True)
        train_loss=0.0;total=0
        for images,labels,ww in train_loader:
            images,labels,ww=images.to(device),labels.to(device),ww.to(device)
            logits=model(images)
            loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*ww).sum()/ww.sum().clamp(min=1e-5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total+=len(labels);train_loss+=float(loss.item())*len(labels)
        hist={'epoch':epoch,'train_loss':train_loss/max(total,1),'train_sample_n':len(sample)}
        history.append(hist)
        print('ENHANCED_EPOCH',json.dumps({'fold':fold,**hist}),flush=True)
        if len(validation_idx)>0 and epoch in MILESTONES:
            val_ds=prior.ZipRows(rows,validation_idx,archives,train=False)
            pred=[];truth=[]
            model.eval()
            with torch.no_grad():
                for im,lab,_ in DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0):
                    prob=torch.sigmoid(model(im.to(device))).cpu().numpy().tolist()
                    pred.extend(prob);truth.extend(lab.numpy().astype(int).tolist())
            val_ds.close()
            labels_expected=[0 if rows[int(i)]['label']=='actual' else 1 for i in validation_idx]
            if truth!=labels_expected or len(pred)!=len(validation_idx):
                raise ValueError('STOP: OOF validation prediction alignment error')
            result=prior.evaluate(truth,pred)
            scores[epoch]={'metrics':result,'probs':pred}
            print('ENHANCED_FOLD',json.dumps({'fold':fold,'epoch':epoch,'metric':result}),flush=True)
    train_ds.close()
    output.mkdir(parents=True,exist_ok=True)
    f=output/'epoch_milestone_oof.csv'
    if len(validation_idx)>0:
        if set(scores)!=set(MILESTONES):raise ValueError('STOP: missed scheduled epoch checkpoint')
        with f.open('w',newline='') as h:
            w=csv.writer(h)
            w.writerow(['path','label','inferred_parent','fold','p_faulty_epoch2','p_faulty_epoch12'])
            for j,i in enumerate(validation_idx):
                rr=rows[int(i)]
                w.writerow([rr['clean_path'],rr['label'],rr['inferred_photographed_source'],fold,
                            scores[2]['probs'][j],scores[12]['probs'][j]])
    else:
        checkpoint=output/'mobilenet_v3_224px_12epoch_dev_refit.pt'
        ck={'architecture':MODEL,'features':{k:v.detach().cpu() for k,v in model.features.state_dict().items()},
            'head':{k:v.detach().cpu() for k,v in model.head.state_dict().items()},
            'pretrained':True,'label_map':{'actual':0,'faulty':1},'fixed_threshold':0.5,
            'image_size':SIZE,'epochs':EPOCHS,'trainable_tail_params':model.tail_parameter_n,
            'dev_parent_count':26,'dev_manifest_sha256':prior.EXPECTED_MANIFEST_SHA,
            'not_untouched_external':True,'training_recipe':RUN_ID,
            'trained_on':'development fold refit: fixed class-stratified 800 actual + 800 faulty samples'}
        torch.save(ck,checkpoint)
    report={'protocol':RUN_ID, 'fold':fold,'development_only':True,'size':SIZE,
            'epochs':EPOCHS,'milestone_epochs':list(MILESTONES),'train_examples':len(sample),
            'train_parent_candidates':len(train_groups),'validation_parent_candidates':len(val_groups),
            'val_examples':len(validation_idx),'fold_parent_overlap':0,'selected_image_sha256':selected_hash,
            'class_weighting':'equal actual/faulty, identical sampled records and RNG seed to original class_only ablation',
            'new_training_augmentation':False,'device':device,
            'validation_metrics':{str(epoch):result['metrics'] for epoch,result in scores.items()},
            'epoch_history':history,'source_groups':'filename_inferred_not_human_verified',
            'test_zip_accessed':False,'stitchingnet_accessed':False,
            'caveat':'Both 224px milestone scores are exploratory OOF development metrics, not a new independent test; 26 inferred parents; prior external data were inspected.'}
    if len(validation_idx)==0:
        report['refit_checkpoint']='mobilenet_v3_224px_12epoch_dev_refit.pt'
        report['refit_sha256']=file_sha(checkpoint)
        report['refit_bytes']=checkpoint.stat().st_size
    (output/'milestone_report.json').write_text(json.dumps(report,indent=2))
    print('ENHANCED_FINAL',json.dumps({'fold':fold,'scores':report['validation_metrics'],
           'checkpoint':report.get('refit_checkpoint')}),flush=True)
    return report


def main(args):
    if args.fold not in ('0','1','2','3','4','refit'):
        raise ValueError('STOP: fold must be one of 0–4 or refit')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    archives={'train':Path(args.train_zip),'val':Path(args.val_zip)}
    print('DEVICE',device,flush=True)
    rows=prior.verify_dev_inputs(args.manifest,archives)
    y,groups,folds=fold_members(rows)
    if args.fold=='refit':
        idx=np.arange(len(rows),dtype=int);val=np.asarray([],dtype=int);fid=99
    else:
        fid=int(args.fold);idx,val=folds[fid]
    set_preprocess()
    train_model(rows,idx,val,fid,archives,Path(args.out),device)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fold',required=True)
    p.add_argument('--train-zip',required=True)
    p.add_argument('--val-zip',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())