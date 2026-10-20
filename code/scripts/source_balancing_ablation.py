"""Controlled development-only source-aware training ablation, October 2026.

Three locked MobileNetV3 strategies on IDENTICAL 26-parent grouped folds:
 A class_only: image-uniform class-balanced BCE; no online augmentation
 B parent_class: group-and-class-balanced BCE; no online augmentation
 C parent_class_aug: same as B + train-only rotation/brightness/contrast

Every arm uses the identical globally class-stratified sampled training IDs, fixed 2 epochs,
optimizer, ImageNet initialization, training seed, 128px preprocessing, and
five validation parent folds. No access to inspected 5-parent holdout or
StitchingNet. Validating on augmented records from filename-inferred parents
cannot establish independent photographed-source generalization.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import grouped_finetune_cv as prior

ARMS=('class_only', 'parent_class', 'parent_class_aug')
ARCH='mobilenetv3'
SEED=prior.SEED
N_SPLITS=prior.N_SPLITS
EPOCHS=prior.EPOCHS
PER_CLASS=800
BATCH=prior.BATCH


def fixed_class_stratified_sample(rows,indices,fold_seed,n_per_class=PER_CLASS):
    """Sample a fixed class count without conditioning on inferred photo parent.

    This matters: taking an equal number from each parent would already
    equalize parent counts and make the treatment comparison meaningless.
    """
    bucket=collections.defaultdict(list)
    for i in indices:
        bucket[rows[int(i)]['label']].append(int(i))
    if set(bucket)!={'actual','faulty'}:
        raise ValueError('STOP: train missing a class')
    chosen=[]
    for label in ('actual','faulty'):
        candidates=bucket[label]
        if len(candidates)<n_per_class:
            raise ValueError('STOP: source contains too few '+label+' images for locked budget')
        ranked=sorted(candidates,key=lambda j:hashlib.sha256(
            f'{fold_seed}:{rows[j]["clean_path"]}'.encode()).hexdigest())
        chosen.extend(ranked[:n_per_class])
    selected=sorted(chosen)
    if len(selected)!=2*n_per_class or len(set(selected))!=len(selected):
        raise ValueError('STOP: global class sampling violated')
    return selected


def class_only_weights(rows,selected):
    """Each class has weight 0.5; no parent identity enters the weighting."""
    classes=collections.Counter(rows[int(i)]['label'] for i in selected)
    if set(classes)!={'actual','faulty'}: raise ValueError('STOP: missing training class')
    out={int(i): 0.5*len(selected)/classes[rows[int(i)]['label']] for i in selected}
    if not np.isclose(sum(out.values()),len(selected)):
        raise ValueError('STOP: class weight normalization')
    return out


def resolve_arm(arm,rows,chosen):
    if arm not in ARMS: raise ValueError('Invalid ablation arm')
    if len(chosen)!=len(set(chosen)): raise ValueError('Duplicate train sample IDs')
    aug=(arm=='parent_class_aug')
    w=class_only_weights(rows,chosen) if arm=='class_only' else prior.photo_class_weights(rows,chosen)
    if set(w)!=set(chosen): raise ValueError('STOP: weights do not map to selected train IDs')
    if min(w.values())<=0 or not all(np.isfinite(x) for x in w.values()):
        raise ValueError('STOP: invalid train-only weights')
    return w,aug


def fit_fold(arm,rows,tr,va,archives,fold_id,refit=False):
    if len(tr)==0: raise ValueError('Empty train')
    if (not refit) and len(va)==0: raise ValueError('Empty validation fold')
    train_parents={rows[int(i)]['inferred_photographed_source'] for i in tr}
    val_parents={rows[int(i)]['inferred_photographed_source'] for i in va}
    if train_parents&val_parents:raise ValueError('STOP: source group overlap')
    chosen=fixed_class_stratified_sample(rows,tr,fold_id,n_per_class=PER_CLASS)
    weights,augment=resolve_arm(arm,rows,chosen)
    # For fair paired comparison, the chosen sample IDs are identical for all arms.
    selected_sha=hashlib.sha256('\n'.join(rows[i]['clean_path'] for i in chosen).encode()).hexdigest()
    ds=prior.ZipRows(rows,chosen,archives,train=augment,weights=weights)
    torch.manual_seed(SEED+fold_id);random.seed(SEED+fold_id);np.random.seed(SEED+fold_id)
    model=prior.FineTuneBinary(ARCH,pretrained=True).to('cpu')
    model.train(True)
    opt=torch.optim.AdamW([
        {'params':list(model.head.parameters()),'lr':0.001},
        {'params':[p for p in model.features.parameters() if p.requires_grad],'lr':0.0002},
    ],weight_decay=0.0001)
    # Independent RNG for the shuffle order: augmentation cannot change which
    # batch/order each arm sees in later epochs.
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=0,drop_last=False,
                      generator=torch.Generator().manual_seed(SEED+fold_id+999))
    hist=[]
    for epoch in range(EPOCHS):
        model.train(True)
        total=0.;n=0
        for images,labels,ww in loader:
            logits=model(images)
            loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*ww).sum()/ww.sum().clamp(min=1e-5)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            n+=len(labels);total+=float(loss.item())*len(labels)
        hist.append({'epoch':epoch+1,'weighted_train_loss':total/n})
        print('EPOCH',json.dumps({'arm':arm,'fold':fold_id,'epoch':epoch+1,'train_n':len(chosen),'loss':total/n}),flush=True)
    predictions=[];labels=[]
    if not refit:
        dsval=prior.ZipRows(rows,va,archives,train=False)
        model.eval()
        with torch.no_grad():
            for images,truth,_ in DataLoader(dsval,batch_size=BATCH,shuffle=False,num_workers=0):
                predictions.extend(torch.sigmoid(model(images)).cpu().numpy().tolist())
                labels.extend(truth.numpy().astype(int).tolist())
        dsval.close()
    ds.close()
    meta={'training_n':len(chosen),'training_parents':len(train_parents),
          'selected_image_paths_sha256':selected_sha,
          'tail_trainable_parameters':model.tail_parameter_n,
          'head_trainable_parameters':sum(p.numel() for p in model.head.parameters()),
          'epochs':hist,'train_online_augmentation':augment,
          'loss_weighting':'per_parent_per_class' if arm!='class_only' else 'class_only'}
    return model,meta,labels,predictions


def main(args):
    if args.arm not in ARMS or args.epochs!=EPOCHS or args.batch_size!=BATCH or args.per_class!=PER_CLASS:
        raise ValueError('STOP: locked preregistered ablation design')
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    archives={'train':Path(args.train_zip),'val':Path(args.val_zip)}
    rows=prior.verify_dev_inputs(args.manifest,archives)
    if len(rows)!=6278:raise ValueError('STOP: unexpected dev count')
    y=np.array([0 if r['label']=='actual' else 1 for r in rows],dtype=int)
    groups=np.array([r['inferred_photographed_source'] for r in rows])
    if len(set(groups))!=26:raise ValueError('STOP: expected 26 development parents')
    kfold=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=20261019)
    predictions=np.full(len(rows),np.nan,dtype=float)
    assignments=np.full(len(rows),-1,dtype=int)
    reports=[]
    for k,(tr,va) in enumerate(kfold.split(np.arange(len(rows)),y,groups)):
        model,meta,labels,pp=fit_fold(args.arm,rows,tr,va,archives,k)
        if labels!=y[va].tolist() or len(pp)!=len(va):raise ValueError('STOP: OOF label/order mismatch')
        predictions[va]=pp;assignments[va]=k
        r=prior.evaluate(labels,pp)
        reports.append({'fold':k,'validation_image_n':len(va),'validation_parent_n':len(set(groups[va])),
                        'parent_overlap':int(len(set(groups[va])&set(groups[tr]))),
                        'metrics':r,**meta})
        print('FOLD_RESULT',json.dumps({'arm':args.arm,'fold':k,'metrics':r}),flush=True)
        del model
    if (assignments<0).any() or not np.isfinite(predictions).all():
        raise ValueError('STOP: incomplete OOF predictions')
    if any(r['parent_overlap']!=0 for r in reports):raise ValueError('STOP: leaked parent fold')
    pooled=prior.evaluate(y.tolist(),predictions.tolist())
    # Refit model once on the entire DEVELOPMENT pool, not the prior holdout.
    final,refit,_,_=fit_fold(args.arm,rows,np.arange(len(rows)),np.asarray([],dtype=int),archives,99,refit=True)
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    ck={
        'architecture':'mobilenetv3','ablation_arm':args.arm,
        'features':final.features.state_dict(),'head':final.head.state_dict(),
        'label_map':{'actual':0,'faulty':1},'pretrained_imagenet':True,
        'trained_convolutional_tail':True,'tail_trainable_parameters':final.tail_parameter_n,
        'dev_manifest_sha256':prior.EXPECTED_MANIFEST_SHA,
        'image_size':prior.IMAGE_SIZE,'threshold':0.5,'epochs':EPOCHS,
        'sample_cap_per_class_global':PER_CLASS,'validation_metric':'development 5fold grouped OOF',
        'source_photo_ids':'FILENAME_INFERRED_NOT_VERIFIED',
        'prior_test_used':False,'stitchingnet_used':False,
    }
    cp=folder/f'mobilenetv3_{args.arm}_refit.pt';torch.save(ck,cp)
    with (folder/'oof_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty'])
        for i,r in enumerate(rows):
            w.writerow([r['clean_path'],r['label'],r['inferred_photographed_source'],int(assignments[i]),float(predictions[i])])
    report={
        'status':'CONTROLLED_3ARM_SOURCE_REWEIGHTING_AND_TRAIN_AUGMENTATION_ABLATION_DEV_ONLY',
        'arm':args.arm,'architecture':ARCH,
        'train_algorithm':'ImageNet1K MobileNetV3-Small last convolutional layers genuinely fine-tuned with binary head',
        'dev_image_n':len(rows),'inferred_photo_parent_n':len(set(groups)),
        'stratified_group_folds':N_SPLITS,'seed':SEED,
        'epochs':EPOCHS,'image_size':prior.IMAGE_SIZE,'batch_size':BATCH,'global_fixed_samples_per_class':PER_CLASS,
        'optimizer':'AdamW head 0.001 tail 0.0002 weight_decay 0.0001',
        'train_transform':'8 degree rotation + 8% brightness/contrast jitter' if args.arm=='parent_class_aug' else 'deterministic resize and ImageNet norm',
        'training_weights':'class balanced, every selected image equally weighted within class' if args.arm=='class_only' else 'class-and-parent balanced within same globally class-stratified selection',
        'source_balance_intervention_without_parent_capped_sampling':True,
        'threshold':0.5,'train_image_selection':'same globally uniform SHA-seeded 800 records PER CLASS for all arms; no parent-specific sampling cap',
        'same_batch_order_all_arms':True,
        'pooled_oof_metrics':pooled,'folds':reports,'final_refit':refit,
        'manifest_sha256':prior.EXPECTED_MANIFEST_SHA,
        'train_zip_sha256':prior.EXPECTED_ZIP_SHA['train'],
        'val_zip_sha256':prior.EXPECTED_ZIP_SHA['val'],
        'checkpoint_sha256':prior.file_sha(cp),'checkpoint_bytes':cp.stat().st_size,
        'same_holdout_not_opened':True,'stitchingnet_not_opened':True,
        'cautions':['Global fixed-size class-stratified training sampling is identical in all arms; source-parent weighting is the sole additional intervention in B vs A.',
                    'Parents inferred from Photoshop filenames; no verified raw photo ancestry.',
                    'All validation examples are correlated historical augmented/synthetic records; 26 parent candidate units only.',
                    'The prior five-parent test and StitchingNet were already inspected; these results are development-only.',
                    'Post-hoc project hypothesis chosen after previous explorations. This is descriptive, not an independent confirmatory test.'],
    }
    (folder/'ablation_report.json').write_text(json.dumps(report,indent=2))
    print('ABLATION_FINAL',json.dumps({'arm':args.arm,'macro_f1':pooled['macro_f1'],
           'balanced_accuracy':pooled['balanced_accuracy'],'faulty_recall':pooled['faulty_recall'],
           'checkpoint_sha256':report['checkpoint_sha256']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--train-zip',required=True)
    p.add_argument('--val-zip',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=EPOCHS)
    p.add_argument('--batch-size',type=int,default=BATCH)
    p.add_argument('--per-class',type=int,default=PER_CLASS)
    main(p.parse_args())