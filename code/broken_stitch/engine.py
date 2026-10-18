"""Supervised training / locked split evaluation."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import StitchDataset,load_manifest,audit_manifest
from .models import create_model
from .metrics import compute_metrics


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _loader(df,part,config,augment=False):
    data=StitchDataset(df,part,image_size=config['image_size'],filter_kind=config['filter'],
                       augment=augment,vgg=config['model'].startswith('vgg'))
    return DataLoader(data,batch_size=config['batch_size'],shuffle=augment,num_workers=config['workers'],
                      pin_memory=torch.cuda.is_available())


def _predict(model,loader,device):
    model.eval();ys=[];ps=[]
    with torch.no_grad():
        for images,labels in loader:
            logits=model(images.to(device));prob=torch.softmax(logits,dim=1)[:,1]
            ys.extend(labels.numpy().tolist());ps.extend(prob.cpu().numpy().tolist())
    return ys,ps


def _manifest_fingerprint(df):
    # Split membership invariant under dataframe row permutation, relative paths are explicit.
    rows=sorted(f'{getattr(r, "relative_path", None) or r.path}\t{r.label}\t{r.source_id}\t{r.split}' for r in df.itertuples(index=False))
    return hashlib.sha256('\n'.join(rows).encode('utf-8')).hexdigest()


def train(manifest,outdir,model_name='cnn',activation='relu',image_size=None,filter_name='none',
          seed=42,epochs=20,patience=5,batch_size=32,workers=0,lr=1e-3,pretrained_weights=None,
          allow_download=False,allow_random_vgg=False,device='auto',skip_sha_audit=False):
    df=load_manifest(manifest,require_splits=True)
    audit=audit_manifest(manifest,check_images=False,hash_all=not skip_sha_audit)
    if skip_sha_audit:raise ValueError('Bypassing exact-duplicate audit is not supported for reportable training')
    seed_everything(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    config={'model':model_name,'activation':activation,'image_size':image_size or (224 if model_name.startswith('vgg') else 100),
            'filter':filter_name,'seed':seed,'epochs':epochs,'patience':patience,'batch_size':batch_size,'workers':workers,
            'lr':lr,'labels':['actual','faulty'],'split_hash':_manifest_fingerprint(df)}
    if epochs<1 or patience<1 or batch_size<1:raise ValueError('Positive epochs, patience, batch-size required')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    model,weights_source=create_model(model_name,activation,pretrained_weights,allow_download,allow_random_vgg)
    config['weights_source']=weights_source
    config['model_weights_provenance_ok']=('RANDOM_UNTRAINED' not in weights_source)
    config['publication_ready']=False  # requires independent human lineage/rights/results audit
    model=model.to(device)
    train_loader=_loader(df,'train',config,True);val_loader=_loader(df,'val',config,False)
    ytrain=np.array([int(y=='faulty') for y in df.loc[df.split=='train','label']])
    counts=np.bincount(ytrain,minlength=2)
    weights=torch.tensor(len(ytrain)/(2*np.maximum(counts,1)),dtype=torch.float32,device=device)
    loss_fn=nn.CrossEntropyLoss(weight=weights)
    optim=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    best=-float('inf');epochs_without=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();running=0.;count=0
        for images,labels in train_loader:
            images=images.to(device);labels=labels.to(device)
            optim.zero_grad(set_to_none=True)
            logits=model(images);loss=loss_fn(logits,labels)
            loss.backward();optim.step()
            running+=loss.item()*len(labels);count+=len(labels)
        yt,p=_predict(model,val_loader,device)
        mt=compute_metrics(yt,p)
        row={'epoch':epoch,'train_loss':running/max(count,1),'val_macro_f1':mt['macro_f1'],
             'val_balanced_accuracy':mt['balanced_accuracy'],'val_accuracy':mt['accuracy']}
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f}",flush=True)
        if mt['macro_f1']>best+1e-8:
            best=mt['macro_f1'];epochs_without=0
            torch.save({'model_state':model.state_dict(),'config':config,'best_epoch':epoch,
                        'best_validation_metrics':mt,'audit':audit},out/'best.pt')
        else:epochs_without+=1
        if epochs_without>=patience:break
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    return {'best_macro_f1':best,'epochs_ran':len(history),'checkpoint':str(out/'best.pt'),'model_weights_provenance_ok':config['model_weights_provenance_ok'],'publication_ready':False}


def predict_checkpoint(checkpoint,manifest,part='val',device='auto',workers=0,skip_sha_audit=False):
    if part not in ('val','test'):raise ValueError('Evaluation limited to val or test')
    if skip_sha_audit:raise ValueError('Bypassing duplicate audit is unsupported')
    df=load_manifest(manifest,require_splits=True)
    audit_manifest(manifest,check_images=False,hash_all=True)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=ckpt['config']
    if cfg['split_hash']!=_manifest_fingerprint(df):raise ValueError('Manifest differs from training split: REFUSE evaluation')
    # Instantiate VGG from random then load full trained checkpoint; no external weights needed.
    model,_=create_model(cfg['model'],cfg['activation'],allow_random_vgg=True)
    model.load_state_dict(ckpt['model_state'],strict=True)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    cfg={**cfg,'workers':workers}
    data=_loader(df,part,cfg,False)
    y,p=_predict(model,data,device)
    return y,p,ckpt


def evaluate(checkpoint,manifest,part='test',output=None):
    y,p,ckpt=predict_checkpoint(checkpoint,manifest,part)
    out={'split':part,'metrics':compute_metrics(y,p),'checkpoint':str(checkpoint),
         'model_weights_provenance_ok':ckpt['config']['model_weights_provenance_ok'],'publication_ready':False, 'note':'Fresh reconstructed run, never historical thesis values.'}
    if output:Path(output).write_text(json.dumps(out,indent=2),encoding='utf8')
    return out


def evaluate_ensemble(checkpoints,manifest,part='test',output=None):
    if len(checkpoints)<2:raise ValueError('At least two checkpoints required for an ensemble')
    ys=[];probs=[];configs=[]
    for path in checkpoints:
        y,p,ckpt=predict_checkpoint(path,manifest,part)
        ys.append(y);probs.append(p);configs.append(ckpt['config'])
    if any(y!=ys[0] for y in ys):raise ValueError('Ensemble label order mismatch')
    if any(c['image_size']!=configs[0]['image_size'] or c['filter']!=configs[0]['filter'] for c in configs):
        raise ValueError('Ensemble members have incompatible preprocessing')
    result={'split':part,'checkpoints':list(map(str,checkpoints)),
            'metrics':compute_metrics(ys[0],np.array(probs).mean(axis=0)),
            'model_weights_provenance_ok':all(c['model_weights_provenance_ok'] for c in configs),'publication_ready':False,
            'note':'Probability averaging of separately trained members (approximate thesis ensemble).'}
    if output:Path(output).write_text(json.dumps(result,indent=2),encoding='utf8')
    return result