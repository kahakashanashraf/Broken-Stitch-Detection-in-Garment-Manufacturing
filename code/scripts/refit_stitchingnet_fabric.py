"""Refit best initialization on ALL StitchingNet fabric development groups.
This produces a usable supervised domain-specific checkpoint; it has NO fresh
untouched test score. Selection must be based on archived 5fold fabric OOF only.
"""
import argparse,json,random,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import fabric_group_adaptation as p


def select_arm(report):
    if report.get('status')!='DESCRIPTIVE_SUPERVISED_STITCHINGNET_ADAPTATION_HELDOUT_FABRIC_NOT_BLIND':
        raise ValueError('STOP: comparison source status not recognized')
    if report.get('n_images')!=3637 or report.get('n_fabrics')!=11:
        raise ValueError('STOP: wrong comparison sample/fabric size')
    metrics=report.get('models',{})
    if set(metrics)!=set(p.ARMS):raise ValueError('STOP: missing model arm')
    values={arm:float(metrics[arm]['macro_f1']) for arm in p.ARMS}
    if not all(np.isfinite(s) and 0<=s<=1 for s in values.values()):
        raise ValueError('STOP: invalid cross-validation scores')
    # Deterministic tie-break: prefer simpler ImageNet init for tied CV scores.
    return max(p.ARMS,key=lambda a:(values[a],int(a=='imagenet')))


def run(args):
    report=json.loads(Path(args.comparison).read_text())
    arm=select_arm(report)
    if arm!=args.arm:raise ValueError('STOP: specified arm is NOT OOF-selected winner')
    if args.epochs!=p.EPOCHS or args.batch_size!=p.BATCH:raise ValueError('STOP: fixed refit recipe mismatch')
    torch.set_num_threads(4)
    random.seed(p.SEED+999);np.random.seed(p.SEED+999);torch.manual_seed(p.SEED+999)
    rows=p.extract_rows(args.stitchingnet_zip)
    if len(rows)!=3637 or len({r[2] for r in rows})!=11:raise ValueError('STOP: full StitchingNet fabric set required')
    model=p.build_model(arm,args.synthetic_weights)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    optim=torch.optim.AdamW([{'params':list(model.head.parameters()),'lr':5e-4},
      {'params':[x for x in model.features.parameters() if x.requires_grad],'lr':1e-4}],weight_decay=1e-4)
    ds=p.FabricZip(args.stitchingnet_zip,rows,np.arange(len(rows)),training=True)
    loader=DataLoader(ds,batch_size=p.BATCH,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(p.SEED+1999))
    cnt=Counter(r[1] for r in rows);weights={label:len(rows)/(2*count) for label,count in cnt.items()}
    history=[]
    for epoch in range(1,p.EPOCHS+1):
        model.train(True)
        loss_total=0.
        for xx,yy,_ in loader:
            xx,yy=xx.to(device),yy.to(device)
            ww=torch.where(yy==0,torch.full_like(yy,weights[0]),torch.full_like(yy,weights[1]))
            logits=model(xx)
            loss=(F.binary_cross_entropy_with_logits(logits,yy,reduction='none')*ww).sum()/ww.sum()
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
            loss_total+=loss.item()*len(yy)
        x={'epoch':epoch,'train_loss_mean':loss_total/len(rows)}
        history.append(x)
        print('FABRIC_REFIT_EPOCH',json.dumps(x),flush=True)
    ds.close()
    model=model.cpu().eval()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    weights_path=out/'fabric_adapted_selected_model.pt'
    checkpoint={'architecture':'mobilenetv3','image_size':p.SIZE,'label_map':{'normal':0,'broken':1},
      'features':{k:v.detach().cpu() for k,v in model.features.state_dict().items()},
      'head':{k:v.detach().cpu() for k,v in model.head.state_dict().items()},
      'dataset':'StitchingNet binary subset normal 00 / broken 02','dataset_md5':p.EXT_MD5,
      'training_fabric_count':11,'train_image_count':3637,'initialization':arm,
      'six_epoch_refit':True,'fixed_threshold':.5,
      'selection_comparison':'stitchingnet-fabric-adapt-comparison-v1',
      'no_untouched_test_accuracy_claim':True,
      'no_zero_shot_claim':True}
    torch.save(checkpoint,weights_path)
    rep={'status':'SUPERVISED_DOMAIN_SPECIFIC_REFIT_SELECTED_BY_GROUPED_DEVELOPMENT_CV',
       'arm':arm,'select_source':'stitchingnet-fabric-adapt-comparison-v1',
       'comparison_cv_macro_f1':{a:report['models'][a]['macro_f1'] for a in p.ARMS},
       'weights_sha256':p.sha(weights_path),'weights_bytes':weights_path.stat().st_size,
       'final_train_loss_history':history,'train_images':3637,'fabrics':11,
       'test_metrics_for_final_refit':None,
       'caution':'This refit uses ALL labeled StitchingNet normal and broken images. It has NO held-out test set. CV results belong to fold-specific models, not this final refit. Independent third dataset required to claim new deployed performance.'}
    (out/'domain_refit_provenance.json').write_text(json.dumps(rep,indent=2))
    print('FABRIC_REFIT_COMPLETE',json.dumps({'selected_init':arm,'checkpoint_sha256':rep['weights_sha256'],'CV_macro_f1_development_only':rep['comparison_cv_macro_f1']}),flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--comparison',required=True)
    q.add_argument('--stitchingnet-zip',required=True)
    q.add_argument('--synthetic-weights',required=True)
    q.add_argument('--arm',choices=p.ARMS,required=True)
    q.add_argument('--epochs',type=int,default=p.EPOCHS)
    q.add_argument('--batch-size',type=int,default=p.BATCH)
    q.add_argument('--out',required=True)
    run(q.parse_args())