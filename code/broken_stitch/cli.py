"""CLI for reproducible manuscript reconstruction experiments."""
import argparse,json
from pathlib import Path
from .data import inspect_to_csv,create_group_split,audit_manifest
from .engine import train,evaluate,evaluate_ensemble
from .external import make_external_manifest,evaluate_external


def parser():
    p=argparse.ArgumentParser(description='Broken stitch detection thesis code reconstruction')
    sub=p.add_subparsers(dest='cmd',required=True)
    inv=sub.add_parser('inventory',help='List image paths and infer class folders; leave source_id blank')
    inv.add_argument('--root',required=True);inv.add_argument('--out',required=True)
    sp=sub.add_parser('split',help='Create group-exclusive train/val/test split')
    sp.add_argument('--manifest',required=True);sp.add_argument('--out',required=True)
    sp.add_argument('--seed',type=int,default=42);sp.add_argument('--attempts',type=int,default=200)
    au=sub.add_parser('audit');au.add_argument('--manifest',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--manifest',required=True);tr.add_argument('--outdir',required=True)
    tr.add_argument('--model',choices=['cnn','vgg_linear','vgg_finetune'],default='cnn')
    tr.add_argument('--activation',choices=['relu','softmax','tanh','sigmoid','selu','elu'],default='relu')
    tr.add_argument('--filter',choices=['none','gaussian','bilateral','median'],default='none')
    tr.add_argument('--epochs',type=int,default=30);tr.add_argument('--patience',type=int,default=6)
    tr.add_argument('--image-size',type=int,default=None)
    tr.add_argument('--batch-size',type=int,default=16);tr.add_argument('--workers',type=int,default=0)
    tr.add_argument('--lr',type=float,default=1e-3);tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--device',default='auto');tr.add_argument('--vgg-weights',default=None)
    tr.add_argument('--allow-download',action='store_true',help='Download ImageNet VGG16 weights on first use')
    tr.add_argument('--allow-random-vgg',action='store_true',help='SMOKE TESTS ONLY: do not use these scores in the paper')
    ev=sub.add_parser('evaluate');ev.add_argument('--manifest',required=True)
    ev.add_argument('--checkpoint',required=True);ev.add_argument('--split',choices=['val','test'],default='test')
    ev.add_argument('--out',required=False)
    ens=sub.add_parser('ensemble');ens.add_argument('--manifest',required=True)
    ens.add_argument('--checkpoints',nargs='+',required=True);ens.add_argument('--split',choices=['val','test'],default='test')
    ens.add_argument('--out',required=False)
    exm=sub.add_parser('external-manifest',help='Create labeled test manifest from MANUALLY VERIFIED StitchingNet normal/broken dirs')
    exm.add_argument('--normal-dir',required=True);exm.add_argument('--broken-dir',required=True);exm.add_argument('--out',required=True)
    ext=sub.add_parser('external-evaluate',help='FROZEN pretrained checkpoint evaluation, no tuning on external test')
    ext.add_argument('--checkpoint',required=True);ext.add_argument('--internal-manifest',required=True)
    ext.add_argument('--external-manifest',required=True);ext.add_argument('--out',required=True)
    ext.add_argument('--device',default='auto')
    return p


def main():
    args=parser().parse_args()
    if args.cmd=='inventory':
        x=inspect_to_csv(args.root,args.out)
        print(json.dumps({'images':len(x),'unverified_sources':len(x),'manifest':args.out}))
    elif args.cmd=='split':
        x=create_group_split(args.manifest,args.out,seed=args.seed,attempts=args.attempts)
        print(pd_tab(x))
    elif args.cmd=='audit':print(json.dumps(audit_manifest(args.manifest),indent=2))
    elif args.cmd=='train':
        result=train(args.manifest,args.outdir,model_name=args.model,activation=args.activation,
                     filter_name=args.filter,seed=args.seed,epochs=args.epochs,patience=args.patience,
                     batch_size=args.batch_size,workers=args.workers,lr=args.lr,image_size=args.image_size,
                     pretrained_weights=args.vgg_weights,allow_download=args.allow_download,
                     allow_random_vgg=args.allow_random_vgg,device=args.device)
        print(json.dumps(result,indent=2))
    elif args.cmd=='evaluate':print(json.dumps(evaluate(args.checkpoint,args.manifest,args.split,args.out),indent=2))
    elif args.cmd=='ensemble':print(json.dumps(evaluate_ensemble(args.checkpoints,args.manifest,args.split,args.out),indent=2))
    elif args.cmd=='external-manifest':print(json.dumps(make_external_manifest(args.normal_dir,args.broken_dir,args.out),indent=2))
    elif args.cmd=='external-evaluate':print(json.dumps(evaluate_external(args.checkpoint,args.internal_manifest,args.external_manifest,args.out,device=args.device),indent=2))


def pd_tab(df):
    return df.groupby(['split','label']).size().to_string()

if __name__=='__main__':main()