"""Deterministic inference for released EfficientNet-B0 frozen-encoder weights.

Usage: python code/scripts/predict_clean_effnet.py --checkpoint best.pt --input-dir /path/to/new_photos --output predictions.csv
No training, no threshold tuning. Class 0 normal (actual), class 1 faulty.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision import transforms as T
from torchvision.models import efficientnet_b0


def load_model(weights):
    ck=torch.load(weights,map_location='cpu',weights_only=True)
    assert ck['arch']=='efficientnet_b0_frozen_encoder'
    assert ck['fixed_threshold']==0.5
    assert ck['class_map']=={'actual':0,'faulty':1}
    assert ck['training_grouping']=='31_FILENAME_INFERRED_PHOTO_PARENTS_NOT_INDEPENDENTLY_VERIFIED'
    encoder=efficientnet_b0(weights=None).features
    encoder.load_state_dict(ck['feature_state'],strict=True)
    encoder.eval()
    pre=ck['preprocess']
    transform=T.Compose([T.Resize(tuple(pre['resize']),interpolation=T.InterpolationMode.BILINEAR),
                         T.ToTensor(),T.Normalize(mean=tuple(pre['mean']),std=tuple(pre['std']))])
    head={k:v.detach().cpu().numpy() for k,v in ck['head'].items()}
    return encoder,transform,head


def run(args):
    torch.set_num_threads(4)
    encoder,transform,h=load_model(args.checkpoint)
    paths=sorted(p for p in Path(args.input_dir).rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if not paths:raise RuntimeError('STOP: no image files')
    results=[]
    with torch.no_grad():
        for pos in range(0,len(paths),args.batch_size):
            block=paths[pos:pos+args.batch_size]
            tensors=[]
            for path in block:
                with Image.open(path) as im:
                    tensors.append(transform(ImageOps.exif_transpose(im).convert('RGB')))
            feats=encoder(torch.stack(tensors)).mean((2,3)).numpy().astype(np.float64)
            z=((feats-h['mean'])/h['scale'])@h['coef']+float(h['intercept'])
            probs=1/(1+np.exp(-np.clip(z,-50,50)))
            for path,prob in zip(block,probs):
                results.append([str(path),float(prob),'faulty' if prob>=0.5 else 'actual'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['file','probability_faulty','predicted_class']);wr.writerows(results)
    print(f'Predicted {len(results)} images -> {out}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=16)
    run(p.parse_args())