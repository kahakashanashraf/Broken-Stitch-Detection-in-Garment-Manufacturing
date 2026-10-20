"""Run deterministic binary prediction with a real fine-tuned grouped-CV checkpoint.

Usage: python code/scripts/predict_finetuned.py --weights model.pt --images /dir --out scores.csv
The checkpoint is a development-data refit. No independent deployment claim.
"""
import argparse
import csv
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0,str(Path(__file__).resolve().parent))
from grouped_finetune_cv import FineTuneBinary,VAL_TF,IMAGE_SIZE


class ImageFiles(Dataset):
    def __init__(self,folder):
        root=Path(folder)
        if not root.is_dir():raise ValueError('Expected folder containing images')
        self.paths=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'))
        if not self.paths:raise ValueError('No images found')
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:
            return VAL_TF(ImageOps.exif_transpose(im).convert('RGB')),str(self.paths[i])


def load_model(weight_file):
    try:
        c=torch.load(weight_file,map_location='cpu',weights_only=True)
    except Exception as e:
        raise ValueError('Not a compatible safe PyTorch checkpoint') from e
    arch=c.get('architecture')
    if arch not in ('mobilenetv3','densenet121','vgg16') or c.get('image_size')!=IMAGE_SIZE:
        raise ValueError('Unsupported model or preprocessing size')
    if c.get('label_map')!={'actual':0,'faulty':1} or c.get('fixed_threshold')!=0.5:
        raise ValueError('Unexpected class map or threshold')
    if c.get('dev_parent_count')!=26 or not c.get('not_untouched_external'):
        raise ValueError('Unknown checkpoint provenance')
    model=FineTuneBinary(arch,pretrained=False)
    model.features.load_state_dict(c['features'],strict=True)
    model.head.load_state_dict(c['head'],strict=True)
    model.eval()
    return model,arch


def run(args):
    model,arch=load_model(args.weights)
    ds=ImageFiles(args.images)
    dl=DataLoader(ds,batch_size=12,shuffle=False,num_workers=0)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f,torch.no_grad():
        w=csv.writer(f);w.writerow(['image_path','p_faulty','prediction','model'])
        for x,paths in dl:
            p=torch.sigmoid(model(x)).numpy().tolist()
            for path,pr in zip(paths,p):w.writerow([path,float(pr),'faulty' if pr>=0.5 else 'actual',arch])
    print(f'Wrote {len(ds)} predictions to {out}. Model is development trained, NOT deployment cleared.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True);p.add_argument('--images',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())