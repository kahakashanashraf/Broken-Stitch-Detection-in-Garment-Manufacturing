"""Inference for the *new* 224px/12epoch development refit checkpoint only.

Images are not used for training. The binary threshold is locked to 0.5.
This model is NOT verified for independent industrial deployment.
"""
import argparse
import csv
from pathlib import Path
import sys

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
sys.path.insert(0,str(Path(__file__).resolve().parent))
from grouped_finetune_cv import FineTuneBinary, EXPECTED_MANIFEST_SHA
from enhanced_mobilenet_224_cv import SIZE, EPOCHS, RUN_ID

PRE=T.Compose([T.Resize((SIZE,SIZE),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),
               T.Normalize((.485,.456,.406),(.229,.224,.225))])

class Images(Dataset):
    def __init__(self,directory):
        folder=Path(directory)
        if not folder.is_dir():raise ValueError('Input must be a directory')
        self.paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'))
        if not self.paths:raise ValueError('No supported images found')
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.paths[index]) as im:
            return PRE(ImageOps.exif_transpose(im).convert('RGB')),str(self.paths[index])


def load_enhanced_checkpoint(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    checks={'architecture':'mobilenetv3','image_size':SIZE,'epochs':EPOCHS,
            'label_map':{'actual':0,'faulty':1},'fixed_threshold':.5,
            'dev_manifest_sha256':EXPECTED_MANIFEST_SHA,'dev_parent_count':26,
            'not_untouched_external':True,'training_recipe':RUN_ID}
    for key,value in checks.items():
        if checkpoint.get(key)!=value:raise ValueError('Unexpected enhanced checkpoint provenance '+key)
    model=FineTuneBinary('mobilenetv3',pretrained=False)
    model.features.load_state_dict(checkpoint['features'],strict=True)
    model.head.load_state_dict(checkpoint['head'],strict=True)
    model.eval()
    return model


def run(args):
    model=load_enhanced_checkpoint(args.weights)
    ds=Images(args.images)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),output.open('w',newline='') as stream:
        w=csv.writer(stream)
        w.writerow(['image_path','probability_faulty','prediction','model'])
        for xx,paths in DataLoader(ds,batch_size=12,num_workers=0,shuffle=False):
            pp=torch.sigmoid(model(xx)).cpu().tolist()
            for path,prob in zip(paths,pp):w.writerow([path,prob,'faulty' if prob>=.5 else 'actual',RUN_ID])
    print(f'Predictions saved for {len(ds)} images. Threshold fixed 0.5; no independent deployment guarantee.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--out',required=True)
    run(p.parse_args())