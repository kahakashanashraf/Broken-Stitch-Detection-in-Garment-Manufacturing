"""Run inference with fully StitchingNet-supervised model on new images (no labels).
No external accuracy is implied by the saved refit weights.
"""
import argparse,csv
from pathlib import Path
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset,DataLoader
import fabric_group_adaptation as p

class Img(Dataset):
    def __init__(self,root):
        self.paths=sorted(x for x in Path(root).rglob('*') if x.is_file() and x.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp'))
        if not self.paths:raise ValueError('No images in folder')
    def __len__(self):return len(self.paths)
    def __getitem__(self,k):
        f=self.paths[k]
        with Image.open(f) as im: x=p.EVAL_TF(ImageOps.exif_transpose(im).convert('RGB'))
        return x,str(f)

def load_refit(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('architecture')!='mobilenetv3' or ck.get('image_size')!=224 or ck.get('label_map')!={'normal':0,'broken':1}:
        raise ValueError('Incompatible architecture or label map')
    if ck.get('dataset_md5')!=p.EXT_MD5 or ck.get('training_fabric_count')!=11 or ck.get('train_image_count')!=3637:
        raise ValueError('Unverified supervised StitchingNet refit provenance')
    if ck.get('initialization') not in p.ARMS or ck.get('no_untouched_test_accuracy_claim') is not True:
        raise ValueError('Missing experimental provenance')
    model=p.FineTuneBinary('mobilenetv3',pretrained=False)
    model.features.load_state_dict(ck['features'],strict=True)
    model.head.load_state_dict(ck['head'],strict=True)
    return model.eval()

def run(args):
    model=load_refit(args.weights)
    ds=Img(args.images)
    outfile=Path(args.out);outfile.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(),outfile.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['file','p_broken','predicted_class','training_dataset'])
        for xx,paths in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
            preds=torch.sigmoid(model(xx)).cpu().tolist()
            for path,prob in zip(paths,preds):
                w.writerow([path,float(prob),'broken' if prob>=.5 else 'normal','StitchingNet supervised'])
    print('WROTE_PREDICTIONS',len(ds),'no independent accuracy evaluation done',flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser()
    q.add_argument('--weights',required=True);q.add_argument('--images',required=True);q.add_argument('--out',required=True)
    run(q.parse_args())