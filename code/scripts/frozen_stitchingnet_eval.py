"""One-shot external evaluation of previously frozen models on StitchingNet normal/broken only.
This script neither fits parameters nor modifies its 0.5 decision threshold.
"""
import argparse,csv,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageOps
from torchvision import transforms as T
from torchvision.models import mobilenet_v3_small
from broken_stitch.metrics import compute_metrics
from broken_stitch.models import create_model

BASENAME=re.compile(r'^[A-K]_(00|02)_(\d+)\.(?:jpg|jpeg|png)$',re.I)
STITCHING_MD5='51b228d9572b925035e9e9af01c96569'
IMAGE_PRE=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
CNN_PRE=T.Compose([T.Resize((100,100),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor()])

def sha(path,algo='sha256'):
    h=hashlib.new(algo)
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def external_list(external):
    with zipfile.ZipFile(external) as z:
        paths=[]
        for n in z.namelist():
            m=BASENAME.fullmatch(Path(n).name)
            if m:paths.append((n,0 if m.group(1)=='00' else 1))
    paths.sort()
    assert sum(label==0 for _,label in paths)==2436
    assert sum(label==1 for _,label in paths)==1201
    return paths

def overlap_check(original,ext,extpaths):
    internal=set()
    with zipfile.ZipFile(original) as z:
        for i,n in enumerate(z.namelist()):
            if n.lower().endswith(('.png','.jpg','.jpeg')) and not n.endswith('/'):
                internal.add(hashlib.sha256(z.read(n)).hexdigest())
    common=[]
    with zipfile.ZipFile(ext) as z:
        for n,_ in extpaths:
            if hashlib.sha256(z.read(n)).hexdigest() in internal:common.append(n)
    if common:raise RuntimeError(f'STOP: {len(common)} exact image overlaps with internal ZIP')
    return {'internal_sha256_image_count':len(internal),'external_binary_images':len(extpaths),'identical_bytes_overlap':0,
            'limitation':'SHA256 does not detect cropped or transformed near-duplicates.'}

def make_backbone_mobilenet(ck):
    assert ck['arch']=='mobilenet_v3_small'
    assert ck['fixed_threshold']==0.5
    assert ck['torchvision_pretrained']=='torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1'
    m=mobilenet_v3_small(weights=None).features
    m.load_state_dict(ck['feature_state'],strict=True);m.eval()
    return m

def predict_zip(z,rows,ck,kind):
    if kind=='mobilenet':
        net=make_backbone_mobilenet(ck)
        h=ck['head'];means=h['mean'].cpu().numpy();scale=h['scale'].cpu().numpy()
        coef=h['coef'].cpu().numpy()[0];bias=float(h['intercept'][0])
        preprocess=IMAGE_PRE
    elif kind=='smallcnn':
        cfg=ck['config']
        assert cfg['model']=='cnn' and cfg['image_size']==100
        assert cfg['filter']=='none'
        net,_=create_model('cnn',cfg['activation'])
        net.load_state_dict(ck['model_state'],strict=True);net.eval()
        preprocess=CNN_PRE
    else:raise ValueError(kind)
    net.eval();torch.set_num_threads(4)
    ys=[];probs=[];bad=[]
    with torch.no_grad():
        for i in range(0,len(rows),24):
            batch=rows[i:i+24]
            images=[]
            for name,label in batch:
                try:
                    data=z.read(name)
                    with Image.open(io.BytesIO(data)) as im:
                        images.append(preprocess(ImageOps.exif_transpose(im).convert('RGB')))
                except Exception as e:
                    bad.append((name,type(e).__name__));continue
            if bad:raise RuntimeError('STOP: invalid external images: '+repr(bad[:3]))
            x=torch.stack(images)
            if kind=='mobilenet':
                ft=net(x).mean((2,3)).numpy()
                logit=((ft-means)/scale)@coef+bias
                p=1.0/(1+np.exp(-np.clip(logit,-45,45)))
                probs.extend(p.tolist())
            else:
                p=net(x).softmax(dim=1)[:,1].numpy()
                probs.extend(p.tolist())
            ys.extend(label for _,label in batch)
            if i%720==0:print(f'external {kind}: {i}/{len(rows)}',flush=True)
    return ys,probs

def evaluate(args):
    # Model selection frozen BEFORE this external file is touched for performance evaluation.
    assert sha(args.external_zip,'md5')==STITCHING_MD5
    mobmeta=json.loads(Path(args.mobilenet_report).read_text())
    assert sha(args.mobilenet_weights)==mobmeta['checkpoint_sha256']
    assert mobmeta['external_stitchingnet_used'] is False
    assert mobmeta['internal_test_used'] is False
    assert sha(args.portable_manifest)==mobmeta['internal_candidate_manifest_sha256']
    cnnmeta=json.loads(Path(args.cnn_provenance).read_text())
    assert sha(args.cnn_weights)==cnnmeta['checkpoint_sha256']
    rows=external_list(args.external_zip)
    cross=overlap_check(args.internal_zip,args.external_zip,rows)
    rep={'label':'FIRST_FROZEN_EXTERNAL_TEST_NO_TUNING','source':'StitchingNet Zenodo 10.5281/zenodo.20352813',
         'license':'CC BY-NC 4.0','external_archive_md5':STITCHING_MD5,
         'selected_classes':{'00':'normal','02':'broken stitch'},'excluded_class_codes':['01','03','04','05','06','07','08','09','10'],
         'class_counts':{'normal':2436,'broken':1201},
         'exact_cross_archive_overlap_check':cross,
         'external_model_selection_or_training':False,'decision_threshold':0.5,
         'caution':'This is one-shot external evaluation; internal source-photo lineage remains unverified. No real-world deployment claim.'}
    cks={'mobilenet':torch.load(args.mobilenet_weights,map_location='cpu',weights_only=True),
         'smallcnn':torch.load(args.cnn_weights,map_location='cpu',weights_only=True)}
    with zipfile.ZipFile(args.external_zip) as z:
        for key in ['mobilenet','smallcnn']:
            y,p=predict_zip(z,rows,cks[key],key)
            rep[key+'_external_metrics']=compute_metrics(y,p)
            print(key, json.dumps(rep[key+'_external_metrics'],indent=2),flush=True)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(rep,indent=2))
    print('Written '+args.out,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('external-zip','internal-zip','mobilenet-weights','mobilenet-report','cnn-weights','cnn-provenance','portable-manifest','out'):
        ap.add_argument('--'+key,required=True)
    evaluate(ap.parse_args())