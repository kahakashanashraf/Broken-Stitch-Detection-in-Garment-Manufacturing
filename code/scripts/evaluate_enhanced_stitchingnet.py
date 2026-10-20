"""NO-TUNING external diagnostic for locked enhanced MobileNetV3 224px/12epoch.

StitchingNet was inspected in earlier model experiments; this external score is
*non-blind descriptive transfer*, NEVER independent untouched validation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_stitchingnet_finetuned as previous
from predict_enhanced_mobilenet_224 import load_enhanced_checkpoint, PRE

EXPECTED_WEIGHTS_SHA256='0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81'
TAG='enhanced-mobilenet-224e12-v1-refit'


def run(args):
    torch.set_num_threads(4)
    if previous.digest(args.external_zip,'md5')!=previous.EXTERNAL_MD5:
        raise ValueError('STOP: original StitchingNet archive MD5 mismatch')
    if previous.digest(args.weights)!=EXPECTED_WEIGHTS_SHA256:
        raise ValueError('STOP: checkpoint hash differs from archived 224px refit')
    model=load_enhanced_checkpoint(args.weights)
    rows=previous.list_binary_members(args.external_zip)
    # Reuse the same official binary subset and metrics implementation as six prior
    # models, changing ONLY the input model and expected image-size preprocessing.
    previous.VAL_TF=PRE
    dl=DataLoader(previous.ExternalArchiveImages(args.external_zip,rows),batch_size=12,
                  shuffle=False,num_workers=0)
    probs=[];labels=[];names=[];fabrics=[]
    with torch.inference_mode():
        for k,(xx,yy,name,fabric) in enumerate(dl):
            probs.extend(torch.sigmoid(model(xx)).cpu().numpy().tolist())
            labels.extend(yy.numpy().tolist());names.extend(name);fabrics.extend(fabric)
            if k%40==0:print('EXTERNAL_224_PROGRESS',len(probs),'/',len(rows),flush=True)
    if len(probs)!=3637 or len(labels)!=len(rows) or list(zip(names,labels,fabrics))!=rows:
        raise ValueError('STOP: external image count/order/labels differ from audited StitchingNet subset')
    overall=previous.summary(labels,probs)
    fabric_metrics={}
    for fab in sorted(set(fabrics)):
        ii=[i for i,x in enumerate(fabrics) if x==fab]
        fabric_metrics[fab]=previous.summary([labels[i] for i in ii],[probs[i] for i in ii])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    report={'status':'POST_HOC_NONBLIND_CROSS_DOMAIN_DESCRIPTIVE_NO_TUNING',
            'model':'MobileNetV3 224px 12epochs fixed class_only sampled development refit',
            'checkpoint_release':TAG,'checkpoint_sha256':EXPECTED_WEIGHTS_SHA256,
            'external_md5':previous.EXTERNAL_MD5,'external_source':'https://zenodo.org/records/20352813',
            'external_data_were_previously_inspected':True,'new_independent_test':False,
            'external_fitting_or_threshold_tuning':False,'fixed_threshold':0.5,
            'input_image_size':224,'evaluation_subset_normal':2436,'evaluation_subset_broken':1201,
            'metrics':overall,'by_fabric':fabric_metrics,
            'caveats':['StitchingNet was already inspected by earlier project experiments; not blind.',
                       'The enhanced recipe was decided after several earlier model evaluations including StitchingNet.',
                       'Source parents in thesis are filename-inferred, not camera-origin authenticated.',
                       'No industrial deployment or causal transfer superiority established.']}
    (out/'external_enhanced_metrics.json').write_text(json.dumps(report,indent=2))
    with (out/'external_enhanced_predictions.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['source_file','fabric','actual_class','p_broken','prediction'])
        for name,fab,lab,p in zip(names,fabrics,labels,probs):
            w.writerow([name,fab,'normal' if lab==0 else 'broken',float(p),'broken' if p>=.5 else 'normal'])
    print('EXTERNAL_ENHANCED_RESULT',json.dumps({'metrics':overall,'checkpoint_sha256':EXPECTED_WEIGHTS_SHA256}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--external-zip',required=True)
    ap.add_argument('--weights',required=True)
    ap.add_argument('--out',required=True)
    run(ap.parse_args())