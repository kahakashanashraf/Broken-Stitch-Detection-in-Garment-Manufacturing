"""Same-image Grad-CAM comparison: thesis-only zero-shot vs fabric-held-out-adapted CNN.

Ten images are PRESELECTED by the prior supervised fabric-heldout Grad-CAM reports:
TP + FN, one each from five grouped fabric folds, for the synthetic-init arm.
This code cannot train, tune, choose models, or change decision threshold; selection
is retrospective, so interpret output as case study, not a representative sample.

No raw copyrighted StitchingNet source pixels are written to the public output.
Only gradient heatmaps, paired heatmap strips, numeric predictions, and metadata.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_stitchingnet_finetuned as ext
from fabric_group_adaptation import cam_for_broken_logit
from predict_enhanced_mobilenet_224 import load_enhanced_checkpoint

EXPECTED_SOURCE_SHA256 = '0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81'
EXPECTED_STITCH_MD5 = '51b228d9572b925035e9e9af01c96569'
SOURCE_TAG_PREFIX = 'stitchingnet-fabric-heldout-synthetic-f'
NEW_TAG = 'thesis-only-vs-fabric-heldout-gradcam-v1'
SIZE = 224
THRESHOLD = 0.5
KIND = ('true_positive', 'false_negative')
PRE = T.Compose([
    T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])


def sha(path, algorithm):
    h=hashlib.new(algorithm)
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(4<<20),b''):h.update(c)
    return h.hexdigest()


def selected_prior_cases(report_root, heat_root, authorized_archive_keys):
    """Validate references to frozen, released fabric-disjoint fold CAM cases."""
    cases=[]
    seen=set()
    for fold in range(5):
        folder=Path(report_root)/f'fold{fold}'
        report_path=folder/'fabric_heldout_report.json'
        report=json.loads(report_path.read_text())
        if (report.get('arm')!='synthetic' or report.get('fold')!=fold or
            report.get('threshold')!=THRESHOLD or report.get('val_fabric_overlap')!=0 or
            report.get('dataset_md5')!=EXPECTED_STITCH_MD5):
            raise ValueError(f'STOP: invalid fabric-grouped source report fold={fold}')
        held=set(report.get('held_out_fabrics',[]))
        train=set(report.get('training_fabrics',[]))
        if not held or train&held:
            raise ValueError('STOP: original supervised fold has overlapping fabric groups')
        selection=report.get('xai_heatmaps',[])
        if {r['category'] for r in selection}!=set(KIND) or len(selection)!=2:
            raise ValueError('STOP: expected source-model true-positive and false-negative CAMs')
        for row in selection:
            path=row['image_identifier']
            if path in seen or path not in authorized_archive_keys or row['fabric'] not in held:
                raise ValueError('STOP: reused/unknown image path or wrong held-out fabric')
            seen.add(path)
            if row['category'] not in KIND:
                raise ValueError('STOP: unexpected source-model label')
            p=float(row['probability_broken'])
            if not np.isfinite(p) or not 0<=p<=1:
                raise ValueError('STOP: invalid reported supervised model probability')
            if row['category']=='true_positive' and p<THRESHOLD:
                raise ValueError('STOP: purported true positive below fixed threshold')
            if row['category']=='false_negative' and p>=THRESHOLD:
                raise ValueError('STOP: purported false negative above threshold')
            expected=f'synthetic_fabricfold{fold}_{row["category"]}_gradcam.png'
            if row['xai_png']!=expected:
                raise ValueError('STOP: source-model heatmap filename inconsistent')
            heat_file=Path(heat_root)/f'fold{fold}'/expected
            with Image.open(heat_file) as im:
                if im.size!=(SIZE,SIZE) or im.mode!='L':
                    raise ValueError('STOP: expected existing 224x224 grayscale Grad-CAM')
            cases.append({'fold':fold,'source_path':path,'fabric':row['fabric'],
                          'supervised_category':row['category'],
                          'supervised_model_p_broken':p,
                          'supervised_xai_png':str(heat_file),
                          'supervised_release':f'https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/{SOURCE_TAG_PREFIX}{fold}-v1'})
    if len(cases)!=10:
        raise ValueError('STOP: source case count not ten')
    return cases


def occlusion_probes(model,x,cam,fraction=0.20):
    """Equal-area top CAM vs bottom CAM occlusion, with normalized-input zeros.

    Masking normalized input to 0 means replacing pixels by ImageNet mean RGB.
    This is just a perturbation check, NOT validated faithfulness or localization.
    """
    if x.shape!=(1,3,SIZE,SIZE) or np.shape(cam)!=(SIZE,SIZE):
        raise ValueError('Unexpected image/CAM dimensions')
    flat=np.asarray(cam,dtype=np.float32).reshape(-1)
    if not np.isfinite(flat).all():raise ValueError('Nonfinite heatmap')
    k=int(round(len(flat)*fraction))
    ranking=np.argsort(flat,kind='stable')
    high=ranking[-k:];low=ranking[:k]
    def perturbed(ids):
        xm=x.detach().clone().reshape(1,3,-1)
        xm[:,:,ids]=0.0
        with torch.no_grad():return float(model(xm.view_as(x)).item())
    with torch.no_grad():baseline=float(model(x).item())
    top_logit=perturbed(high)
    low_logit=perturbed(low)
    return {'fraction_masked':k/len(flat),
            'baseline_broken_logit':baseline,
            'top20_percent_cam_occlusion_logit_drop':baseline-top_logit,
            'bottom20_percent_cam_occlusion_logit_drop':baseline-low_logit,
            'top_minus_bottom_logit_drop':low_logit-top_logit,
            'replaces_masked_input_with':'ImageNet mean RGB via zero in normalized space',
            'limitation':'Occlusion introduces out-of-distribution patterns; not validated causal attribution'}


def combined_strip(source_heat, new_heat, case_id, out):
    """Only grayscale gradient maps, never original licensed StitchingNet pixels."""
    with Image.open(source_heat) as im:
        sup=im.convert('L')
    thesis=Image.fromarray(np.uint8(np.round(255*np.clip(new_heat,0,1))),mode='L')
    if sup.size!=thesis.size or sup.size!=(SIZE,SIZE):raise ValueError('Heatmap dimensions mismatch')
    panel=Image.new('RGB',(SIZE*2, SIZE+36),'white')
    panel.paste(thesis.convert('RGB'),(0,36))
    panel.paste(sup.convert('RGB'),(SIZE,36))
    d=ImageDraw.Draw(panel)
    d.text((7,9),'Thesis only / zero-shot',fill='black')
    d.text((SIZE+7,9),'Fabric-heldout trained',fill='black')
    panel.save(out)
    return {'filename':out.name,'case':case_id,'left':'Thesis-only, frozen, no StitchingNet supervised training',
            'right':'Synthetic initialized, trained on StitchingNet OTHER fabrics, evaluated on held-out fabric',
            'note':'Each CAM normalized per image. Brightness cannot be compared as absolute model evidence.'}


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if sha(args.weights,'sha256')!=EXPECTED_SOURCE_SHA256:
        raise ValueError('STOP: thesis-only refit checkpoint SHA-256 changed')
    if sha(args.stitchingnet_zip,'md5')!=EXPECTED_STITCH_MD5:
        raise ValueError('STOP: official StitchingNet original ZIP MD5 changed')
    model=load_enhanced_checkpoint(args.weights)
    model.eval();torch.set_num_threads(4)
    catalog=ext.list_binary_members(args.stitchingnet_zip)
    keys={r[0] for r in catalog}
    by_key={r[0]:(r[1],r[2]) for r in catalog}
    cases=selected_prior_cases(args.supervised_reports,args.supervised_heats,keys)
    records=[]
    with zipfile.ZipFile(args.stitchingnet_zip,'r') as z:
        for case in cases:
            true_label,fabric=by_key[case['source_path']]
            if true_label!=1 or fabric!=case['fabric']:
                raise ValueError('STOP: reference XAI selections are not truly broken labels')
            raw=z.read(case['source_path'])
            with Image.open(__import__('io').BytesIO(raw)) as im:
                x=PRE(ImageOps.exif_transpose(im).convert('RGB')).unsqueeze(0)
            with torch.no_grad():p=float(torch.sigmoid(model(x)).item())
            heat=cam_for_broken_logit(model,x)
            if heat.shape!=(SIZE,SIZE) or not np.isfinite(heat).all():
                raise ValueError('STOP: actual thesis Grad-CAM failed')
            if heat.min()<0 or heat.max()>1.000001:
                raise ValueError('STOP: Grad-CAM normalization invalid')
            probe=occlusion_probes(model,x,heat)
            fold=case['fold'];category=case['supervised_category']
            stem=f'fold{fold}_{category}'
            fname=stem+'_thesis_gradcam.png'
            Image.fromarray(np.uint8(np.round(heat*255)),mode='L').save(out/fname)
            panel=combined_strip(case['supervised_xai_png'],heat,stem,out/(stem+'_paired_gradcam.png'))
            item={
                'source_image_path':case['source_path'],'fabric':fabric,
                'true_class':'broken','case_selected_from':'fabric-heldout supervised synthetic initialization fold reports',
                'supervised_fold':fold,'supervised_category':category,
                'supervised_train_fabrics_disjoint_from_this_fabric':True,
                'thesis_only_probability_broken':p,
                'thesis_only_prediction':'broken' if p>=THRESHOLD else 'normal',
                'supervised_fold_model_probability_broken':case['supervised_model_p_broken'],
                'thesis_only_gradcam_heatmap':fname,
                'supervised_existing_reference_release':case['supervised_release'],
                'supervised_existing_reference_heatmap':Path(case['supervised_xai_png']).name,
                'paired_gradient_panel':panel['filename'],
                'thesis_only_occlusion':probe}
            records.append(item)
            print('XAI_PAIRED_CASE',json.dumps({'fold':fold,'case':category,'fabric':fabric,
                                                'thesis_p':p,'supervised_fold_p':case['supervised_model_p_broken']}),flush=True)
    summary={
        'status':'GRADCAM_SAME_IMAGE_DESCRIPTIVE_SYNTHETIC_TO_REAL_WITH_OCClUSION_CHECK',
        'protocol':'Frozen thesis-only enhanced MobileNetV3 versus PREVIOUSLY FIT fabric-heldout supervised Grad-CAM on ten matching real photos',
        'source_releases':'stitchingnet-fabric-heldout-synthetic-f[0-4]-v1',
        'thesis_checkpoint_sha256':EXPECTED_SOURCE_SHA256,
        'official_stitchingnet_md5':EXPECTED_STITCH_MD5,
        'threshold':THRESHOLD,'selected_cases':len(records),
        'selection_process':'1 supervised true-positive and 1 supervised false-negative per fold selected using previously published reports; NOT a representative random sample',
        'new_model_training_or_tuning':False,'independent_blind_test':False,
        'thesis_only_model_exposed_to_stitchingnet_labels_in_this_run':False,
        'contrast':'Zero-shot thesis model never trained on StitchingNet versus OTHER-fabric fine-tuned model in each prior fold',
        'visual_data_license':'No original StitchingNet image pixels republished, just independently computed gradient heatmaps',
        'method':'Grad-CAM broken-class logit, original actual convolutional feature activations+gradients; RGB nearest-size bilinear 224x224; positive ReLU weighted average; minmax separately per image',
        'interpretation':'Qualitative XAI and occlusion sanity checks only. Cannot infer causal localization, pixelwise defect identification, or valid target-domain performance from these ten examples.',
        'cases':records}
    (out/'xai_same_image_report.json').write_text(json.dumps(summary,indent=2))
    with (out/'xai_cases.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['fold','category','fabric','source_image','thesis_p_broken','supervised_fold_p_broken','thesis_pred','top_cam_occlusion_delta','bottom_cam_occlusion_delta','paired_png'])
        for r in records:
            o=r['thesis_only_occlusion']
            w.writerow([r['supervised_fold'],r['supervised_category'],r['fabric'],r['source_image_path'],
                        r['thesis_only_probability_broken'],r['supervised_fold_model_probability_broken'],r['thesis_only_prediction'],
                        o['top20_percent_cam_occlusion_logit_drop'],o['bottom20_percent_cam_occlusion_logit_drop'],r['paired_gradient_panel']])
    print('XAI_SUMMARY',json.dumps({'cases':len(records),'report':'xai_same_image_report.json','no_retraining':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stitchingnet-zip',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--supervised-reports',required=True)
    p.add_argument('--supervised-heats',required=True)
    p.add_argument('--out',required=True)
    run(p.parse_args())