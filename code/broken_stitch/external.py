"""StitchingNet *binary normal-vs-broken* holdout; never train on the holdout.

Only human-designated label directories are accepted. Other defect types are
excluded rather than silently mapped to the broken-stitch class.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import torch
from PIL import Image
from .data import IMAGE_EXTENSIONS, sha256_file, StitchDataset, load_manifest
from .engine import _manifest_fingerprint
from .models import create_model
from .metrics import compute_metrics
from torch.utils.data import DataLoader


def make_external_manifest(normal_dir, broken_dir, output):
    normal, broken = Path(normal_dir).resolve(), Path(broken_dir).resolve()
    if not normal.is_dir() or not broken.is_dir():
        raise ValueError('Provide both manually verified normal/ and broken-stitch/ directories')
    if normal==broken or normal in broken.parents or broken in normal.parents:
        raise ValueError('Normal and broken directories must not contain one another')
    rows=[]
    for root, label in ((normal,'actual'),(broken,'faulty')):
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS: continue
            try:
                with Image.open(p) as im: im.verify()
            except Exception as exc:
                raise ValueError(f'Invalid image {p}: {type(exc).__name__}') from exc
            # These are image IDs, NOT independently-verified photograph groups.
            rows.append({'path':str(p), 'label':label, 'split':'test',
                         'source_id':'external_image:'+str(p), 'dataset':'StitchingNet',
                         'is_synthetic':'unknown', 'lineage_status':'unverified'})
    df=pd.DataFrame(rows)
    if df.empty or set(df.label)!={'actual','faulty'}:
        raise ValueError('External binary benchmark requires both nonempty classes')
    # Same exact bytes in normal and broken = ambiguous labels, STOP.
    hashes={}
    for row in df.itertuples(index=False):
        sha=sha256_file(row.path)
        if sha in hashes and hashes[sha]!=row.label:
            raise ValueError(f'Image byte-identical across classes: {row.path}')
        hashes[sha]=row.label
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest,index=False)
    return {'n_images':len(df),'class_counts':df.label.value_counts().to_dict(),
            'manifest':str(dest), 'grouping':'UNVERIFIED per-image identity',
            'scope':'normal vs broken ONLY; all other 9 defect labels excluded'}


def evaluate_external(checkpoint, internal_manifest, external_manifest, output, device='auto'):
    """Use saved model and 0.5 decision threshold; no external-set tuning permitted."""
    train_df=load_manifest(internal_manifest,require_splits=True)
    ext_df=load_manifest(external_manifest,require_splits=False)
    if set(ext_df.get('split',pd.Series(dtype=str)))!={'test'}:
        raise ValueError('External samples must be labeled test only')
    if set(ext_df.label)!={'actual','faulty'}:
        raise ValueError('Both classes required')
    if 'dataset' not in ext_df or set(ext_df.dataset)!={'StitchingNet'}:
        raise ValueError('External manifest must explicitly declare StitchingNet origin')
    if set(train_df.path)&set(ext_df.path):
        raise ValueError('External paths overlap internal training/validation/test manifest')
    # Check every internal image (not just training) to avoid hidden contamination.
    original_hashes={sha256_file(p) for p in train_df.path}
    collisions=[p for p in ext_df.path if sha256_file(p) in original_hashes]
    if collisions:raise ValueError(f'EXTERNAL LEAKAGE: {len(collisions)} byte-identical images overlap internal dataset')
    checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=checkpoint_data['config']
    if cfg['split_hash']!=_manifest_fingerprint(train_df):
        raise ValueError('Internal manifest fingerprint changed since model training')
    if cfg.get('weights_source') == 'RANDOM_UNTRAINED' and cfg['model'].startswith('vgg'):
        raise ValueError('Random-weight VGG smoke model is not valid for a real benchmark')
    model,_=create_model(cfg['model'],cfg['activation'],allow_random_vgg=True)
    model.load_state_dict(checkpoint_data['model_state'],strict=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    model.to(device).eval()
    dataset=StitchDataset(ext_df,'test',image_size=cfg['image_size'],filter_kind=cfg['filter'],augment=False,vgg=cfg['model'].startswith('vgg'))
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    y,p=[],[]
    with torch.no_grad():
        for images, labels in loader:
            probs=model(images.to(device)).softmax(dim=1)[:,1].cpu().tolist()
            y.extend(labels.tolist());p.extend(probs)
    report={'dataset':'StitchingNet', 'task':'NORMAL vs BROKEN_STITCH only',
            'evaluation':'zero-shot external evaluation of FROZEN checkpoint and 0.5 threshold',
            'metrics':compute_metrics(y,p), 'n_external':len(y),
            'internal_manifest_sha256':_manifest_fingerprint(train_df),
            'lineage_note':'Independent camera/source-group status is not established by directory names.',
            'license':'CC BY-NC 4.0 (check official dataset terms); do not redistribute without respecting license',
            'publication_ready':False,
            'caveats':['Internal training lineage must be human audited',
                       'External data were not used for model selection',
                       'Exact SHA does not rule out transformed-image duplicates',
                       'No claims about the other StitchingNet defect classes']}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    return report