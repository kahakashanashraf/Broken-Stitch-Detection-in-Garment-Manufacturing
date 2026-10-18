"""Dataset inventory, safe source-grouped splitting, and per-image preprocessing.

The original thesis's 181 base images and augmented derivatives are NOT assumed to
be independent. Preparing an inventory does not authorize training until every
row has a manually audited nonempty source_id.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

LABEL_TO_INT = {'actual': 0, 'faulty': 1}
LABEL_ALIASES = {'normal': 'actual', 'correct': 'actual', 'good':'actual', 'actual':'actual',
                 'wrong':'faulty', 'broken':'faulty', 'defect':'faulty', 'defective':'faulty', 'faulty':'faulty'}
IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}


def inspect_to_csv(root: str, output: str) -> pd.DataFrame:
    """Write image inventory; preserve source_id blanks instead of guessing parentage."""
    base=Path(root).resolve()
    if not base.is_dir(): raise ValueError(f'Dataset directory does not exist: {base}')
    files=sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files: raise ValueError('No images found after extracting dataset.zip')
    rows=[]
    for p in files:
        rel=p.relative_to(base).as_posix()
        components=[s.lower() for s in p.relative_to(base).parts[:-1]]
        categories={LABEL_ALIASES[s] for s in components if s in LABEL_ALIASES}
        label=categories.pop() if len(categories)==1 else ''
        width=height=0; error=''
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                width,height=im.size
        except (UnidentifiedImageError, OSError, ValueError) as e: error=type(e).__name__
        rows.append({'path':str(p),'relative_path':rel,'label':label,'source_id':'',
                     'is_synthetic':'unknown','split':'','width':width,'height':height,
                     'bytes':p.stat().st_size,'decode_error':error})
    df=pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output,index=False)
    return df


def _canonical_path(path: str) -> Path:
    p=Path(path).expanduser().resolve()
    if not p.is_file(): raise ValueError(f'File not found: {p}')
    return p


def load_manifest(path: str, require_splits=False, verify_images=False) -> pd.DataFrame:
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    required=['path','label','source_id']+(['split'] if require_splits else [])
    absent=[c for c in required if c not in df]
    if absent: raise ValueError(f'Missing manifest column(s): {absent}')
    if df.empty: raise ValueError('Empty manifest')
    for col in ('path','label','source_id'):
        if (df[col].str.strip()=='').any():
            ix=df.index[df[col].str.strip()==''].tolist()[:5]
            raise ValueError(f'{col} has blank values at rows {ix}; manual lineage verification required')
    if not df.label.isin(LABEL_TO_INT).all():
        raise ValueError('Labels must be exactly actual or faulty')
    if df.path.duplicated().any():raise ValueError('Duplicate file path entries in manifest')
    if 'is_synthetic' in df and not df.is_synthetic.isin(['yes','no','unknown']).all():
        raise ValueError('is_synthetic must be yes, no or unknown')
    if require_splits:
        if not df.split.isin(['train','val','test']).all():raise ValueError('split must be train/val/test')
        _validate_partition(df)
    if verify_images:
        for p in df.path:
            with Image.open(_canonical_path(p)) as im: im.verify()
    return df


def _validate_partition(df):
    names={'train','val','test'}
    if set(df.split)!=names: raise ValueError(f'All partitions required: {names}')
    # One source can have both labels if an edited defect was derived from a normal image.
    if (df.groupby('source_id').split.nunique()>1).any():
        bad=df.groupby('source_id').split.nunique().loc[lambda s:s>1].index.tolist()[:10]
        raise ValueError(f'SOURCE LEAKAGE: source_id across multiple partitions: {bad}')
    for part in ('train','val','test'):
        classes=set(df.loc[df.split==part,'label'])
        if classes!={'actual','faulty'}:
            raise ValueError(f'{part} contains {sorted(classes)}; both labels required')


def create_group_split(source_manifest: str, output: str, seed=42, test_fraction=.15, val_fraction=.15, attempts=200):
    if not (0<test_fraction<1 and 0<val_fraction<1 and test_fraction+val_fraction<1):
        raise ValueError('Invalid test/val fractions')
    df=load_manifest(source_manifest)
    if df.source_id.nunique()<8:raise ValueError('At least eight independent source groups needed for 3-way split')
    # Iterate group-exclusive candidates; balance proportions and class support.
    y=df.label.map(LABEL_TO_INT).to_numpy()
    groups=df.source_id.to_numpy()
    best=None
    overall=y.mean()
    for attempt in range(attempts):
        splitter=GroupShuffleSplit(n_splits=1,test_size=test_fraction,random_state=seed+attempt)
        trainval_idx,test_idx=next(splitter.split(df,y,groups))
        g2=groups[trainval_idx]; y2=y[trainval_idx]
        if len(set(g2))<3: continue
        splitter2=GroupShuffleSplit(n_splits=1,test_size=val_fraction/(1-test_fraction),random_state=seed+10000+attempt)
        tpos,vpos=next(splitter2.split(df.iloc[trainval_idx],y2,g2))
        indices={'train':trainval_idx[tpos],'val':trainval_idx[vpos],'test':test_idx}
        if any(len(set(y[idx]))<2 for idx in indices.values()):continue
        score=sum(abs(len(idx)/len(df)-({'train':1-test_fraction-val_fraction,'val':val_fraction,'test':test_fraction}[name])) for name,idx in indices.items())
        score+=.5*sum(abs(y[idx].mean()-overall) for idx in indices.values())
        if best is None or score<best[0]:best=(score,indices)
    if best is None:
        raise ValueError('No 3-way split has BOTH labels in EACH split; collect more independent source groups, or use a grouped CV protocol')
    df=df.copy();df['split']=''
    for part,idx in best[1].items():df.loc[idx,'split']=part
    _validate_partition(df)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False)
    return df


def sha256_file(path,chunk=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while (b:=f.read(chunk)):h.update(b)
    return h.hexdigest()


def audit_manifest(manifest: str, check_images=True, hash_all=True):
    """Return audit summary and block train/test exact-duplicate leakage."""
    df=load_manifest(manifest,require_splits=True,verify_images=check_images)
    summary={'total':len(df),'source_groups':int(df.source_id.nunique()),
             'by_split_label':pd.crosstab(df.split,df.label).to_dict(),
             'synthetic_flags':df.is_synthetic.value_counts().to_dict() if 'is_synthetic' in df else {},
             'limitations':['source_id lineage is user-supplied, not independently validated',
                            'exact file hashes do not detect cropped/rotated near duplicates']}
    if hash_all:
        hashes=[sha256_file(p) for p in df.path]
        tmp=df.copy();tmp['sha']=hashes
        dup=tmp.groupby('sha').agg(splits=('split','nunique'),groups=('source_id','nunique'))
        cross=dup[dup.splits>1]
        if len(cross):
            raise ValueError(f'EXACT-IMAGE LEAKAGE: {len(cross)} hashes appear in multiple partitions')
        summary['exact_duplicate_hash_groups']=int((dup.groups>1).sum())
        summary['exact_cross_source_duplicate_groups']=int((dup.groups>1).sum())
    return summary


class StitchDataset(Dataset):
    def __init__(self, df, split, image_size=100, filter_kind='none', augment=False, vgg=False):
        if split not in ('train','val','test'):raise ValueError('Unexpected partition')
        self.rows=df.loc[df.split==split].reset_index(drop=True)
        self.size=image_size;self.filter=filter_kind
        if augment and split!='train':raise ValueError('Augmentation forbidden outside train')
        ops=[]
        if augment:
            ops += [T.RandomAffine(degrees=8,translate=(.05,.05),scale=(.95,1.05),shear=5),T.RandomHorizontalFlip(p=.5)]
        ops += [T.ToTensor()]
        if vgg:
            ops += [T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))]
        self.transform=T.Compose(ops)
    def __len__(self):return len(self.rows)
    def __getitem__(self,idx):
        r=self.rows.iloc[idx]
        with Image.open(r.path) as image:
            image=ImageOps.exif_transpose(image).convert('RGB').resize((self.size,self.size),resample=Image.Resampling.BILINEAR)
        if self.filter!='none':
            rgb=np.array(image)
            if self.filter=='gaussian':rgb=cv2.GaussianBlur(rgb,(3,3),sigmaX=0)
            elif self.filter=='bilateral':rgb=cv2.bilateralFilter(rgb,5,35,35)
            elif self.filter=='median':rgb=cv2.medianBlur(rgb,3)
            else:raise ValueError(f'Unknown filter: {self.filter}')
            image=Image.fromarray(rgb)
        return self.transform(image),int(LABEL_TO_INT[r.label])