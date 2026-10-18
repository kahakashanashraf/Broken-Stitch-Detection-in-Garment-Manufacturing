import numpy as np
import pandas as pd
from PIL import Image
import pytest
from broken_stitch.data import (inspect_to_csv,load_manifest,create_group_split,audit_manifest,StitchDataset)
from broken_stitch.models import create_model
from broken_stitch.metrics import compute_metrics


def build_dataset(tmp_path,n=28):
    root=tmp_path/'images'
    for i in range(n):
        # A paired normal/edit synthetic defect sharing the SAME source group.
        for name in ('actual','faulty'):
            p=root/name/f'parent{i:03}_{name}.png';p.parent.mkdir(parents=True,exist_ok=True)
            rng=np.random.default_rng(i+ (0 if name=='actual' else 10_000))
            arr=rng.integers(0,255,(32,32,3),dtype=np.uint8)
            if name=='faulty':arr[:,13:18]=0
            Image.fromarray(arr).save(p)
    m=tmp_path/'inventory.csv'
    df=inspect_to_csv(root,m)
    df['source_id']=df.relative_path.str.extract(r'parent(\d+)')[0]
    df['is_synthetic']=np.where(df.label=='faulty','yes','no')
    df.to_csv(m,index=False)
    return m,df


def test_inventory_never_invents_parentage(tmp_path):
    m,df=build_dataset(tmp_path)
    first=pd.read_csv(m,keep_default_na=False)
    assert (first.source_id!='').all() # explicitly assigned by test setup
    root=tmp_path/'images';other=tmp_path/'raw.csv'
    raw=inspect_to_csv(root,other)
    assert (raw.source_id=='').all()
    with pytest.raises(ValueError,match='source_id has blank'):
        create_group_split(other,tmp_path/'no.csv')


def test_source_disjoint_partitions_and_aug_only_train(tmp_path):
    m,_=build_dataset(tmp_path)
    out=tmp_path/'split.csv'
    df=create_group_split(m,out,seed=100)
    assert not (df.groupby('source_id').split.nunique()>1).any()
    assert set(df.groupby('split').label.nunique())=={2}
    assert len(StitchDataset(df,'val',image_size=32))>0
    with pytest.raises(ValueError,match='outside train'):
        StitchDataset(df,'test',augment=True)
    check=audit_manifest(out)
    assert check['total']==56


def test_source_leakage_rejected(tmp_path):
    m,_=build_dataset(tmp_path)
    out=tmp_path/'split.csv';df=create_group_split(m,out)
    source=df.iloc[0].source_id
    idx=df.index[(df.source_id==source)].tolist()
    assert len(idx)>=2
    df.loc[idx[0],'split']='val' if df.loc[idx[0],'split']!='val' else 'test'
    df.to_csv(out,index=False)
    with pytest.raises(ValueError,match='SOURCE LEAKAGE'):load_manifest(out,require_splits=True)


def test_exact_duplicate_leakage_rejected(tmp_path):
    m,_=build_dataset(tmp_path)
    out=tmp_path/'split.csv';df=create_group_split(m,out)
    x=df[df.split=='train'].iloc[0];y=df[df.split=='test'].iloc[0]
    from shutil import copyfile
    copyfile(x.path,y.path)
    with pytest.raises(ValueError,match='EXACT-IMAGE LEAKAGE'):audit_manifest(out)


def test_metrics_and_models():
    m=compute_metrics([0,0,1,1],[.1,.2,.8,.9])
    assert m['macro_f1']==1 and m['faulty']['recall']==1
    import torch
    model,_=create_model('cnn')
    with torch.no_grad():out=model(torch.randn(2,3,32,32))
    assert tuple(out.shape)==(2,2)
    with pytest.raises(ValueError,match='VGG needs'):
        create_model('vgg_linear')


def test_vgg_random_smoke_flag_and_features():
    import torch
    m,origin=create_model('vgg_linear',allow_random_vgg=True)
    assert origin.startswith('RANDOM_UNTRAINED')
    m.eval()
    with torch.no_grad():out=m(torch.randn(1,3,32,32))
    assert tuple(out.shape)==(1,2)