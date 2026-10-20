"""Strict grouped experimental integrity and real Grad-CAM gradient smoke-tests."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fabric_group_adaptation as f


def synthetic_rows():
    rows=[]
    for fab in f.FABRICS:
        for lab in (0,1):
            for i in range(9):
                rows.append((f'fake/{fab}_{lab}_{i}.jpg',lab,fab))
    return rows


def test_fivefold_fabric_exclusive_without_overlaps():
    rows=synthetic_rows();folds,assignment=f.grouped_folds(rows)
    assert len(folds)==5
    seen=[]
    for tr,va,trfab,vafab in folds:
        assert not(set(trfab)&set(vafab))
        assert len(va)>0 and len(tr)>0
        assert {rows[i][1] for i in va}=={0,1}
        seen+=va.tolist()
    assert len(seen)==len(rows) and set(seen)==set(range(len(rows)))
    for fab in f.FABRICS:
        assert len({assignment[i] for i,r in enumerate(rows) if r[2]==fab})==1


def test_scores_reject_wrong_data():
    with pytest.raises(ValueError):f.scores([0,0],[.1,.3])
    with pytest.raises(ValueError):f.scores([0,1],[float('nan'),.9])
    assert f.scores([0,0,1,1],[.01,.02,.98,.99])['macro_f1']==1


def test_gradcam_nonzero_valid_and_hook_removed():
    torch.manual_seed(113)
    model=f.FineTuneBinary('mobilenetv3',pretrained=False)
    x=torch.randn(1,3,224,224)
    initial=len(model.features._forward_hooks)
    heat=f.cam_for_broken_logit(model,x)
    assert heat.shape==(224,224)
    assert np.isfinite(heat).all() and 0<=heat.min()<=heat.max()<=1
    assert len(model.features._forward_hooks)==initial


def test_license_data_and_no_test_leakage_guard():
    txt=Path(f.__file__).read_text()
    assert f.EXT_MD5=='51b228d9572b925035e9e9af01c96569'
    assert f.SIZE==224 and f.EPOCHS==6 and f.BATCH==16
    assert f.ARMS==('imagenet','synthetic')
    assert 'fabric group crosses OOF folds' in txt
    assert 'not a new independent blind test' in txt.lower() or 'not a new independent' in txt.lower()
    assert 'source_file' in txt and 'xai_heatmaps' in txt