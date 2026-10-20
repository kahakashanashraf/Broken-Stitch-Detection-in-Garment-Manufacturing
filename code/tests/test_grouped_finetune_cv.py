from pathlib import Path
import sys
import pytest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from grouped_finetune_cv import FineTuneBinary,balanced_parent_cap,photo_class_weights,evaluate

def test_group_balanced_cap():
    rows=[]
    for parent in ['a','b']:
        for label in ['actual','faulty']:
            for i in range(10):rows.append({'inferred_photographed_source':'photo_candidate:'+parent,'label':label,'clean_path':f'{parent}_{label}_{i}'})
    sel=balanced_parent_cap(rows,np.arange(len(rows)),123,cap=3)
    assert len(sel)==12
    weights=photo_class_weights(rows,sel)
    assert sum(weights.values())==pytest.approx(12)
    assert sum(weights[j] for j in sel if rows[j]['label']=='actual')==pytest.approx(6)

def test_actual_convolutional_tail_trainable():
    for arch in ('mobilenetv3','densenet121','vgg16'):
        m=FineTuneBinary(arch,pretrained=False)
        trainable=[name for name,p in m.features.named_parameters() if p.requires_grad]
        assert len(trainable)>0 and m.tail_parameter_n>0
        assert len(trainable)<len(list(m.features.named_parameters()))
        x=torch.randn(1,3,128,128)
        m.eval()
        with torch.no_grad():y=m(x)
        assert y.shape==(1,) and torch.isfinite(y).all()

def test_metrics_are_classwise():
    x=evaluate([0,0,1,1],[0.1,0.3,0.6,0.9])
    assert x['macro_f1']==1 and x['faulty_recall']==1

def test_no_external_access_codepath():
    t=(Path(__file__).resolve().parents[1] / 'scripts' / 'grouped_finetune_cv.py').read_text()
    assert "'train':Path(args.train_zip),'val':Path(args.val_zip)" in t
    assert "choices=['mobilenetv3','densenet121','vgg16']" in t