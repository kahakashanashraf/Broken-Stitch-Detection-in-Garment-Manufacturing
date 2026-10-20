from pathlib import Path
import sys
import tempfile
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from grouped_finetune_cv import FineTuneBinary
from predict_finetuned import load_model

def test_checkpoint_roundtrip_and_probability():
    model=FineTuneBinary('mobilenetv3',pretrained=False)
    model.eval()
    with tempfile.TemporaryDirectory() as t:
        path=Path(t)/'model.pt'
        torch.save({'architecture':'mobilenetv3','image_size':128,'label_map':{'actual':0,'faulty':1},
                    'fixed_threshold':.5,'dev_parent_count':26,'not_untouched_external':True,
                    'features':model.features.state_dict(),'head':model.head.state_dict()},path)
        loaded,arch=load_model(path)
        assert arch=='mobilenetv3'
        x=torch.randn(1,3,128,128)
        with torch.no_grad():
            p=model(x);q=loaded(x)
        assert torch.allclose(p,q)

def test_invalid_provenance_rejected():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'bad.pt';torch.save({'architecture':'vgg16','image_size':128,'label_map':{'actual':0,'faulty':1},'fixed_threshold':.9},p)
        with pytest.raises(ValueError):load_model(p)