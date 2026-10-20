from pathlib import Path
import tempfile
import sys
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict_enhanced_mobilenet_224 import load_enhanced_checkpoint
from enhanced_mobilenet_224_cv import RUN_ID,SIZE,EPOCHS
from grouped_finetune_cv import EXPECTED_MANIFEST_SHA,FineTuneBinary


def fixture_ck(model):
    return {'architecture':'mobilenetv3','image_size':SIZE,'epochs':EPOCHS,
            'features':model.features.state_dict(),'head':model.head.state_dict(),
            'label_map':{'actual':0,'faulty':1},'fixed_threshold':.5,
            'dev_manifest_sha256':EXPECTED_MANIFEST_SHA,'dev_parent_count':26,
            'not_untouched_external':True,'training_recipe':RUN_ID}


def test_checkpoint_roundtrip_224_and_provenance():
    model=FineTuneBinary('mobilenetv3',pretrained=False);model.eval()
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'model.pt';torch.save(fixture_ck(model),path)
        loaded=load_enhanced_checkpoint(path)
        x=torch.randn(1,3,224,224)
        with torch.inference_mode():
            assert torch.allclose(model(x),loaded(x),atol=1e-6)


def test_wrong_preprocess_is_rejected():
    model=FineTuneBinary('mobilenetv3',pretrained=False)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'bad.pt'
        bad=fixture_ck(model);bad['image_size']=128
        torch.save(bad,path)
        with pytest.raises(ValueError,match='image_size'):load_enhanced_checkpoint(path)