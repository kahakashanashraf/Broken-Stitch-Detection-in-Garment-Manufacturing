"""Guard real attribution and prohibit inadvertent retraining or arbitrary CAM samples."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import xai_thesis_vs_fabric_holdout as xai
from grouped_finetune_cv import FineTuneBinary


def test_actual_gradient_and_fixed_occlusion_dimensions():
    torch.manual_seed(20261021)
    net=FineTuneBinary('mobilenetv3',pretrained=False)
    net.eval()
    x=torch.randn(1,3,224,224)
    from fabric_group_adaptation import cam_for_broken_logit
    heat=cam_for_broken_logit(net,x)
    assert np.isfinite(heat).all() and heat.shape==(224,224)
    result=xai.occlusion_probes(net,x,heat)
    assert result['fraction_masked']==pytest.approx(0.2,abs=1e-4)
    assert all(np.isfinite(result[k]) for k in ('baseline_broken_logit','top20_percent_cam_occlusion_logit_drop','bottom20_percent_cam_occlusion_logit_drop'))


def test_paired_heatmap_has_no_source_photo(tmp_path):
    left=tmp_path/'left.png';Image.new('L',(224,224),color=180).save(left)
    right=np.linspace(0,1,224*224,dtype=np.float32).reshape(224,224)
    result=xai.combined_strip(left,right,'example',tmp_path/'panel.png')
    assert result['filename']=='panel.png'
    with Image.open(tmp_path/'panel.png') as im:
        assert im.size==(448,260)


def test_fail_closed_wrong_fold_group_identity(tmp_path):
    rdir=tmp_path/'reports';hdir=tmp_path/'heat'
    for k in range(5):
        (rdir/f'fold{k}').mkdir(parents=True)
        (hdir/f'fold{k}').mkdir(parents=True)
        xais=[]
        for kind,p in [('true_positive',.9),('false_negative',.1)]:
            fname=f'synthetic_fabricfold{k}_{kind}_gradcam.png'
            Image.new('L',(224,224),color=150).save(hdir/f'fold{k}'/fname)
            xais.append({'image_identifier':f'/fake/{k}_{kind}.png','fabric':'A','category':kind,
                         'probability_broken':p,'xai_png':fname})
        (rdir/f'fold{k}'/'fabric_heldout_report.json').write_text(json.dumps({
          'arm':'synthetic','fold':k,'threshold':.5,'val_fabric_overlap':0,
          'training_fabrics':['B'],'held_out_fabrics':['A'],'dataset_md5':xai.EXPECTED_STITCH_MD5,
          'xai_heatmaps':xais}))
    keys={f'/fake/{k}_{kind}.png' for k in range(5) for kind in xai.KIND}
    assert len(xai.selected_prior_cases(rdir,hdir,keys))==10
    # Flip source provenance; expect a strong fail-closed error.
    bad=json.loads((rdir/'fold3'/'fabric_heldout_report.json').read_text())
    bad['training_fabrics']=['A']
    (rdir/'fold3'/'fabric_heldout_report.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='overlapping fabric'):
        xai.selected_prior_cases(rdir,hdir,keys)


def test_readonly_requirements_in_code():
    code=Path(xai.__file__).read_text()
    assert 'EXPECTED_SOURCE_SHA256' in code
    assert 'load_enhanced_checkpoint(args.weights)' in code
    assert 'selected_prior_cases' in code
    assert 'cam_for_broken_logit(model,x)' in code
    assert 'new_model_training_or_tuning' in code
    assert 'independent_blind_test' in code
    assert 'torch.optim' not in code and 'optimizer.step' not in code