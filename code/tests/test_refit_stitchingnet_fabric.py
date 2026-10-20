import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refit_stitchingnet_fabric import select_arm


def sample(a,b):
    return {'status':'DESCRIPTIVE_SUPERVISED_STITCHINGNET_ADAPTATION_HELDOUT_FABRIC_NOT_BLIND',
            'n_images':3637,'n_fabrics':11,
            'models':{'imagenet':{'macro_f1':a},'synthetic':{'macro_f1':b}}}


def test_refit_follows_locked_cv_winner():
    assert select_arm(sample(.7,.8))=='synthetic'
    assert select_arm(sample(.8,.7))=='imagenet'
    assert select_arm(sample(.8,.8))=='imagenet'


def test_refit_rejects_fake_or_incomplete_results():
    d=sample(.9,.7);d['n_fabrics']=9
    with pytest.raises(ValueError,match='sample/fabric'):select_arm(d)
    d=sample(.9,.7);del d['models']['synthetic']
    with pytest.raises(ValueError,match='missing model'):select_arm(d)
    d=sample(.9,float('nan'))
    with pytest.raises(ValueError,match='invalid'):select_arm(d)


def test_domain_checkpoint_explicitly_cannot_claim_independent_test():
    src=(Path(__file__).resolve().parents[1]/'scripts'/'refit_stitchingnet_fabric.py').read_text()
    assert "'no_untouched_test_accuracy_claim':True" in src
    assert "'no_zero_shot_claim':True" in src
    assert "'test_metrics_for_final_refit':None" in src