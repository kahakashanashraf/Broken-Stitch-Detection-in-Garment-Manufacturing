"""Verify external evaluation checkpoint is frozen and no threshold tuning occurs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_enhanced_stitchingnet import EXPECTED_WEIGHTS_SHA256


def test_external_checkpoint_digest_frozen():
    assert EXPECTED_WEIGHTS_SHA256=='0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81'


def test_external_score_is_readonly_fixed_threshold_and_nonblind():
    s=(ROOT/'scripts'/'evaluate_enhanced_stitchingnet.py').read_text()
    assert 'previous.VAL_TF=PRE' in s
    assert 'model=load_enhanced_checkpoint(args.weights)' in s
    assert "'external_fitting_or_threshold_tuning':False" in s
    assert "'new_independent_test':False" in s
    assert 'fit(' not in s and 'optimizer' not in s
    assert "'fixed_threshold':0.5" in s