"""Protocol safety tests — synthetic only, no external data access."""
import importlib.util
from pathlib import Path

script=Path(__file__).resolve().parents[1]/'scripts'/'evaluate_stitchingnet_finetuned.py'

def test_exact_checkpoint_sha_list_and_three_arms():
    t=script.read_text()
    for s in ('class_only','parent_class','parent_class_aug','mobilenetv3','densenet121','vgg16'):
        assert f"'{s}'" in t
    assert "digest(file)!=SHA[arm]" in t

def test_uses_fixed_threshold_no_optimization():
    t=script.read_text()
    assert "p>=0.5" in t and "ck.get('threshold')!=0.5" in t
    for disallowed in ('model.fit(', 'optimizer.step(', 'load_state_dict_from_external_labels'):
        assert disallowed not in t

def test_no_false_blind_claim():
    t=script.read_text()
    assert "'evaluation_untouched_or_blind':False" in t
    assert "'test_time_training_or_calibration':False" in t

def test_source_license_and_fabric_breakdown():
    t=script.read_text()
    assert 'STITCHINGNET' not in t or '51b228d9572b925035e9e9af01c96569' in t
    assert "'by_fabric':breakdown" in t
    assert '2436/3637' in t