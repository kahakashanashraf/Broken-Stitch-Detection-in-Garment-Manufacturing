"""Guardrails for frozen holdout evaluation. No dataset needed."""
from pathlib import Path
import importlib.util

def test_frozen_holdout_script_has_no_model_training():
    p=Path(__file__).resolve().parents[2]/'scripts'/'one_shot_photo_parent_holdout.py'
    t=p.read_text()
    assert 'predict_zip' in t and 'EXPECTED_CHECKPOINT_SHA' in t
    assert 'manifest_sha256' in t and 'fixed_threshold' in t
    assert 'LogisticRegression(' not in t and '.fit(' not in t

def test_frozen_holdout_rejects_changed_threshold():
    p=Path(__file__).resolve().parents[2]/'scripts'/'one_shot_photo_parent_holdout.py'
    t=p.read_text()
    assert "ck['fixed_threshold']!=0.5" in t
    assert "'test'" in t and "'train'" in t
