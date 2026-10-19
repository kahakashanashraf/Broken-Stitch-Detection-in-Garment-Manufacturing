"""Ensure split sensitivity study does not load formerly heldout labels for fitting."""
from pathlib import Path
import ast

P = Path(__file__).resolve().parents[1] / 'scripts' / 'development_cv_split_sensitivity.py'


def test_dev_protocol_excludes_test_images():
    s = P.read_text()
    assert "r['split'] in ('train', 'val')" in s
    assert "len(dev) != 6278" in s
    assert "len(heldout) != 1086" in s
    assert 'groups_dev & groups_test' in s
    assert 'original_zip_file_size_bytes' in s


def test_only_train_fold_fits_scaler_and_classifier():
    s = P.read_text()
    assert 'StandardScaler().fit(X[tr])' in s
    assert 'model.fit(scaler.transform(X[tr]), y[tr])' in s
    assert 'model.predict_proba(scaler.transform(X[va]))' in s
    assert 'StratifiedGroupKFold(' in s
    assert 'StratifiedKFold(' in s
    assert 'crossing_parent_n' in s
    assert "'no_new_independent_test':True" in s


def test_source_script_parses():
    ast.parse(P.read_text())