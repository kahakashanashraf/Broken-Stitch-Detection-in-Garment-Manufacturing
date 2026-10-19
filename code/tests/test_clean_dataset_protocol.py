"""Unit tests for fail-closed photo-root grouping. Synthetic names only."""
import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'scripts'/'build_clean_grouped_dataset.py'
s=importlib.util.spec_from_file_location('grouped_build',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_parent_extraction_across_labels():
    assert m.family_photo_base('actual_original_aq','actual')=='aq'
    assert m.family_photo_base('faulty_original_aq6','faulty')=='aq'

def test_wrong_label_or_missing_fault_variant_fails():
    with pytest.raises(ValueError):m.family_photo_base('faulty_original_aq','faulty')
    with pytest.raises(ValueError):m.family_photo_base('actual_original_aq3','actual')
    with pytest.raises(ValueError):m.family_photo_base('faulty_original_aq2','actual')

def test_reject_group_cross_split_before_count_validation():
    rows=[
      {'relative_path':'dataset/train/actual/x.png','split':'train','label':'actual',
       'candidate_image_family':'actual_original_aa','inferred_photographed_source':'photo_candidate:aa'},
      {'relative_path':'dataset/test/faulty/y.png','split':'test','label':'faulty',
       'candidate_image_family':'faulty_original_aa1','inferred_photographed_source':'photo_candidate:aa'}
    ]
    with pytest.raises(ValueError,match='crosses splits'):m.validate_group_splits(rows)

def test_reject_wrong_photo_root():
    rows=[{'relative_path':'dataset/train/faulty/y.png','split':'train','label':'faulty',
       'candidate_image_family':'faulty_original_bb1','inferred_photographed_source':'photo_candidate:aa'}]
    with pytest.raises(ValueError,match='inconsistent'):m.validate_group_splits(rows)

def test_no_new_test_leakage_claim():
    source=p.read_text()
    assert 'test_condition' in source and 'not a fresh untouched test' in source
