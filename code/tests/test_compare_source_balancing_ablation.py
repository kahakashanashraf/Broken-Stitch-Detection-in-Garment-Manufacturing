import csv
import json
import sys
from pathlib import Path
import pytest

here=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(here))
import compare_source_balancing_ablation as comparison


def fake_oof(folder,kind,wrong_fold=False):
    dest=folder/kind;dest.mkdir(parents=True,exist_ok=True)
    with (dest/'oof_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty'])
        for parent in range(26):
            for i in range(242 if parent<12 else 241):
                fold=parent%5
                if wrong_fold and parent==4 and i==0:fold=1
                lab='actual' if i%2==0 else 'faulty'
                w.writerow([f'train/{lab}/p{parent}_{i}.png',lab,f'photo_candidate:p{parent}',fold,0.3 if lab=='actual' else 0.7])
    return dest/'oof_predictions.csv'


def test_comparison_rejects_mismatched_fold(tmp_path):
    files={a:fake_oof(tmp_path,a,a=='parent_class_aug') for a in comparison.ARMS}
    with pytest.raises(ValueError,match='crosses OOF validation folds'):
        comparison.compare(files,tmp_path/'output')


def test_missing_test_data_guard_in_source():
    x=(here/'compare_source_balancing_ablation.py').read_text()
    assert 'test_and_stitchingnet_opened' in x
    assert 'paired_parent_bootstrap_descriptive_95pct' in x
    assert 'selected_image_paths_sha256' in x