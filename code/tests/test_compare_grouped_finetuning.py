from pathlib import Path
import sys
import csv
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_grouped_finetuning as c

def test_metrics_class_balanced():
    x=c.metrics([0,0,1,1],[.1,.2,.8,.9])
    assert x['macro_f1']==1 and x['faulty_recall']==1

def test_strict_group_and_path_validation(tmp_path):
    p=tmp_path/'oof.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty'])
        w.writerow(['train/actual/a','actual','photo_candidate:aa',0,.3])
        w.writerow(['train/actual/a','actual','photo_candidate:aa',0,.3])
    with pytest.raises(ValueError,match='6278'):
        c.read_oof(p)

def test_bootstrap_previously_inspected_holdout_excluded():
    t=(Path(__file__).resolve().parents[1]/'scripts'/'compare_grouped_finetuning.py').read_text()
    assert "'test_and_stitchingnet_data_access':False" in t
    assert 'inferred' in t and 'bootstrap' in t