"""Synthetic data guards for paired baseline vs enhanced OOF comparison."""
from pathlib import Path
import csv
import sys
import pytest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_enhanced_mobilenet_224 import read_old,read_new,score


def test_invalid_probabilities_are_rejected(tmp_path):
    p=tmp_path/'bad.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty']);w.writerow(['x','actual','photo_candidate:a',0,'NaN'])
    with pytest.raises(ValueError,match='invalid previous probability'):
        read_old(p)


def test_paired_new_probabilities_and_parent_fold_are_checked(tmp_path):
    rows={'x':{'label':'faulty','parent':'photo_candidate:a','fold':0,'baseline128e2':.3}}
    fs=[]
    for k in range(5):
        p=tmp_path/f'f{k}.csv'
        with p.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['path','label','inferred_parent','fold','p_faulty_epoch2','p_faulty_epoch12'])
            if k==0:w.writerow(['x','faulty','photo_candidate:BAD',0,.3,.4])
        fs.append(p)
    with pytest.raises(ValueError,match='mismatch'):
        read_new(fs,rows)


def test_binary_score_is_class_wise():
    a=score(np.array([0,0,1,1]),np.array([.1,.3,.6,.9]))
    assert a['macro_f1']==1.0 and a['faulty_recall']==1.0
    assert a['faulty_missed']==0