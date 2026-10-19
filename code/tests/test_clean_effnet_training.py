"""Unit checks without downloading original images or ImageNet pretrained weights."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC=Path(__file__).resolve().parents[1]/'scripts'/'train_clean_effnet.py'
spec=importlib.util.spec_from_file_location('train_clean_effnet',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def _row(label,grp,split='train'):
    return {'label':label,'inferred_photographed_source':grp,'split':split}


def test_photo_class_weighting_gives_equal_mass_per_class_and_photo():
    rows=([_row('actual','x')]*10+[_row('actual','y')]*2+
          [_row('faulty','x')]*3+[_row('faulty','z')]*9)
    weights=m.weights_by_photo_and_class(rows)
    assert len(weights)==len(rows)
    assert abs(weights[:10].sum()-weights[10:12].sum())<1e-6
    assert abs(weights[12:15].sum()-weights[15:].sum())<1e-6
    assert abs(weights[:12].sum()-weights[12:].sum())<1e-6
    assert abs(weights.sum()-len(rows))<1e-6


def test_weighting_rejects_validation_data():
    with pytest.raises(ValueError,match='train images'):
        m.weights_by_photo_and_class([_row('actual','x'),_row('faulty','y','val')])


def test_saved_head_probability_matches_sklearn():
    x=np.array([[1.0,2.0],[2.0,1.0],[4.0,1.0],[5.0,3.0],
                [1.5,3.5],[6.0,1.1]],dtype=np.float64)
    y=np.array([0,0,1,1,0,1])
    sc=StandardScaler().fit(x)
    clf=LogisticRegression(C=1.0).fit(sc.transform(x),y)
    head={'coef':clf.coef_[0],'intercept':clf.intercept_[0],
          'mean':sc.mean_,'scale':sc.scale_}
    a=m.logistic_predict_numpy(x,head)
    b=clf.predict_proba(sc.transform(x))[:,1]
    assert np.allclose(a,b,atol=1e-7)


def test_test_archive_absent_from_training_script():
    s=SRC.read_text()
    assert "paths={'train':Path(args.train_zip),'val':Path(args.val_zip)}" in s
    assert 'test.zip' not in s.replace('test.zip and StitchingNet','')
    assert "'test_accessed':False" in s