import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fabric_prior_adjustment as prior


def test_threshold_only_from_training_fabric_counts():
    r={'train_samples':2700,'training_label_counts':{'0':1800,'1':900}}
    assert prior.train_prior_threshold(r)==pytest.approx(2/3)
    r['training_label_counts']={'0':900,'1':1800}
    assert prior.train_prior_threshold(r)==pytest.approx(1/3)


def test_reject_bad_training_counts():
    with pytest.raises(ValueError):prior.train_prior_threshold({'train_samples':1000,'training_label_counts':{'0':990,'1':10}})
    with pytest.raises(ValueError):prior.train_prior_threshold({'train_samples':1000,'training_label_counts':{'0':200,'1':300}})


def test_operating_point_metrics():
    y=[0,0,1,1]
    assert prior.measures(y,[0,0,1,1])['macro_f1']==1
    assert prior.measures(y,[0,1,0,1])['macro_f1']==pytest.approx(.5)


def test_strictly_descriptive_not_independent_test():
    text=Path(prior.__file__).read_text()
    assert 'post-hoc exploratory' in text
    assert 'heldout_fabric_labels_used_to_choose_thresholds' in text