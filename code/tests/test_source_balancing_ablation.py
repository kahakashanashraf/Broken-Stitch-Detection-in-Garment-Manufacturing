"""Ablation is a paired A/B/C same-backbone intervention, not architecture search."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

here=Path(__file__).resolve().parents[1]/'scripts'
import sys
sys.path.insert(0,str(here))
import source_balancing_ablation as a

def toy():
    result=[]
    for source in ('aa','bb','cc'):
        for lab in ('actual','faulty'):
            for i in range({'aa':4,'bb':2,'cc':3}[source]):
                result.append({'label':lab,'split':'train','clean_path':f'train/{lab}/{source}-{i}.png',
                               'inferred_photographed_source':'photo_candidate:'+source})
    return result

def test_class_vs_parent_weights_are_interventions():
    rows=toy();ids=list(range(len(rows)))
    wa,augment_a=a.resolve_arm('class_only',rows,ids)
    wb,augment_b=a.resolve_arm('parent_class',rows,ids)
    wc,augment_c=a.resolve_arm('parent_class_aug',rows,ids)
    assert not augment_a and not augment_b and augment_c
    assert wb==wc and wa!=wb
    for w in [wa,wb,wc]:
        assert set(w)==set(ids)
        assert sum(w.values())==pytest.approx(len(ids))
        for lab in ('actual','faulty'):
            assert sum(weight for i,weight in w.items() if rows[i]['label']==lab)==pytest.approx(len(ids)/2)
    for group in ('aa','bb','cc'):
        ix=[i for i,r in enumerate(rows) if r['label']=='actual' and r['inferred_photographed_source'].endswith(group)]
        assert sum(wb[i] for i in ix)==pytest.approx(len(ids)/6)

def test_same_ids_across_arms():
    rows=toy();indices=list(range(len(rows)))
    selected=a.fixed_class_stratified_sample(rows,indices,0,n_per_class=4)
    assert len(selected)==8
    assert a.resolve_arm('class_only',rows,selected)[0].keys()==a.resolve_arm('parent_class_aug',rows,selected)[0].keys()

def test_invalid_weights_or_arms_rejected():
    rows=toy();ids=list(range(len(rows)))
    with pytest.raises(ValueError): a.resolve_arm('not_real',rows,ids)
    with pytest.raises(ValueError): a.resolve_arm('class_only',rows,[0,0])

def test_source_code_never_reads_test_images():
    txt=(here/'source_balancing_ablation.py').read_text()
    assert 'prior.verify_dev_inputs' in txt
    assert "SPLITS=('train','val')" not in txt  # existing canonical defines dev pool internally
    assert 'stitchingnet_not_opened' in txt
    assert 'fixed_class_stratified_sample' in txt
    assert 'StratifiedGroupKFold' in txt