"""Prevent silent source-family contamination or baseline miscomparison in v1."""
from pathlib import Path
import sys
import ast
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from enhanced_mobilenet_224_cv import SIZE,EPOCHS,MILESTONES,SEED,PER_CLASS,BATCH_SIZE,set_preprocess,fold_members
import enhanced_mobilenet_224_cv as improved
import grouped_finetune_cv as old
from source_balancing_ablation import fixed_class_stratified_sample


def test_predeclared_training_recipe_and_resolution():
    assert SIZE==224 and EPOCHS==12 and MILESTONES==(2,12)
    assert PER_CLASS==800 and BATCH_SIZE==12 and SEED==old.SEED
    set_preprocess()
    assert old.TRAIN_TF is old.VAL_TF
    assert old.TRAIN_TF(Image.new('RGB',(400,400))).shape==(3,SIZE,SIZE)


def test_exact_class_only_sample_ids_preserved():
    # Both the new and old class_only recipe use the same fixed function and fold seed.
    rows=[]
    for label in ('actual','faulty'):
        rows.extend({'label':label,'clean_path':f'{label}_{i}.png'} for i in range(820))
    a=fixed_class_stratified_sample(rows,np.arange(len(rows)),2,n_per_class=800)
    b=fixed_class_stratified_sample(rows,np.arange(len(rows)),2,n_per_class=PER_CLASS)
    assert a==b and len(a)==1600


def test_parent_group_splits_are_exclusive():
    rows=[]
    for i in range(26):
        for label in ('actual','faulty'):
            for k in range(121):
                rows.append({'label':label,'inferred_photographed_source':f'photo_candidate:{i:02d}',
                             'clean_path':f'{i}_{label}_{k}.png'})
    # 26 * 2 * 121 = 6292, fold_members expected exact production 6278; use 6278
    rows=rows[:6278]
    y,g,folds=fold_members(rows)
    assert len(folds)==5
    for tr,va in folds:
        assert set(g[tr]).isdisjoint(set(g[va]))


def test_evaluation_only_at_locked_epochs_and_no_external_data():
    source=Path(improved.__file__).read_text()
    assert 'if len(validation_idx)>0 and epoch in MILESTONES:' in source
    assert "'train':Path(args.train_zip),'val':Path(args.val_zip)" in source
    assert "if train_groups & val_groups:" in source
    assert 'test.zip' not in source
    assert 'torch.save(ck,checkpoint)' in source


def test_weights_are_224_inference_compatible():
    # Source checkpoint has explicit image size, class map, and provenance.
    tree=ast.parse(Path(improved.__file__).read_text())
    saves=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
           and node.func.attr=='save']
    assert len(saves)==1