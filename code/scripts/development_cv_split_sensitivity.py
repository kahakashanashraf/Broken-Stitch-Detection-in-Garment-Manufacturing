"""Prespecified split-protocol sensitivity study on DEVELOPMENT records ONLY.

OLD held-out five inferred photographed source parents and StitchingNet are NOT used.
Image-stratified CV is an intentionally contaminated diagnostic (all variants of
one parent can cross folds), not an estimate of deployment generalization.
Grouped CV is based on unverified filename-inferred photographed source IDs.
The comparison is observational; split/composition differs between protocols.
"""
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from broken_stitch.metrics import compute_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mobilenet_feature_baseline as baseline

BAD = 'dataset/train/actual/actual_original_AB.png_579370ba-65d0-4793-a3c0-0f73fbf6c367.png'
EXPECTED_ZIP_SIZE = 2146784353
SEED = 20261019
N_FOLDS = 5


def crossvalidate_features(X, y, groups, splitter, protocol):
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    val_assign = np.full(len(y), -1, dtype=int)
    fold_reports = []
    folds = splitter.split(X, y, groups) if protocol == 'inferred_photo_grouped' else splitter.split(X, y)
    for k, (tr, va) in enumerate(folds):
        tr_groups = set(groups[tr]); va_groups = set(groups[va]); crossing = tr_groups & va_groups
        if protocol == 'inferred_photo_grouped' and crossing:
            raise ValueError('STOP: source-photo parent group leakage in grouped fold')
        if len(set(y[tr])) != 2 or len(set(y[va])) != 2:
            raise ValueError(f'STOP: fold {k} is missing a class')
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(
            C=1.0, class_weight='balanced', solver='lbfgs', max_iter=350,
            random_state=SEED,
        )
        model.fit(scaler.transform(X[tr]), y[tr])
        p = model.predict_proba(scaler.transform(X[va]))[:, 1]
        predictions[va] = p
        val_assign[va] = k
        scores = compute_metrics(y[va], p)
        fold_reports.append({
            'fold':k, 'train_n':len(tr), 'val_n':len(va),
            'val_normal':int((y[va] == 0).sum()),
            'val_faulty':int((y[va] == 1).sum()),
            'train_parent_n':len(tr_groups), 'val_parent_n':len(va_groups),
            'crossing_parent_n':len(crossing),
            'metric':scores,
        })
    if np.isnan(predictions).any() or not np.all(val_assign >= 0):
        raise ValueError('STOP: CV did not produce a full out-of-fold prediction set')
    metric = compute_metrics(y, predictions)
    return {
        'protocol':protocol, 'oof_sample_n':len(y), 'oof_metrics':metric,
        'folds':fold_reports,
        'fold_macro_f1_mean':float(np.mean([x['metric']['macro_f1'] for x in fold_reports])),
        'fold_macro_f1_sd':float(np.std([x['metric']['macro_f1'] for x in fold_reports], ddof=1)),
        'fold_bal_accuracy_mean':float(np.mean([x['metric']['balanced_accuracy'] for x in fold_reports])),
        'fold_bal_accuracy_sd':float(np.std([x['metric']['balanced_accuracy'] for x in fold_reports], ddof=1)),
    }


def run(args):
    if Path(args.zip).stat().st_size != EXPECTED_ZIP_SIZE:
        raise ValueError('STOP: expected original source ZIP size')
    raw = Path(args.manifest).read_bytes()
    rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if len(rows) != 7365 or sum(r['relative_path'] == BAD for r in rows) != 1:
        raise ValueError('STOP: unexpected source manifest or corrupt-image list')
    rows = [r for r in rows if r['relative_path'] != BAD]
    dev = [r for r in rows if r['split'] in ('train', 'val')]
    heldout = [r for r in rows if r['split'] == 'test']
    if len(dev) != 6278 or len(heldout) != 1086:
        raise ValueError('STOP: this run must only use the 6278-image development pool')
    groups_dev = {r['inferred_photographed_source'] for r in dev}
    groups_test = {r['inferred_photographed_source'] for r in heldout}
    if len(groups_dev) != 26 or len(groups_test) != 5 or (groups_dev & groups_test):
        raise ValueError('STOP: group separation from previously inspected holdout invalid')
    assert all(r['label'] in ('actual', 'faulty') for r in rows)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = baseline.extractor(pretrained=True)
    with zipfile.ZipFile(args.zip) as z:
        X, y, invalid = baseline.features_from_zip(model, z, dev)
    if invalid or len(X) != len(dev):
        raise ValueError('STOP: feature extraction skipped a development image')
    groups = np.asarray([r['inferred_photographed_source'] for r in dev])
    random_result = crossvalidate_features(
        X, y, groups,
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
        'image_stratified_diagnostic',
    )
    grouped_result = crossvalidate_features(
        X, y, groups,
        StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
        'inferred_photo_grouped',
    )
    if sum(f['crossing_parent_n'] == 0 for f in grouped_result['folds']) != N_FOLDS:
        raise ValueError('STOP: group leak')
    if sum(f['crossing_parent_n'] > 0 for f in random_result['folds']) != N_FOLDS:
        raise ValueError('STOP: intended random-image contamination not present')
    oof_diff = random_result['oof_metrics']['macro_f1'] - grouped_result['oof_metrics']['macro_f1']
    result = {
        'status':'EXPLORATORY_DEVELOPMENT_ONLY_SPLIT_PROTOCOL_DIAGNOSTIC',
        'no_prior_holdout_access':True,
        'no_stitchingnet_access':True,
        'no_new_independent_test':True,
        'same_feature_extractor':'Frozen ImageNet1K-V1 MobileNetV3-Small',
        'same_classifier':'StandardScaler train-fold only, logistic C=1 class_weight balanced threshold 0.5',
        'dev_image_count':len(dev),
        'dev_parent_candidate_count':len(groups_dev),
        'previously_inspected_heldout_parents_not_used':len(groups_test),
        'original_zip_file_size_bytes':EXPECTED_ZIP_SIZE,
        'manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'seed':SEED, 'folds':N_FOLDS,
        'random_cv':random_result,
        'parent_grouped_cv':grouped_result,
        'random_minus_grouped_oof_macro_f1':float(oof_diff),
        'interpretation':'Protocol-sensitivity diagnostic; does NOT causally estimate bias of historical 94.91% or validate true photographic source identities.',
        'caveat':'The 26 candidate parent IDs are inferred from naming; fold metrics on augmented records are correlated and not production estimates. CV built after other test scores had been inspected.',
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print('DIAGNOSTIC_SUMMARY', json.dumps({
        'random_oof_macro_f1':random_result['oof_metrics']['macro_f1'],
        'grouped_oof_macro_f1':grouped_result['oof_metrics']['macro_f1'],
        'difference':oof_diff,
        'random_oof_balacc':random_result['oof_metrics']['balanced_accuracy'],
        'grouped_oof_balacc':grouped_result['oof_metrics']['balanced_accuracy'],
        'random_fold_crossing_n':[x['crossing_parent_n'] for x in random_result['folds']],
        'grouped_fold_crossing_n':[x['crossing_parent_n'] for x in grouped_result['folds']],
    }, indent=2), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--zip', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    run(p.parse_args())