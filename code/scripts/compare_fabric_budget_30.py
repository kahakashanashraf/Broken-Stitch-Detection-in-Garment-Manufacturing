"""Audit and aggregate the immutable 6-vs-30-epoch fabric-heldout experiments.

Data already inspected in the project: descriptive OOF analysis, not an
untouched independent test. No tuning of threshold or epoch on held-out data.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import compare_fabric_adaptation as prior

ARMS = ("imagenet", "synthetic")
EPOCHS = (6, 30)
EXPECTED_N = 3637
EXPECTED_FABRICS = set("ABCDEFGHIJK")
RUN_ID = "stitchingnet_fabric_heldout_budget6_vs30_v2"


def read_arm(root, arm):
    predictions = {}
    fold_of_fabric = {}
    commits = set()
    for fold in range(5):
        folder = Path(root) / arm / str(fold)
        csv_file = folder / "budget_heldout_oof.csv"
        report_file = folder / "budget_heldout_report.json"
        report = json.loads(report_file.read_text())
        if (report.get("experiment") != RUN_ID or report.get("arm") != arm
                or report.get("fold") != fold or report.get("epochs") != 30
                or report.get("milestones") != [6, 30]
                or report.get("batch_size") != 16
                or report.get("resolution") != 224
                or report.get("data_zip_md5") != "51b228d9572b925035e9e9af01c96569"):
            raise ValueError("STOP: mismatched training report/protocol")
        if report.get("heldout_samples", 0) <= 0:
            raise ValueError("STOP: missing heldout sample count")
        if set(report["training_fabrics"]) & set(report["heldout_fabrics"]):
            raise ValueError("STOP: fold fabric leakage")
        if any(f not in EXPECTED_FABRICS for f in report["training_fabrics"] + report["heldout_fabrics"]):
            raise ValueError("STOP: invalid report fabric identity")
        commits.add(report["source_commit"])
        rows = list(csv.DictReader(csv_file.open(newline="")))
        if len(rows) != report["heldout_samples"]:
            raise ValueError("STOP: heldout count disagrees with report")
        for r in rows:
            name = r["source_file"]
            fabric = r["fabric"]
            label = r["actual_class"]
            values = tuple(float(r["probability_broken_epoch" + str(ep)]) for ep in EPOCHS)
            if (name in predictions or fabric not in EXPECTED_FABRICS
                    or fabric not in report["heldout_fabrics"]
                    or label not in {"normal", "broken"} or int(r["fold"]) != fold
                    or any(not np.isfinite(p) or not 0 <= p <= 1 for p in values)):
                raise ValueError("STOP: duplicate, invalid, or misassigned heldout sample")
            if fabric in fold_of_fabric and fold_of_fabric[fabric] != fold:
                raise ValueError("STOP: fabric assigned to more than one heldout fold")
            fold_of_fabric[fabric] = fold
            predictions[name] = (fabric, label, fold, *values)
    if len(commits) != 1 or "local_unverified" in commits:
        raise ValueError("STOP: fold code commits mismatch or missing")
    if len(predictions) != EXPECTED_N or set(fold_of_fabric) != EXPECTED_FABRICS:
        raise ValueError("STOP: incomplete 3,637-image 11-fabric OOF")
    if set(fold_of_fabric.values()) != set(range(5)):
        raise ValueError("STOP: missing heldout fold")
    if Counter(v[1] for v in predictions.values()) != Counter({"normal": 2436, "broken": 1201}):
        raise ValueError("STOP: invalid class counts")
    return predictions, next(iter(commits))


def build_comparison(root, output):
    predictions = {}
    commits = set()
    for arm in ARMS:
        predictions[arm], commit = read_arm(root, arm)
        commits.add(commit)
    if len(commits) != 1:
        raise ValueError("STOP: arms came from different code commits")
    reference = predictions["imagenet"]
    keys = sorted(reference)
    if any(set(predictions[arm]) != set(keys) for arm in ARMS):
        raise ValueError("STOP: initialization arms evaluated different images")
    if any(predictions["synthetic"][k][:3] != reference[k][:3] for k in keys):
        raise ValueError("STOP: mismatched fabric, label or fold between arms")
    y = np.array([reference[k][1] == "broken" for k in keys], dtype=int)
    fabrics = np.array([reference[k][0] for k in keys])
    scores = {(arm, epoch): np.array([predictions[arm][k][3 + e] for k in keys])
              for arm in ARMS for e, epoch in enumerate(EPOCHS)}
    result = {arm: {str(ep): prior.metrics(y, scores[(arm, ep)]) for ep in EPOCHS}
              for arm in ARMS}
    per_fold = {}
    for fold in range(5):
        mask = np.array([reference[k][2] == fold for k in keys])
        per_fold[str(fold)] = {
            "heldout_fabrics": sorted(set(fabrics[mask])),
            "metrics": {arm: {str(ep): prior.metrics(y[mask], scores[(arm, ep)][mask])
                             for ep in EPOCHS} for arm in ARMS}
        }
    rng = np.random.default_rng(20261021)
    cluster_ids = {f: np.where(fabrics == f)[0] for f in sorted(set(fabrics))}
    by_arm = {}
    for arm in ARMS:
        diffs = []
        for _ in range(2000):
            draw = rng.choice(sorted(cluster_ids), len(cluster_ids), replace=True)
            indices = np.concatenate([cluster_ids[f] for f in draw])
            yy = y[indices]
            if set(yy) != {0, 1}:
                continue
            a = f1_score(yy, scores[(arm, 6)][indices] >= .5, average="macro")
            b = f1_score(yy, scores[(arm, 30)][indices] >= .5, average="macro")
            diffs.append(b - a)
        if len(diffs) < 1900:
            raise ValueError("STOP: insufficient fabric-bootstrap samples")
        by_arm[arm] = {
            "delta_macro_f1_epoch30_minus_epoch6":
                result[arm]["30"]["macro_f1"] - result[arm]["6"]["macro_f1"],
            "descriptive_fabric_cluster_bootstrap_95pct":
                np.percentile(diffs, [2.5, 97.5]).tolist(),
            "valid_replicates": len(diffs)
        }
    report = {
        "status": "DESCRIPTIVE_SUPERVISED_STITCHINGNET_FABRIC_OOF_NONBLIND",
        "experiment": RUN_ID, "code_commit": next(iter(commits)),
        "n_images": len(y), "n_fabrics": len(cluster_ids),
        "comparison": "6 vs 30 epochs, same initial weights and training stream, same 5 fabric-group folds, 0.5 threshold",
        "metric": "pooled held-out-fabric OOF macro-F1 primary; balanced accuracy, recall and false alarms secondary",
        "scores": result, "paired_training_budget_differences": by_arm,
        "per_fold": per_fold,
        "limitations": [
            "Data already inspected before this comparison, so exploratory and not a blind test.",
            "Outer-fold scores at both milestones were read-only; selecting the winner using them would make follow-up estimates non-confirmatory.",
            "11 fabric identities do not guarantee independent image-capture sessions or machines.",
            "A change in epoch budget does not prove transfer to unseen factories.",
            "Two initialization arms do not imply a novel architecture or demonstrated superiority over prior paper protocols."
        ]
    }
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fabric_budget6_vs30_comparison.json").write_text(json.dumps(report, indent=2))
    lines = ["# Held-out fabric supervised training budget: 6 vs 30 epochs", "",
             "**Exploratory, previously inspected StitchingNet dataset; no blind external test.**", "",
             "| Initialization | 6-epoch macro-F1 | 30-epoch macro-F1 | Difference | 30-epoch broken recall | 30-epoch false alarms |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        a = result[arm]["6"]
        b = result[arm]["30"]
        d = by_arm[arm]
        lines.append(f"| {arm} | {a['macro_f1']:.4f} | {b['macro_f1']:.4f} | {d['delta_macro_f1_epoch30_minus_epoch6']:+.4f} | {b['broken_recall']:.4f} | {b['fp']} |")
    lines.extend(["", "Paired differences have *descriptive* 11-fabric cluster bootstrap intervals.",
                  "Holdout groups were not used for training or epoch selection; 6 and 30 were fixed before training.",
                  "No industrial/SOTA claim, and no inference on a newly acquired independent factory dataset."])
    (dest / "FABRIC_BUDGET6_VS30_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("BUDGET_COMPARE", json.dumps({"scores": result, "differences": by_arm}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_comparison(args.root, args.out)
