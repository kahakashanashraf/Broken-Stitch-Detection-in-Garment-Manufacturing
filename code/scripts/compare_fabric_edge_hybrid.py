"""Paired experiment: archived MobileNetV3 baseline vs MobileNetV3+Sobel edge branch.

Same StitchingNet supervised fabric-heldout split (eleven fabric IDs, five folds).
Already inspected data; therefore exploratory, not independent confirmation.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import compare_fabric_adaptation as prior

BASELINE_DIR = "baseline"
HYBRID_DIR = "hybrid"
RUN_ID = "stitchingnet_synthetic_mobilenet_sobel_fusion_v1"


def checked_five_folds(root, model):
    root = Path(root)
    paths = []
    commits = set()
    for k in range(5):
        location = root / model / str(k)
        if model == HYBRID_DIR:
            report = json.loads((location / "fabric_edge_report.json").read_text())
            if (report.get("id") != RUN_ID or report.get("initialization") != "synthetic"
                    or report.get("fold") != k or report.get("epochs") != 6
                    or report.get("batch_size") != 16 or report.get("image_size") != 224
                    or report.get("dataset_md5") != "51b228d9572b925035e9e9af01c96569"
                    or report.get("synthetic_checkpoint_sha256") !=
                       "0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81"):
                raise ValueError("STOP: mismatched hybrid fold provenance")
            if set(report["train_fabrics"]) & set(report["heldout_fabrics"]):
                raise ValueError("STOP: train/heldout fabric overlap")
            commits.add(report.get("source_commit"))
            p = location / "fabric_edge_oof_predictions.csv"
            listed = list(csv.DictReader(p.read_text().splitlines()))
            if len(listed) != report["heldout_samples"] or set(x["fabric"] for x in listed) != set(report["heldout_fabrics"]):
                raise ValueError("STOP: report fabric identifiers or count disagree with OOF")
        else:
            p = location / "fabric_heldout_oof_predictions.csv"
        paths.append(p)
    if model == HYBRID_DIR and (len(commits) != 1 or not next(iter(commits))
                               or "local_unverified" in commits):
        raise ValueError("STOP: incomplete hybrid code revision provenance")
    return prior.read_oof(paths), next(iter(commits)) if model == HYBRID_DIR else None


def compare(root, out):
    baseline, _ = checked_five_folds(root, BASELINE_DIR)
    hybrid, revision = checked_five_folds(root, HYBRID_DIR)
    if set(baseline) != set(hybrid):
        raise ValueError("STOP: different OOF sample IDs")
    keys = sorted(baseline)
    for key in keys:
        if baseline[key][:3] != hybrid[key][:3]:
            raise ValueError("STOP: baseline/hybrid fabric, class or fold mismatch")
    y = np.array([int(baseline[k][1] == "broken") for k in keys])
    fabrics = np.array([baseline[k][0] for k in keys])
    probabilities = {
        "baseline": np.array([baseline[k][3] for k in keys]),
        "hybrid": np.array([hybrid[k][3] for k in keys]),
    }
    result = {arm: prior.metrics(y, probs) for arm, probs in probabilities.items()}
    groups = {f: np.where(fabrics == f)[0] for f in sorted(set(fabrics))}
    rng = np.random.default_rng(20261021)
    differences = []
    for _ in range(2000):
        draw = rng.choice(list(groups), len(groups), replace=True)
        indices = np.concatenate([groups[f] for f in draw])
        yy = y[indices]
        if set(yy) != {0, 1}:
            continue
        new = f1_score(yy, probabilities["hybrid"][indices] >= .5, average="macro")
        old = f1_score(yy, probabilities["baseline"][indices] >= .5, average="macro")
        differences.append(new - old)
    if len(differences) < 1900:
        raise ValueError("STOP: insufficient valid fabric-bootstrap draws")
    per_fold = {}
    for fold in range(5):
        selection = np.array([baseline[k][2] == fold for k in keys])
        per_fold[str(fold)] = {
            "fabrics": sorted(set(fabrics[selection])),
            "baseline": prior.metrics(y[selection], probabilities["baseline"][selection]),
            "hybrid": prior.metrics(y[selection], probabilities["hybrid"][selection])
        }
    report = {
        "experiment": RUN_ID,
        "status": "EXPLORATORY_STITCHINGNET_SUPERVISED_FABRIC_OOF_NOT_BLIND",
        "hybrid_code_commit": revision,
        "n_images": len(keys), "n_fabric_groups": len(groups),
        "fixed_training_epochs": 6, "fixed_threshold": .5,
        "same_folds_same_synthetic_initialization": True,
        "baseline_release_pattern": "stitchingnet-fabric-heldout-synthetic-f{0..4}-v1",
        "hybrid_release_pattern": "stitchingnet-edgehybrid-synthetic-f{0..4}-v1",
        "models": result,
        "delta_hybrid_minus_baseline_macro_f1": (
            result["hybrid"]["macro_f1"] - result["baseline"]["macro_f1"]),
        "descriptive_11fabric_bootstrap_95pct": np.percentile(differences, [2.5, 97.5]).tolist(),
        "bootstrap_draws": len(differences),
        "per_fold": per_fold,
        "limitations": [
            "The dataset was previously inspected; evaluation is retrospective and exploratory.",
            "Same fabric categories can still share machine/session/camera correlation.",
            "Edge maps are hand-designed features, not validated defect segmentations.",
            "Any improvement over the old baseline needs independent newly acquired fabric/session confirmation.",
            "This experiment compares an added branch, NOT a novel architecture or a clean published SOTA benchmark."
        ]
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge_hybrid_paired_comparison.json").write_text(json.dumps(report, indent=2))
    lines = ["# Edge branch vs original MobileNetV3 — Fabric-heldout comparison", "",
             "Retrospective, previously inspected StitchingNet 11-fabric supervised CV; **not independent external testing**.", "",
             "| Model | Macro-F1 | Balanced accuracy | Broken recall | False alarms | Missed broken |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in ["baseline", "hybrid"]:
        s = result[name]
        lines.append(f"| {name} | {s['macro_f1']:.4f} | {s['balanced_accuracy']:.4f} | {s['broken_recall']:.4f} | {s['fp']} | {s['fn']} |")
    lines.extend(["", f"Observed hybrid minus baseline macro-F1: **{report['delta_hybrid_minus_baseline_macro_f1']:+.4f}**.",
                  "The 11-fabric-cluster descriptive bootstrap interval is " +
                  str(report["descriptive_11fabric_bootstrap_95pct"]) + ".",
                  "Evaluation folds, labels and decision threshold were held fixed. Interpret as hypothesis testing on an already-seen dataset."])
    (out / "EDGE_HYBRID_PAIRED_COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("EDGE_HYBRID_PAIRED", json.dumps({"scores": result,
          "difference": report["delta_hybrid_minus_baseline_macro_f1"]}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    compare(a.root, a.out)
