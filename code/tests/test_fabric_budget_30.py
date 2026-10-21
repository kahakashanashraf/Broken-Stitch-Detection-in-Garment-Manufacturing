"""Synthetic-data integrity checks for the 6-vs-30 fixed fabric budget trial."""
import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fabric_budget_30_cv as trial
import compare_fabric_budget_30 as compare


def test_locked_budget_protocol():
    assert trial.EPOCHS == 30 and trial.MILESTONES == (6, 30)
    assert trial.BATCH_SIZE == 16 and trial.ARMS == ("imagenet", "synthetic")
    assert trial.SEED == 20261020 and trial.FOLDS == 5
    args = argparse.Namespace(epochs=30, batch_size=16, arm="imagenet", fold="0")
    trial.validate_args(args)
    args.epochs = 6
    with pytest.raises(ValueError, match="STOP"):
        trial.validate_args(args)
    args.epochs = 30
    args.fold = "5"
    with pytest.raises(ValueError, match="STOP"):
        trial.validate_args(args)


def make_fake_outputs(root):
    """3,637 aligned samples, 11 full fabrics, five folds and both classes."""
    fabrics = "ABCDEFGHIJK"
    fold_map = {f: i % 5 for i, f in enumerate(fabrics)}
    entries = []
    # Exactly 2436 normal and 1201 broken; spread both across every fabric.
    for label, count in [("normal", 2436), ("broken", 1201)]:
        for i in range(count):
            fab = fabrics[i % 11]
            entries.append((f"fake/{label}_{i}_{fab}.png", fab, label, fold_map[fab]))
    for arm in compare.ARMS:
        for fold in range(5):
            folder = root / arm / str(fold)
            folder.mkdir(parents=True)
            subset = [r for r in entries if r[3] == fold]
            ff = sorted(f for f in fabrics if fold_map[f] == fold)
            report = {
                "experiment": compare.RUN_ID, "arm": arm, "fold": fold,
                "epochs": 30, "milestones": [6, 30], "batch_size": 16,
                "resolution": 224, "data_zip_md5": "51b228d9572b925035e9e9af01c96569",
                "source_commit": "test-frozen-commit", "heldout_samples": len(subset),
                "heldout_fabrics": ff,
                "training_fabrics": sorted(set(fabrics) - set(ff))
            }
            (folder / "budget_heldout_report.json").write_text(json.dumps(report))
            with (folder / "budget_heldout_oof.csv").open("w", newline="") as fd:
                w = csv.writer(fd)
                w.writerow(["source_file", "fabric", "actual_class", "fold",
                            "probability_broken_epoch6", "probability_broken_epoch30"])
                for name, fab, lab, k in subset:
                    good = 0.85 if lab == "broken" else 0.15
                    # Same class labels; different risk profiles at milestone 30.
                    improved = 0.9 if lab == "broken" else 0.1
                    w.writerow([name, fab, lab, k, good, improved])
    return entries


def test_all_folds_and_arms_aggregate_without_test_data(tmp_path):
    entries = make_fake_outputs(tmp_path / "input")
    report = compare.build_comparison(tmp_path / "input", tmp_path / "out")
    assert report["n_images"] == 3637
    assert report["n_fabrics"] == 11
    assert report["code_commit"] == "test-frozen-commit"
    assert report["scores"]["imagenet"]["6"]["macro_f1"] == pytest.approx(1)
    assert (tmp_path / "out" / "fabric_budget6_vs30_comparison.json").exists()
    assert len(entries) == 3637


def test_fail_closed_incomplete_fold(tmp_path):
    make_fake_outputs(tmp_path)
    file = tmp_path / "imagenet" / "0" / "budget_heldout_oof.csv"
    lines = file.read_text().splitlines()
    file.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="STOP"):
        compare.build_comparison(tmp_path, tmp_path / "out")


def test_fail_closed_report_provenance_mismatch(tmp_path):
    make_fake_outputs(tmp_path)
    file = tmp_path / "synthetic" / "3" / "budget_heldout_report.json"
    obj = json.loads(file.read_text())
    obj["source_commit"] = "incorrect-commit"
    file.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="STOP"):
        compare.build_comparison(tmp_path, tmp_path / "out")


def test_fail_closed_cross_arm_label_or_fabric_mismatch(tmp_path):
    make_fake_outputs(tmp_path)
    file = tmp_path / "synthetic" / "0" / "budget_heldout_oof.csv"
    lines = file.read_text().splitlines()
    parts = lines[1].split(",")
    parts[2] = "broken" if parts[2] == "normal" else "normal"
    lines[1] = ",".join(parts)
    file.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="STOP"):
        compare.build_comparison(tmp_path, tmp_path / "out")
