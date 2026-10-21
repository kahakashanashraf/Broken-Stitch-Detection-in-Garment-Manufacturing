"""CPU-only reproducibility and shape tests for the proposed edge-fusion ablation."""
import argparse
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fabric_edge_hybrid_cv as hybrid
import fabric_group_adaptation as base
import compare_fabric_edge_hybrid as comparison


def test_gate_starts_as_exact_baseline_not_fabric_guess():
    torch.manual_seed(42)
    backbone = base.FineTuneBinary("mobilenetv3", pretrained=False)
    baseline_rng = torch.get_rng_state().clone()
    model = hybrid.EdgeFusionMobileNet(backbone)
    assert torch.equal(baseline_rng, torch.get_rng_state()), "edge init changed baseline augmentation RNG"
    assert not model.edge_gate.item()
    model.eval()
    x = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        ref = backbone(x)
        prediction = model(x)
    assert prediction.shape == ref.shape == (2,)
    assert torch.equal(ref, prediction), "zero-gated hybrid must equal pretrained baseline"


def test_edge_branch_gradients_and_backbone_batchnorm_freeze():
    torch.manual_seed(71)
    backbone = base.FineTuneBinary("mobilenetv3", pretrained=False)
    model = hybrid.EdgeFusionMobileNet(backbone)
    model.train()
    x = torch.randn(2, 3, 96, 96)
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(model.edge_gate.grad).all()
    assert all(not m.training for m in model.backbone.features.modules()
               if isinstance(m, torch.nn.BatchNorm2d))
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.edge_gate.fill_(0.2)
    model(x).square().mean().backward()
    assert model.edge_head.weight.grad is not None
    assert torch.isfinite(model.edge_head.weight.grad).all()
    assert model.edge_features[0].weight.grad is not None


def test_train_budget_and_input_lock():
    assert hybrid.EPOCHS == base.EPOCHS == 6
    assert hybrid.BATCH == base.BATCH == 16
    assert hybrid.ARM == "synthetic"
    assert hybrid.FOLDS == 5
    arg = argparse.Namespace(fold="0", epochs=6, batch_size=16)
    hybrid.validate_args(arg)
    arg.epochs = 30
    with pytest.raises(ValueError, match="STOP"):
        hybrid.validate_args(arg)


def fake_fold_sets(root):
    fabrics = "ABCDEFGHIJK"
    all_rows = []
    folds = {f: i % 5 for i, f in enumerate(fabrics)}
    for label, num in [("normal", 2436), ("broken", 1201)]:
        for i in range(num):
            fab = fabrics[i % len(fabrics)]
            all_rows.append((f"case/{fab}_{label}_{i}.png", fab, label, folds[fab]))
    for kind in ("baseline", "hybrid"):
        for fold in range(5):
            p = root / kind / str(fold)
            p.mkdir(parents=True, exist_ok=True)
            subset = [r for r in all_rows if r[3] == fold]
            if kind == "hybrid":
                held = sorted(f for f in fabrics if folds[f] == fold)
                report = {"id": comparison.RUN_ID, "initialization": "synthetic",
                          "fold": fold, "epochs": 6, "batch_size": 16,
                          "image_size": 224,
                          "dataset_md5": base.EXT_MD5,
                          "synthetic_checkpoint_sha256": base.SYNTH_SHA,
                          "source_commit": "synthetic-test-commit",
                          "train_fabrics": sorted(set(fabrics) - set(held)),
                          "heldout_fabrics": held, "heldout_samples": len(subset)}
                (p / "fabric_edge_report.json").write_text(json.dumps(report))
                name = "fabric_edge_oof_predictions.csv"
            else:
                name = "fabric_heldout_oof_predictions.csv"
            with (p / name).open("w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["source_file", "fabric", "actual_class", "fold", "probability_broken"])
                for key, fabric, label, assigned in subset:
                    w.writerow([key, fabric, label, assigned,
                                0.9 if label == "broken" else 0.1])
    return all_rows


def test_paired_aggregator_happy_path(tmp_path):
    fake_fold_sets(tmp_path / "input")
    report = comparison.compare(tmp_path / "input", tmp_path / "output")
    assert report["n_images"] == 3637 and report["n_fabric_groups"] == 11
    assert report["models"]["hybrid"]["macro_f1"] == pytest.approx(1)
    assert report["delta_hybrid_minus_baseline_macro_f1"] == pytest.approx(0)
    assert (tmp_path / "output" / "edge_hybrid_paired_comparison.json").exists()


def test_paired_aggregator_fails_on_missing_record(tmp_path):
    fake_fold_sets(tmp_path)
    name = tmp_path / "hybrid" / "0" / "fabric_edge_oof_predictions.csv"
    rows = name.read_text().splitlines()
    name.write_text("\n".join(rows[:-1]) + "\n")
    with pytest.raises(ValueError, match="STOP"):
        comparison.compare(tmp_path, tmp_path / "output")


def test_paired_aggregator_fails_on_false_pairing(tmp_path):
    fake_fold_sets(tmp_path)
    name = tmp_path / "hybrid" / "0" / "fabric_edge_oof_predictions.csv"
    rows = name.read_text().splitlines()
    fields = rows[1].split(",")
    fields[2] = "normal" if fields[2] == "broken" else "broken"
    rows[1] = ",".join(fields)
    name.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="STOP"):
        comparison.compare(tmp_path, tmp_path / "output")
