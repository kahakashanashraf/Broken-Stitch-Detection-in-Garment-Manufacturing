"""Matched supervised StitchingNet budget experiment: 6 vs 30 epochs.

All folds are the SAME 11-fabric, five-heldout-fold partitions as the immutable
2026-10-20 six-epoch experiment. Only training duration changes. Each run
emits both epoch-6 and epoch-30 OOF probabilities from the *same* initialization
and optimizer trajectory, with no validation-driven epoch selection.

Retrospective/development study: StitchingNet has been inspected previously;
NOT a new blind independent test, zero-shot transfer or factory deployment result.
"""
import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import fabric_group_adaptation as base

RUN_ID = "stitchingnet_fabric_heldout_budget6_vs30_v2"
EPOCHS = 30
MILESTONES = (6, 30)
BATCH_SIZE = base.BATCH
ARMS = base.ARMS
SEED = base.SEED
FOLDS = base.FOLDS
DATA_MD5 = base.EXT_MD5


def validate_args(args):
    if args.epochs != EPOCHS or args.batch_size != BATCH_SIZE:
        raise ValueError("STOP: only the locked 30-epoch, batch-16 protocol is allowed")
    if args.arm not in ARMS or args.fold not in [str(i) for i in range(FOLDS)]:
        raise ValueError("STOP: invalid arm/fabric fold")


def predict_fixed(model, archive, rows, va, device):
    """Eval-only, no thresholds, training gradients or model-selection feedback."""
    ds = base.FabricZip(archive, rows, va, training=False)
    try:
        probs, labels, indices = [], [], []
        model.eval()
        with torch.inference_mode():
            for x, y, idx in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0):
                probs.extend(torch.sigmoid(model(x.to(device))).cpu().tolist())
                labels.extend(y.to(torch.int32).tolist())
                indices.extend(int(i) for i in idx.tolist())
        expected = [int(i) for i in va]
        if indices != expected or labels != [rows[i][1] for i in expected]:
            raise ValueError("STOP: heldout image/label alignment mismatch")
        if not len(probs) or not np.isfinite(probs).all() or any(not 0 <= p <= 1 for p in probs):
            raise ValueError("STOP: invalid probabilities")
        return probs, base.scores(labels, probs)
    finally:
        ds.close()


def run_fold(args):
    validate_args(args)
    rows = base.extract_rows(args.stitchingnet_zip)
    folds, assignments = base.grouped_folds(rows)
    fold = int(args.fold)
    tr, va, train_fabrics, heldout_fabrics = folds[fold]
    if set(train_fabrics) & set(heldout_fabrics):
        raise ValueError("STOP: train/heldout fabric overlap")
    if any(assignments[i] != fold for i in va):
        raise ValueError("STOP: inconsistent OOF assignments")

    # Same seeds, model, optimizer, transforms, training rows and loader
    # ordering as the six-epoch baseline in fabric_group_adaptation.py.
    seed = SEED + 100 * fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = base.build_model(args.arm, args.synthetic_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 5e-4},
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 1e-4},
    ], weight_decay=1e-4)

    counts = Counter(rows[int(i)][1] for i in tr)
    if set(counts) != {0, 1}:
        raise ValueError("STOP: training class missing")
    weights = {label: len(tr) / (2 * n) for label, n in counts.items()}
    ds = base.FabricZip(args.stitchingnet_zip, rows, tr, training=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 999))
    milestones = {}
    history = []
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train(True)
            loss_total = 0.0
            for x, y, _ in loader:
                x, y = x.to(device), y.to(device)
                weight = torch.where(y == 0,
                                     torch.full_like(y, weights[0]),
                                     torch.full_like(y, weights[1]))
                logits = model(x)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none")
                        * weight).sum() / weight.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.item()) * len(y)
            history.append({"epoch": epoch, "train_loss_mean": loss_total / len(tr)})
            print("BUDGET_EPOCH", json.dumps({"arm": args.arm, "fold": fold,
                  "epoch": epoch, "loss": history[-1]["train_loss_mean"]}), flush=True)
            if epoch in MILESTONES:
                probs, metric = predict_fixed(model, args.stitchingnet_zip, rows, va, device)
                milestones[epoch] = {"probabilities": probs, "metrics": metric}
                print("BUDGET_MILESTONE", json.dumps({"arm": args.arm, "fold": fold,
                      "epoch": epoch, "metrics": metric}), flush=True)
    finally:
        ds.close()
    if set(milestones) != set(MILESTONES):
        raise ValueError("STOP: scheduled milestones incomplete")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "budget_heldout_oof.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["source_file", "fabric", "actual_class", "fold",
                         "probability_broken_epoch6", "probability_broken_epoch30"])
        for j, i in enumerate(va):
            name, label, fabric = rows[int(i)]
            writer.writerow([name, fabric, "broken" if label else "normal", fold,
                             milestones[6]["probabilities"][j],
                             milestones[30]["probabilities"][j]])
    report = {
        "experiment": RUN_ID, "status": "EXPLORATORY_ALREADY_INSPECTED_STITCHINGNET",
        "source_commit": os.environ.get("GITHUB_SHA", "local_unverified"),
        "arm": args.arm, "fold": fold, "epochs": EPOCHS, "milestones": list(MILESTONES),
        "batch_size": BATCH_SIZE, "resolution": base.SIZE,
        "data_zip_md5": DATA_MD5, "source_checkpoint_sha256":
            base.SYNTH_SHA if args.arm == "synthetic" else None,
        "seed": seed, "training_fabrics": train_fabrics,
        "heldout_fabrics": heldout_fabrics, "train_samples": len(tr),
        "heldout_samples": len(va), "train_label_counts": {str(k): v for k, v in counts.items()},
        "optimizer": "AdamW", "head_lr": 5e-4, "tail_lr": 1e-4,
        "weight_decay": 1e-4, "augmentation": "rotation8,brightness8%,contrast8% train-only",
        "fine_tuning": "same MobileNetV3 final convolutional tail + head as baseline",
        "epoch_loss": history,
        "milestone_metrics": {str(e): milestones[e]["metrics"] for e in MILESTONES},
        "validation_use": "read-only milestones, never update optimizer or select checkpoint",
        "caveat": "StitchingNet was already inspected; same 11 fabric groups, non-blind exploratory OOF. Groups may share capture sessions. No novel architecture or independent test."
    }
    (out / "budget_heldout_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitchingnet-zip", required=True)
    ap.add_argument("--synthetic-weights", required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--fold", choices=[str(i) for i in range(FOLDS)], required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--out", required=True)
    run_fold(ap.parse_args())
