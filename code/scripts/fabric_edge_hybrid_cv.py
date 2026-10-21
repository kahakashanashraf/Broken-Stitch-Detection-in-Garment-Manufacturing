"""Experimental MobileNetV3 + explicit Sobel/texture branch on unseen StitchingNet fabrics.

Only ONE intervention relative to existing synthetic-initialized 6-epoch fabric CV:
add a small trainable edge branch and zero-initialized residual logit gate.
Same training data, five fabric-disjoint folds, seed, fixed 0.5 threshold,
weighted BCE, transformations, batch size, optimizer for shared parameters.

This is a descriptive, non-blind architecture ablation on previously inspected
StitchingNet data, NOT a novel detector, independent evaluation or segmentation.
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import fabric_group_adaptation as base

RUN_ID = "stitchingnet_synthetic_mobilenet_sobel_fusion_v1"
EPOCHS = base.EPOCHS
BATCH = base.BATCH
FOLDS = base.FOLDS
ARM = "synthetic"
MODEL = "mobilenetv3_small_sobel_residual"
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).reshape(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).reshape(1, 1, 3, 3)


class EdgeFusionMobileNet(nn.Module):
    """Fixed Sobel high-frequency map + learned 24-D edge feature; gated logit fusion.

    The gate is zero at initialization, so initial predictions EXACTLY equal the
    original pretrained binary MobileNet's predictions. Training can learn when
    edge information helps or hurts. It is not a pixel localization model.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Preserve global CPU RNG after original backbone creation so the
        # baseline's subsequent train-only random augmentations stay comparable.
        with torch.random.fork_rng(devices=[]):
            self.edge_features = nn.Sequential(
                nn.Conv2d(1, 12, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(12, 24, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.edge_head = nn.Linear(24, 1)
        self.edge_gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("mean", torch.tensor((.485, .456, .406)).reshape(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", torch.tensor((.229, .224, .225)).reshape(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("sobel_x", SOBEL_X.clone(), persistent=False)
        self.register_buffer("sobel_y", SOBEL_Y.clone(), persistent=False)

    def forward(self, x):
        primary = self.backbone(x)
        # Existing dataset preprocessing applies ImageNet channel normalization.
        rgb = x * self.std + self.mean
        gray = (.2989 * rgb[:, 0:1] + .5870 * rgb[:, 1:2]
                + .1140 * rgb[:, 2:3])
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        # No recomputed fold-wide normalization (which could leak statistics).
        edge = torch.sqrt(gx.square() + gy.square() + 1e-6) / 4.0
        edge_score = self.edge_head(self.edge_features(edge)).flatten()
        return primary + self.edge_gate.tanh() * edge_score


def make_model(synthetic_weights):
    return EdgeFusionMobileNet(base.build_model(ARM, synthetic_weights))


def validate_args(args):
    if args.epochs != EPOCHS or args.batch_size != BATCH or str(args.fold) not in tuple(str(i) for i in range(FOLDS)):
        raise ValueError("STOP: only exact matched six-epoch supervised fabric protocol permitted")


def heldout_predictions(model, archive, rows, indices, device):
    ds = base.FabricZip(archive, rows, indices, training=False)
    probabilities, labels, names = [], [], []
    try:
        model.eval()
        with torch.inference_mode():
            for images, yy, ii in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0):
                probabilities.extend(torch.sigmoid(model(images.to(device))).cpu().tolist())
                labels.extend(yy.int().tolist())
                names.extend(ii.tolist())
    finally:
        ds.close()
    if names != [int(i) for i in indices] or labels != [rows[int(i)][1] for i in indices]:
        raise ValueError("STOP: OOF alignment mismatch")
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
        raise ValueError("STOP: invalid heldout probabilities")
    return probabilities, labels


def run_fold(args):
    validate_args(args)
    rows = base.extract_rows(args.stitchingnet_zip)
    folds, assignments = base.grouped_folds(rows)
    k = int(args.fold)
    tr, va, training_fabrics, heldout_fabrics = folds[k]
    if set(training_fabrics) & set(heldout_fabrics) or any(assignments[i] != k for i in va):
        raise ValueError("STOP: fabric identity leakage")
    seed = base.SEED + 100 * k
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = make_model(args.synthetic_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    # Original head/tail optimizer groups are the same as in base.fit_one.
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.head.parameters(), "lr": 5e-4},
        {"params": [p for p in model.backbone.features.parameters() if p.requires_grad],
         "lr": 1e-4},
        {"params": list(model.edge_features.parameters())
                   + list(model.edge_head.parameters()) + [model.edge_gate], "lr": 5e-4},
    ], weight_decay=1e-4)
    counter = Counter(rows[int(i)][1] for i in tr)
    if set(counter) != {0, 1}:
        raise ValueError("STOP: both training classes required")
    class_weights = {label: len(tr) / (2 * n) for label, n in counter.items()}
    ds = base.FabricZip(args.stitchingnet_zip, rows, tr, training=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 999))
    history = []
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train(True)
            total = 0.0
            for images, yy, _ in loader:
                images, yy = images.to(device), yy.to(device)
                ww = torch.where(yy == 0, torch.full_like(yy, class_weights[0]),
                                 torch.full_like(yy, class_weights[1]))
                logits = model(images)
                loss = (F.binary_cross_entropy_with_logits(logits, yy, reduction="none")
                        * ww).sum() / ww.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(yy)
            hist = {"epoch": epoch, "train_loss_mean": total / len(tr),
                    "edge_gate": float(model.edge_gate.detach().cpu())}
            history.append(hist)
            print("EDGE_FUSION_EPOCH", json.dumps({"fold": k, **hist}), flush=True)
    finally:
        ds.close()
    probabilities, labels = heldout_predictions(model, args.stitchingnet_zip, rows, va, device)
    metrics = base.scores(labels, probabilities)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fabric_edge_oof_predictions.csv").open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["source_file", "fabric", "actual_class", "fold", "probability_broken"])
        for i, probability in zip(va, probabilities):
            name, lab, fab = rows[int(i)]
            w.writerow([name, fab, "broken" if lab else "normal", k, float(probability)])
    report = {
        "id": RUN_ID, "model": MODEL, "initialization": ARM,
        "source_commit": os.environ.get("GITHUB_SHA", "local_unverified"),
        "fold": k, "epochs": EPOCHS, "batch_size": BATCH, "image_size": base.SIZE,
        "dataset_md5": base.EXT_MD5, "synthetic_checkpoint_sha256": base.SYNTH_SHA,
        "train_fabrics": training_fabrics, "heldout_fabrics": heldout_fabrics,
        "train_samples": len(tr), "heldout_samples": len(va),
        "train_class_counts": {str(label): n for label, n in counter.items()},
        "seed": seed, "training_history": history, "heldout_metrics": metrics,
        "residual_gate_init": 0.0, "threshold": .5,
        "comparison_baseline_release": f"stitchingnet-fabric-heldout-synthetic-f{k}-v1",
        "caveat": "Already-inspected StitchingNet, fabric-heldout descriptive CV only. The edge branch is not a validated localization mask; training on other StitchingNet fabrics is not zero-shot external testing."
    }
    (out / "fabric_edge_report.json").write_text(json.dumps(report, indent=2))
    print("EDGE_FUSION_FOLD_FINAL", json.dumps({"fold": k, "metrics": metrics}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stitchingnet-zip", required=True)
    p.add_argument("--synthetic-weights", required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--out", required=True)
    run_fold(p.parse_args())
