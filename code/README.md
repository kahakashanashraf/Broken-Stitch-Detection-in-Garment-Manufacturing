# Broken Stitch Detection — auditable code reconstruction (ICEEICT 2027)

**Status: runnable reconstruction, NOT an exact recovery of the original 2022 code or results.**
The old source files and checkpoints were not provided. This is a **new PyTorch implementation**
faithful to the *experiment families* described in the thesis; the original used TensorFlow/Keras.
The precise architecture, preprocessing strengths, augmentation order, fine-tuned layers,
ensemble specification, optimizer schedule, sampling and original splits are not recoverable.
Differences are declared below and in code, not disguised as historical settings.

## What's implemented

- Image inventory and class-folder detection (`actual/correct/normal`, `faulty/wrong/broken/defect`).
- Mandatory **source_id** for every derived image: normal photograph and its Photoshopped/augmented derivatives share ONE parent group even when labels differ.
- Group-exclusive train/validation/test partition search, with both classes in every partition.
- Verifier for group crossing and cross-partition SHA-256 duplicate files.
- CNN approximate baseline (three 3×3 Conv blocks, 2×2 max-pooling, batch norm, 25% dropout, 2-way logits); six historical-style activation alternatives.
- Filtering: none, Gaussian, bilateral, and **median**. Historical "salt and pepper filter" is ambiguous: median filtering here is explicitly an *approximation* of salt-and-pepper noise removal, not a verified reproduction.
- ImageNet VGG-16 frozen feature extractor + linear head, final-block VGG-16 fine-tuning + MLP head.
- Average probability ensemble of independently trained VGG-linear heads (ensemble size and design are **new declared choices**).
- Weighted cross-entropy (a newly selected mitigation for imbalance), Adam optimizer, early stopping on **validation macro-F1**.
- Held-out test evaluation: accuracy, balanced accuracy, macro-F1, class precision/recall/F1, ROC-AUC, average precision, confusion matrix.
- Run configs, checkpoint metadata, dataset partition fingerprint, history, smoke/unit tests, optional figure generator.

**NO scientific results were generated from the user's dataset.** A provided `dataset.zip` was ~2.15 GB and the Drive connector has a 256 MiB download limit; the original images could not be inspected here. All execution checks used **tiny locally generated dummy images**. Nothing from that test may go into the conference paper.

## Setup

Use Python 3.10–3.12 in a new environment. Install matching PyTorch/torchvision builds first (GPU-specific packages from the official PyTorch installation selector), then:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For an offline Kaggle run, attach the extracted dataset and a locally available **official ImageNet-1K VGG-16 PyTorch weights file**. The expected `--vgg-weights` is the unmodified torchvision VGG16 state_dict `.pth`, or a `.pth` dictionary containing `state_dict`. **Never label random weights as pretrained.** Alternatively `--allow-download` works only with internet during training and may require >500 MB free disk. `--allow-random-vgg` is a smoke-test-only option and marks the result `model_weights_provenance_ok=false`.

## 1. Unzip original data to a folder

```bash
unzip dataset.zip -d data/raw
python -m broken_stitch.cli inventory --root data/raw --out data/inventory.csv
```

Inspect `data/inventory.csv`: `relative_path,path,label,source_id,is_synthetic,split,width,height,decode_error`.
If class folders do not carry recognizable names, fill `label` manually using **exactly** `actual` or `faulty`.

**STOP until `source_id` values are audited and filled in for every image.** Do not assign one group ID per augmented image. The source ID identifies the initial unique normal/real-fault photograph from which that example was made. If a Photoshop defective image is based on a correct image, both labels share the same source ID. Without parentage, a safe evaluation is not possible on the augmented set.

If filenames genuinely encode source IDs, this helper can *suggest* groups, which must be manually reviewed:

```bash
python scripts/assign_lineage_from_names.py --inventory data/inventory.csv \
   --out data/with_source_ids.csv --regex 'parent(?P<source>[0-9]+)'
```

If the original photographs and their derivatives cannot be matched, do **not** guess; recover the source images or label a new independent dataset. A class folder alone proves neither image independence nor real-vs-Photoshopped provenance. For `is_synthetic`, manually mark `yes/no/unknown`; the script never guesses.

## 2. Group-disjoint split and strict audit

```bash
python -m broken_stitch.cli split --manifest data/with_source_ids.csv \
  --out data/split.csv --seed 42
python -m broken_stitch.cli audit --manifest data/split.csv
```

The split searches for train/val/test partitions approximating 70%/15%/15% by **image count**, while enforcing source-disjointness and both labels in each partition. It is NOT a claim of strict perfect stratification. Prefer source-group cross-validation if only a few independent groups exist. All parents and augmented descendants must share their split. The audit computes exact SHA-256 checks on every file; visually similar rotated/cropped derivatives can still escape detection, so human lineage verification remains essential. The 181 original records in the thesis may not represent 181 distinct camera images.

## 3. Train the baselines

```bash
# CNN baseline at 100×100 pixels
python -m broken_stitch.cli train --manifest data/split.csv \
  --outdir runs/cnn_relu --model cnn --activation relu --epochs 30

# Gaussian filtering ablation (identical filter for train/val/test)
python -m broken_stitch.cli train --manifest data/split.csv \
  --outdir runs/cnn_gaussian --model cnn --filter gaussian --epochs 30

# Six activation experiments, filters and VGG runs: optional full matrix
bash scripts/run_matrix.sh data/split.csv /path/to/vgg16_imagenet_weights.pth

# Frozen ImageNet-pretrained VGG-16 + linear head
python -m broken_stitch.cli train --manifest data/split.csv \
  --outdir runs/vgg_linear --model vgg_linear \
  --vgg-weights /path/to/vgg16_weights.pth --lr 0.0001

# Last VGG block trainable + MLP
python -m broken_stitch.cli train --manifest data/split.csv \
  --outdir runs/vgg_finetune --model vgg_finetune \
  --vgg-weights /path/to/vgg16_weights.pth --lr 0.00001
```

The selected checkpoint is the epoch with best **validation macro-F1**. Do not select model by inspecting held-out test repeatedly. `runs/<name>/best.pt`, `history.json`, `run_config.json` preserve metadata and state. Full reruns need matching hardware/software seeds and exact split file.

## 4. Evaluate only AFTER choosing the best model from validation

```bash
python -m broken_stitch.cli evaluate --manifest data/split.csv \
  --checkpoint runs/vgg_finetune/best.pt --split test --out final_test.json

python -m broken_stitch.cli ensemble --manifest data/split.csv \
  --checkpoints runs/vgg_linear_seed_{13,29,47}/best.pt \
  --split test --out final_ensemble_test.json
```

The tool rejects a manifest whose split or labels differ from those of the training checkpoint. It does **not** prove metadata or source IDs were correctly assigned by a human; the data-governance audit is still required.

To draw new figures **only from actual reconstructed results**:

```python
from broken_stitch.report import plot_from_files
plot_from_files('runs/cnn_relu/history.json','final_test.json','figures_new')
```

## Differences versus thesis and publication guardrails

| Topic | 2022 thesis | Rebuilt project |
|---|---|---|
| Framework | TensorFlow / Keras | PyTorch 2.x |
| Data size | 181 initial; 20,000 / 10,000 / 9,134 inconsistently reported | Determined strictly from inventory CSV |
| Augment then split? | Methods text may imply this, unverified | **Split by parent source, then online train augmentation** |
| Image size | 100×100 in CNN data-loading discussion | CNN 100×100, VGG 224×224 |
| CNN architecture | Not enough layer-by-layer detail | New explicitly defined 3-conv CNN approximation |
| Filters | Gaussian, bilateral, ambiguous salt-pepper | Gaussian, bilateral, median explicitly distinguished |
| VGG model | VGG16 features, one-layer NN, ensemble, multilayer | Frozen feature-linear, last-block fine-tuned MLP, probability ensemble |
| Test metrics | Training, validation / test labels inconsistently used | Held-out test **only after tuning**, full class-wise metrics |
| Reported 94.91% | Historical **validation** accuracy | **Not reproducible claim**; entirely new results only after real-data run |

`examples/HISTORICAL_THESIS_RESULTS_DO_NOT_COMPARE.json` holds thesis numbers for editorial checking only, never as regression-test targets.

## Release gates for the ICEEICT manuscript

**GO to training:** independent source groups identified; class labels and real/synthetic status verified; data rights assessed; `audit` succeeds; no cross-group duplicates; authors approve protocol.

**GO to paper results update:** CPU/GPU training completed on original dataset, split frozen and logged, repeated seeds and per-class metrics reviewed, no contamination of test set. If these gates do not pass, retain the thesis metrics as historical and don't insert fabricated verification claims.

The reconstruction is **not** ready to claim the paper's historic VGG performance, no matter how many smoke tests pass.

## 5. Update for the 2026 source-disjoint + StitchingNet research direction

The conference working draft now proposes an extension; **it does not yet include new results**.
The binary public external benchmark requires a strict subset: **normal vs. broken stitch only**
(published counts: 2,436 and 1,201 images). The other nine defect classes are out of scope.
Dataset source: https://github.com/hyungjungkim/StitchingNet (CC BY-NC 4.0). Do not
redistribute the dataset with this code; check license terms and original files.

After downloading and extracting StitchingNet **yourself**, manually locate verified
folders consisting exclusively of normal and broken stitch samples. The CLI does NOT
automatically infer class labels from arbitrary filenames or folders:

```bash
python -m broken_stitch.cli external-manifest \
  --normal-dir /path/to/StitchingNet/verified_normal_images \
  --broken-dir /path/to/StitchingNet/verified_broken_stitch_images \
  --out data/stitchingnet_binary_external.csv

# Train and freeze model using original source-disjoint THESIS manifest FIRST.
python -m broken_stitch.cli external-evaluate \
  --checkpoint runs/vgg_finetune/best.pt \
  --internal-manifest data/split.csv \
  --external-manifest data/stitchingnet_binary_external.csv \
  --out results/stitchingnet_external_frozen.json
```

The external test uses the original checkpoint/threshold (0.5) with no adaptation.
It rejects file hash collisions against the full internal manifest, and refuses changed
internal split fingerprints. It does NOT resolve all near duplicates, fabric-domain
confounding, or uncertain stitch source IDs. These require human audit. **Do not** tune on
StitchingNet before claiming it is an unseen external test. All output includes
`publication_ready: false` intentionally. Higher metrics do not auto-clear publication gates.

Planned optional lightweight baseline (MobileNetV3/EfficientNet-B0) and random-split
comparison are **not implemented in v2** and must not be claimed as completed.
Do not confuse software tests or historical thesis numbers with new scientific data.