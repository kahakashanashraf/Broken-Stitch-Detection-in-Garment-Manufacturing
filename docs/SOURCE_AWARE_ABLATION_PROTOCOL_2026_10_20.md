# Controlled source-aware training ablation — broken-stitch research

**Question:** Under the same photo-parent-group-disjoint evaluation protocol, does additional source-aware loss reweighting improve broken-stitch detection compared with ordinary class-balanced fine-tuning? Does train-only augmentation add improvement?

## Why the earlier proposal needed correction

A fixed **40 images per source/class** sampling cap was already a source-balancing intervention. As most Photoshop-faulty parent groups have over 110 images, the sampled class counts per group would become nearly equal, potentially making per-image and per-parent loss weights numerically identical. Such a comparison could **not** test the stated hypothesis. The corrected experiment samples **800 images from each class globally (1,600/fold) without per-parent balancing**, using deterministic SHA-derived ordering. All arms use precisely those same selected training examples for each fold.

## Locked interventions

| Arm | Training image selection | Class balancing | Extra per-parent balancing | Additional online image augmentation |
|---|---|---|---|---|
| `class_only` (A) | Identical global 800+800 | Yes | No | No |
| `parent_class` (B) | Identical global 800+800 | Yes | Yes | No |
| `parent_class_aug` (C) | Identical global 800+800 | Yes | Yes | Train only, ±8° rotation and ±8% brightness/contrast jitter |

**Paired A–B** isolates incremental photo-parent/class loss reweighting relative to class-only weight normalization in the *same* globally class-stratified 1,600-example training budget. **Paired B–C** isolates additional online augmentation. It does *not* test the effect of source-balancing **sampling** or independently verified photos.

Shared model and training details: torchvision ImageNet1K pretrained MobileNetV3-Small, 128×128 RGB, fixed final convolutional tail trainable, ImageNet-normalized, sigmoid binary linear head, AdamW with head LR `1e-3`, convolutional tail `2e-4`, weight decay `1e-4`, fixed 2 epochs, batch 12, fixed threshold 0.5. Batch shuffle has an independent fixed torch random generator, so augmentation cannot change example order across arms.

### Data and fair measurement

- Cleaned dataset v2 signed original files; only `train.zip`, `val.zip`, and `clean_manifest.csv` downloaded. Verify SHA-256 and full ZIP CRCs.
- **26 inferred photographed-parent candidates** in the combined development pool (6,278 historical augmented records). The **5 previously inspected holdout candidate parents and all StitchingNet data are excluded**. No new independent test result.
- Same fixed `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261019)` partitions in all arms. No candidate-parent crosses folds.
- All OOF predictions per arm are released; identical paths, labels, folds, and fold-level selected-image hashes required by the automatic comparison job.
- Report pooled OOF macro-F1, balanced accuracy, faulty recall, ROC-AUC, and paired candidate-parent-cluster bootstrap (2,000 draws) for B–A and C–B. Descriptive bootstrap intervals only, no confirmatory p-values or unqualified industrial performance claims.
- Final checkpoint for each arm refits on development samples only; these weights are **not** the model that generated out-of-fold metrics.

### Scientific limitations and decision rule

The original raw photographs and Photoshop edit histories have **not** been independently recovered: 31 source-root IDs are inferred from filenames and visual matches; group-independent evaluation is not independently certified. Prior models and heldout results informed this research question, so the present ablation is **exploratory/post-hoc**. All images include correlated historical augmented/Photoshop-derived samples. There are only 26 candidate source units in CV. Prior five-parent test and StitchingNet were inspected and cannot be used for another untouched test after model selection.

Interpretation rule: a positive B–A pooled difference with a paired *descriptive* interval that does not cross zero supports **development-pool effectiveness of additional inferred-source loss reweighting under the locked protocol**, but not an industrial generalization claim. If B fails, report it as a negative finding rather than selectively replacing the test with a better-looking split. Report C–B separately; do not conflate augmentation and source reweighting.

### Artifacts

- Source: `code/scripts/source_balancing_ablation.py`
- Paired evaluation: `code/scripts/compare_source_balancing_ablation.py`
- GitHub Actions workflow: `.github/workflows/ablate-source-balancing.yml`
- Individual model reports, true trained state dictionaries and OOF predictions: planned distinct immutable GitHub prereleases per arm
- Matched A/B/C comparison: planned `source-balanced-ablation-comparison-v1` prerelease if all arms succeed