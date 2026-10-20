# Enhanced MobileNetV3: 224×224, 12-epoch grouped development CV (20 October 2026)

## Actual executed results

An ImageNet-pretrained MobileNetV3-Small with trainable last convolutional tail and binary classifier head was fitted on the cleaned v2 dataset's 26 **filename-inferred** photo parents, using 5-grouped-fold OOF predictions for all 6,278 development images. **This was actually executed**, with final refit weights saved separately. The earlier inspected five-parent test group and StitchingNet were **not accessed during training or grouped-CV model comparison**.

| Recipe, exactly matched development folds and sample selection | Pooled macro-F1 | Balanced accuracy | Faulty recall |
|---|---:|---:|---:|
| Earlier class_only MobileNetV3: 128px, 2 epochs | 0.7434739 | 0.7525949 | 0.7587598 |
| New 224px, 2 epochs | 0.7559726 | 0.7675781 | 0.7580035 |
| **New 224px, 12 epochs** | **0.7799454** | **0.7816663** | **0.8285858** |

Observed macro-F1 difference 224px epoch 2 minus prior 128px epoch 2: **+0.01250**. The fixed 224px epoch 12 minus 224px epoch 2: **+0.02397**. Entire longer, higher-resolution recipe vs prior baseline: **+0.03647**, or **+3.65 percentage points**, with **+6.98 pp faulty recall**.

**Uncertainty:** Paired 2,000-draw bootstrap resampling 26 inferred photo parents gave a *descriptive* 95% interval for macro-F1 improvement of **[-0.00859, +0.07897]**; this includes zero. Fold-level heterogeneity was substantial and the provenance is not fully human verified. Improved point estimates are not guaranteed to generalize to genuinely new physical source photos.

## Matched actual training protocol

- Same 5 StratifiedGroupKFold partitions, same fixed RNG seeds, same global SHA-stratified selection of **800 actual + 800 faulty per training fold**, same baseline class-only BCE weighting and frozen ImageNet initialization. All CV predictions are from training-fold-excluded source groups.
- Trainable MobileNetV3 final convolutional tail and trained binary head; AdamW head LR 0.001, convolutional tail LR 0.0002, weight decay 0.0001, batch size 12, **no online augmentation**. The new intervention changed input from 128px to 224px and extended fitting from 2 to 12 epochs; milestones were predeclared, not selected post-hoc by validation curves.
- Final 12-epoch refit used the entire development *pool of candidate parents* but **only 1,600 sampled image records**, not all 6,278. The released final refit is not the average of OOF fold checkpoints.
- Physical source IDs remain *inferred from file lineage*, not authenticated unaugmented camera origins. Original photos were augmented/Photoshop-derived; this experiment alone does not measure out-of-factory generalization.

## Verified artifacts

- [Successful real six-job grouped CV and final refit](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39449510789)
- [Successful paired OOF comparison and descriptive source-cluster uncertainty](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39450663852)
- [Detailed machine-readable CV results and paired development predictions](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/enhanced-mobilenet-224e12-v1-comparison)
- [Authentic trained MobileNetV3 224px 12epoch checkpoint](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/enhanced-mobilenet-224e12-v1-refit)
- [Training code](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/code/scripts/enhanced_mobilenet_224_cv.py)
- [New checkpoint inference code](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/code/scripts/predict_enhanced_mobilenet_224.py)
- [Previously inspected StitchingNet external transfer workflow for frozen new checkpoint](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39451285491) — nonblind descriptive transfer only; wait for successful final release for the exact score.

**No new independent holdout score is claimed.** Although the 26-parent grouped-CV point estimate improved, the parent bootstrap interval contains zero. The dataset has a small number of inferred physical sources, and stronger model training may still overfit altered image artifacts.