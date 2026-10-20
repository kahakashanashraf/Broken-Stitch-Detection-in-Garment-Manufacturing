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
- [Successful frozen StitchingNet external transfer for enhanced checkpoint](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39451285491) — nonblind descriptive transfer only; [exact released metrics and per-image predictions](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/enhanced-224e12-stitchingnet-descriptive-v1).

**No new independent holdout score is claimed.** Although the 26-parent grouped-CV point estimate improved, the parent bootstrap interval contains zero. The dataset has a small number of inferred physical sources, and stronger model training may still overfit altered image artifacts.
## New checkpoint: actual StitchingNet domain transfer

The frozen development-trained 224px/12epoch refit was evaluated **read-only** on StitchingNet normal class 00 (n=2,436) and broken class 02 (n=1,201), 3,637 images, across 11 fabric groups. Official archive MD5 was checked; the checkpoint digest was locked. The classification threshold stayed **0.5**, without calibrating, fitting, sampling or selecting by StitchingNet labels. **StitchingNet had already been examined earlier in the project**, so this is **not** fresh, blind or untouched model generalization.

| Metric | Previous class-only MobileNetV3 128px/2epochs | Enhanced MobileNetV3 224px/12epochs |
|---|---:|---:|
| Accuracy | 0.753368 | 0.745395 |
| Macro-F1 | 0.721634 | **0.725330** |
| Balanced accuracy | 0.721963 | **0.738805** |
| Broken recall | 0.629475 | **0.719400** |
| ROC-AUC | 0.776860 | **0.821453** |
| Missed broken stitches out of 1,201 | 445 | **337** |
| False broken alarms out of 2,436 normal | **452** | 589 |

Enhanced confusion matrix (rows true normal/broken; columns predicted normal/broken): **[[1847, 589], [337, 864]]**. The enhanced model misses 108 fewer broken stitches at the cost of 137 additional false alarms. Compared with the source-aware+online augmentation model's 70.77% broken recall, enhanced recall is 71.94% under the same fixed 0.5 threshold.

**Interpretation:** Grouped-development macro-F1 increased +3.65 percentage points, but previously inspected external StitchingNet macro-F1 increased only **+0.37 percentage points**, while broken recall improved **+8.99 points** and normal false alarm burden rose. This is a meaningful operating-point tradeoff, not proof of superiority in new factories. No paired inferential test is asserted for the small cross-dataset macro-F1 change. Do not tune a threshold on these already-inspected external labels and then call the model an independent test winner.

- [Full verified read-only external workflow](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39451285491)
- [External metrics, fabric breakdown and all 3,637 predictions](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/enhanced-224e12-stitchingnet-descriptive-v1)
- [Earlier six-model descriptive StitchingNet comparison](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/docs/STITCHINGNET_FINE_TUNED_EXTERNAL_2026_10_20.md)
