# Real-data adaptation and XAI: stitching defects across unseen fabric types

**Evidence frozen 20 October 2026** · Experimental project, **NOT industrial deployment cleared**.

## Research question

A legacy dataset contains 7,364 readable synthetic/augmented images from 31 **filename-inferred** photographic roots. Our thesis-only 224×224, 12-epoch MobileNetV3 had **0.77995 grouped development OOF macro-F1** (26 inferred source roots), but only **0.72533 macro-F1** and **0.71940 broken recall** in a descriptive 3,637-image transfer to StitchingNet.

We tested whether training on actual production-type sewing defects helps, **without allowing the same fabric type to appear on both sides of the training/evaluation fold**. This is a DIFFERENT scientific protocol: once labeled StitchingNet images are used to train a model, its performance is **not** the thesis-only zero-shot transfer result. StitchingNet was already inspected in previous project experiments; even fabric-disjoint CV here is retrospective/exploratory rather than fresh blind confirmation.

## Dataset and training

- Public real sewing-condition dataset: **StitchingNet**, source https://zenodo.org/records/20352813 and https://github.com/hyungjungkim/StitchingNet, official authors' declared license **CC BY-NC 4.0**.
- Official ZIP MD5: \`51b228d9572b925035e9e9af01c96569\`.
- Binary study: class 00 normal 2,436, class 02 broken stitch 1,201, **3,637 records** from **11 distinct fabric/thread configurations**, A–K. All nine other defect classes excluded.
- Identical **five StratifiedGroupKFold** folds by complete fabric group, same fold partitions for both models, 224×224 resize, six epochs, batch 16, class-weighted binary loss, train-only small brightness/contrast/rotation augmentation, and a last-convolutional-stage trainable MobileNetV3 head.
- Arms differ ONLY in pretrained initialization: **ImageNet** versus **ImageNet → synthetic thesis-derived training** (frozen SHA-256 \`0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81\`).
- Each outer fold sees labeled StitchingNet training fabrics but NEVER its held-out fabric IDs during fitting. All model threshold=0.5 for the PRIMARY comparisons. All five folds for both arms executed and released with source IDs, predictions and attention heatmaps.
- Limit: machine, imaging conditions and stitching process may still correlate across fabrics. The 11 fabric groups are not 11 unrelated datasets.

## Actual pooled out-of-fold results (fixed 0.5 threshold)

| Method / training provenance | Macro-F1 | Balanced accuracy | Accuracy | Broken recall | Broken misses / 1,201 | Normal false alarms / 2,436 |
|---|---:|---:|---:|---:|---:|---:|
| Thesis-only 224×224 MobileNetV3, **no StitchingNet training**, old nonblind transfer | 0.72533 | 0.73881 | 0.74539 | 0.71940 | 337 | 589 |
| ImageNet initialized **and trained on StitchingNet** other fabrics, grouped OOF | 0.79690 | 0.83613 | 0.80423 | **0.93006** | **84** | 628 |
| ImageNet→thesis-pretrained **and trained on StitchingNet** other fabrics, grouped OOF | **0.80892** | **0.84197** | **0.81771** | 0.91341 | 104 | **559** |

***Do not compare the three rows as identical models on the same independent test protocol.*** The old reference did not fit to StitchingNet; the new models were intentionally fitted on other StitchingNet fabrics. We have not magically improved zero-shot performance to 0.80892.

ImageNet vs thesis-synthetic initialization *within the matched new protocol* differs by **+0.01202 macro-F1**. Paired 11-fabric-cluster 2000-draw descriptive bootstrap difference interval is **[−0.00901, +0.03599]**. It includes zero: an incremental synthetic-pretraining advantage **is not established**.

Fabric-held-out fold macro-F1 varies widely:

| Entirely held-out fabrics | ImageNet | Thesis-synthetic init |
|---|---:|---:|
| D/E | 0.75384 | 0.78064 |
| I/J | 0.68665 | 0.69312 |
| B/F | 0.96959 | 0.96341 |
| C/G/K | 0.71661 | 0.73857 |
| A/H | 0.90538 | 0.91257 |

**Conclusion:** Some held-out fabric groups achieve ~0.96–0.97 macro-F1; others are around 0.69. Fabric robustness and false alarms remain open problems even after labeled target-domain adaptation. Do not claim state-of-the-art industrial performance.

## XAI: actual Grad-CAM (not a stock/artificial explanation)

Each of the **ten group-fold model Releases** includes gradient-derived 224×224 **Grad-CAM grayscale heatmaps** for a selected broken-stitch true positive and false negative. These reflect gradient sensitivity of the **broken-class logit** through the pretrained MobileNet convolutional feature layer. Reports bind every heatmap to its exact image path, actual class, fabric code and predicted broken probability.

What they establish: real code-level differentiable saliency computed from the actual fitted fold model. What they **do not** establish: localization accuracy, causal model explanations, absence of spurious texture/fabric cues, or reliability of attention on an independent dataset. No original StitchingNet image pixels are republished in heatmap assets. The original dataset can be acquired from its license-bearing source.

Example XAI artifacts:
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-fabric-heldout-synthetic-f3-v1
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-fabric-heldout-imagenet-f1-v1

## Posthoc train-fabric prior operating point (separate exploratory analysis)

The training class-balanced loss assumes equal class weights; the **train-fabric** observed broken prevalence is ≈33%, so a *prespecified formula*, applied after observing initial results, tests threshold \`t = normal_count(train fabrics)/all_count(train fabrics)\` (approximately 0.665–0.673 per fold). The numerical threshold is based **only on each training fold class counts**, never its held-out labels. **However, this analysis idea was proposed post-hoc** after examining results, so the adjusted score is exploratory, not independent model confirmation:

| Model | Original grouped OOF macro-F1 | Train-prior-adjusted grouped OOF macro-F1 | Adjusted broken recall |
|---|---:|---:|---:|
| ImageNet init | 0.79690 | 0.80718 | 0.92007 |
| Thesis-synthetic init | 0.80892 | **0.81767** | 0.90425 |

Threshold correction changes the decision rule, **not model weights**. The better adjusted score is not an unbiased estimate of a new confirmatory experiment. No operating-point tuning was done on a genuinely independent untouched source.

Release https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-train-prior-adjustment-v1

## Data, weights and full reproducibility

- Ten actual successful fit jobs: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39454366380
- Five-fold paired model comparison, successful: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39456509244
- Immutable complete paired comparison: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-fabric-adapt-comparison-v1
- Unit-test CI verifies real Grad-CAM gradients, complete fabric fold exclusivity, score validation and fail-closed refit provenance.
- Training/evaluation: \`code/scripts/fabric_group_adaptation.py\`; paired CV comparison: \`code/scripts/compare_fabric_adaptation.py\`; posthoc threshold check: \`code/scripts/fabric_prior_adjustment.py\`.
- Model selection and full-data refit are separate: \`code/scripts/refit_stitchingnet_fabric.py\`; \`code/scripts/predict_refit_fabric.py\`. **The final full-data refit has no untouched test accuracy**, and its expected release is https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-fabric-adapt-selected-refit-v1 after workflow success.

## Responsible research positioning

> **Confirmed case-specific findings:** Original augmentation-family split leakage exists; a source-grouped partition fixes observable overlap. Within our explored protocols, more training / resolution improves internal grouped development OOF; direct thesis-only cross-domain generalization remains much weaker. Labeled target-domain adaptation greatly improves broken-stitch recall on fabric-held-out folds but does not eliminate fabric-specific false alarms. The incremental benefit from thesis pretraining over ImageNet alone is **not robustly established** across 11 fabrics.

Potential paper direction: **\"From Synthetic to Real Broken-Stitch Detection: Leakage-Controlled Reconstruction, Fabric-Disjoint Adaptation and Grad-CAM Failure Analysis\"**. This is a case-study/empirical domain-generalization paper, NOT a novel state-of-the-art architecture claim. Independent new fabric/session collection, original synthetic-data rights, project coauthor review, and venue AI-authorship restrictions remain open before any submission.
