# 2026-10-18 — Broken Stitch Dataset Lineage, Pretrained Baseline and First External Evaluation

## Summary and scientific caution

The team has completed (1) computational lineage reconstruction from source image filenames and visual candidate matching, (2) one new pretrained MobileNetV3-Small baseline fit **without internal test or external labels**, and (3) a **one-shot external evaluation** on the publisher's StitchingNet normal/broken images. The experiments are auditable. **The true photographed-source lineage is not yet human-verified**, and source-family splits used for pretrained training were based on the earlier 182 UUID-stripped groups, not the newly inferred 31 source photographs. Do not claim publication-ready independent-source accuracy or real-world defect detection.

## Original thesis archive, label quality and leakage

- GitHub source asset: 2,146,784,353-byte `dataset.zip`.
- 7,365 image records (2,734 actual/normal and 4,631 faulty), 400×400; 7,364 decoded; one unreadable actual image.
- Earlier original train/val/test directories leaked all **182/182** UUID-stripped augmented filename families across splits. SHA-256 exact duplicates across folders were not found, but such checks miss transformed derivatives.
- Filename-to-photo audit showed **31 `actual_original_XX` families and 151 `faulty_original_XXN` families**. All 151 faulty families map syntactically to one of the 31 normal families. Visual comparison ranked a number of corresponding cross-class pairs extremely highly (e.g. AQ to AQ1).
- This supports an *inferred* common photographed-source graph with ~31 sources. It does NOT establish that every Photoshop modification was derived from the corresponding actual image: original pre-augmentation photographs and editor provenance were not independently examined. The 181 historical / 182 indexed variant-family count also needs reconciliation.
- A new **proposed** candidate-photo-parent split uses 21/5/5 parent IDs for train/validation/test, and contains all 7,365 file records pending removal of the unreadable PNG. **No model weights have been trained on this new grouping in this run.**

Artifacts: [Original ZIP inventory](../audits/original_dataset_inventory_2026_10_18.json), [original split collision audit](../audits/candidate_parent_overlap_2026_10_18.json), [visually ranked lineage candidates](../audits/photo_lineage_review/), [inferred photo-parent graph and portable splits](../audits/photographed_source_inference/).

## Pretrained baseline: fixed, no external adaptation

- Model family: **ImageNet1K-V1 pretrained MobileNetV3-Small convolutional backbone, frozen features**.
- Head: `StandardScaler` and `LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', random_state=42)`. Input 224×224 RGB, ImageNet normalization; decision threshold fixed at 0.5.
- Training: 5,167 actual-archive images from the **earlier 182 candidate filename-family train partition**; validation: 1,120 from candidate-family validation; 1,077 internal test images were **not accessed** by the MobileNet training workflow.
- No multiple model/backbone hyperparameter searches were made in this workflow, and external StitchingNet labels were not used to fit or select anything.
- Validation metrics: accuracy **0.698214**, balanced accuracy **0.643818**, macro-F1 **0.648282**, normal recall **0.433735**, faulty recall **0.853901**, faulty AUROC **0.694742**.
- Public frozen checkpoint and full metadata: [MobileNet release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-v3-small-frozen-feature-v1).

## Untouched *first* external evaluation: StitchingNet

Official publisher source: [StitchingNet GitHub](https://github.com/hyungjungkim/StitchingNet), [Zenodo 10.5281/zenodo.20352813](https://zenodo.org/records/20352813), official ZIP MD5 `51b228d9572b925035e9e9af01c96569`, CC BY-NC 4.0. The 14,565-image archive's class codes/counts match its published metadata. For a binary **normal vs broken** task, retain **class 00 normal (2,436)** and **class 02 broken stitch (1,201)**, exclude other nine defect classes. Test N=3,637; no external train/validation split and no threshold tuning. Cross-archive exact SHA-256 overlap check passed (no byte-identical images). NOTE: semantic near duplicates cannot be ruled out by SHA alone.

| Model (FROZEN before external test) | Accuracy | Balanced accuracy | Macro-F1 | Normal recall | Broken recall | Faulty ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Rebuilt SmallCNN from scratch on earlier candidate-family train split | 0.356063 | 0.496921 | 0.314555 | 0.082102 | 0.911740 | 0.482144 |
| Pretrained MobileNetV3-Small + balanced logistic head | **0.664559** | **0.642579** | **0.635371** | **0.707307** | **0.577852** | **0.687403** |

MobileNet confusion matrix, rows actual 0/faulty 1 and columns predicted actual 0/faulty 1: `[[1723,713],[507,694]]`. Its **507/1,201 missed broken stitches** make it unsuitable for deployment. Accuracy-only comparison is misleading: always predicting the dataset majority class (normal) yields **2,436/3,637 = 66.98%**, approximately the MobileNet's 66.46%. Macro-F1 and balanced accuracy reveal a more meaningful improvement over degenerate baselines.

SmallCNN confusion matrix: `[[200,2236],[106,1095]]`. It has extremely low normal recall (8.21%), even though broken recall is high; ROC-AUC below 0.5 suggests failure under this domain shift.

**Full reproducible artifact:** [First frozen external test release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1). [Successful GitHub Actions run](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39135577701).

## Explicit next gates — NO-GO for scientific submission as an independent-source detector

1. Obtain/verify unedited photographs and any Photoshop edit provenance to validate all 151 claimed links to the 31 parent photos. Expert image-review contact sheets are **candidate evidence**, not proof.
2. Train a genuinely **photo-parent-disjoint** model using the new 31-parent inferred split. This is a *new experiment*, not an already completed result. Further freeze/approval is needed before treating its held-out metrics as independent.
3. Record a new independent evaluation dataset for future iterative tuning. **Do not repeatedly test on this same StitchingNet benchmark and continue calling it untouched**, now that its results have been inspected.
4. Provide uncertainty/error analysis and per-fabric performance; do not claim real-world/production viability. The latest external broken-stitch recall of 57.8% is inadequate for unattended defect inspection.
5. Update the IEEE paper's claims and author declarations only after verification; historical 94.91% is an unreplicated 2022 validation metric, not comparable to the above frozen external test.

These details preserve dataset source status, model lineage, training/validation distinction, and reproducibility for subsequent independent review.

## Follow-up: a better-parent-grouped pretrained baseline fitted, internal validation ONLY

After the original one-shot StitchingNet result was completed, a separate baseline was trained using the **31 normal-photo candidate families** linked to all 151 Photoshop faulty-name families. This follow-up model does NOT use the original old 182-family partition, and **was never evaluated on StitchingNet** (which must no longer be called untouched for subsequent runs).

New split by inferred photograph parent IDs: **21 train / 5 validation / 5 preserved internal test**, with group exclusivity across *both class labels*. Exactly one corrupted normal PNG was excluded. Training **5,019 images** (1,878 normal / 3,141 faulty); validation **1,259** (433/826); internal holdout **1,086** (422/664, not touched by this model).

Model: frozen ImageNet1K-V1 MobileNetV3-Small backbone, fixed C=1 balanced logistic head, 224×224 RGB. No external selection or tuning.

| New inferred-photo-parent validation metric | Value |
|---|---:|
| Accuracy | 0.648133 |
| Balanced accuracy | **0.672504** |
| Macro-F1 | **0.641908** |
| Normal recall | 0.750577 |
| Faulty recall | 0.594431 |
| Faulty ROC-AUC | 0.708058 |
| Confusion matrix (actual/faulty by prediction) | [[325,108],[335,491]] |

[Checkpoint `mobilenet_v3_small_photo_parent_inferred.pt` and metadata](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1) | [Successful training workflow](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39136623853).

**Interpretation:** The mapping is inference from Photoshop-related names plus some high visual similarities, not manual audit of unaltered source photographs. Because validation consists of only five inferred photographed sources, metrics may be variable and are still exploratory. This model is NOT the model that achieved **0.635 external macro-F1** on StitchingNet earlier. No comparable external results exist for the new photo-parent model.
