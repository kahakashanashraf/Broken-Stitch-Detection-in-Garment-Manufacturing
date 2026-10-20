# Broken Stitch Detection in Garment Manufacturing

[![Unit tests](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/workflows/tests.yml/badge.svg)](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/workflows/tests.yml)

**Unpublished 2022 thesis reproduction and ICEEICT 2027 research-extension working project.**

This repository contains a modern **PyTorch reconstruction** of CNN/VGG-16 experiment families, validation-safety tests, historical thesis figures, a public copy of the raw original data archive, and an IEEE-format **working manuscript**. The original 2022 source code and model weights have NOT been recovered.

## Published project assets

| Asset | Location | Status |
|---|---|---|
| Original dataset ZIP (2,146,784,353 bytes) | [GitHub release — original-thesis-dataset-v1](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/original-thesis-dataset-v1) | Uploaded. Archive integrity checked; **dataset labels, lineage and rights not audited** |
| Dataset SHA-256, file size and archive checks | [Release metadata JSON](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/download/original-thesis-dataset-v1/dataset_release_metadata.json) | Uploaded |
| Historical thesis figures (10 extracted images) | [`paper/figures/`](paper/figures/) | Uploaded; historical figures, not new experimental evidence |
| Six-page IEEE paper PDF (anonymous working draft) | [`paper/ICEEICT2027_BrokenStitch_Research_Extension_v2_Anonymous.pdf`](paper/ICEEICT2027_BrokenStitch_Research_Extension_v2_Anonymous.pdf) | Uploaded, compiled in GitHub Actions; NOT submission-cleared |
| IEEE manuscript LaTeX | [`paper/main.tex`](paper/main.tex) | Uploaded |
| Original 2022 weights | [`models/README.md`](models/README.md) | **Not recovered** |
| Newly trained reconstructed CNN weights | [Exploratory model release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/exploratory-cnn-candidate-family-v1) | **Trained but FAILED normal recall and not publication-ready** |

> **Research caveat:** The historical thesis reports **94.91% validation accuracy** for its final VGG-16 configuration; this has **not been reproduced** on a verified source-disjoint split. No new real-data or StitchingNet accuracy is claimed. Public data access alone does not establish permission for third-party reuse; see [data provenance and license notes](docs/DATA_AND_LICENSE.md).

## Code and evaluation protocol

| Area | Location |
|---|---|
| PyTorch CNN, VGG-16, filters, ensemble, training CLI | [`code/`](code/) |
| Unit and toy-smoke tests | [`code/tests/`](code/tests/) |
| Drive ZIP inventory and Colab GPU notebook | [`notebooks/`](notebooks/) |
| Source-disjoint and external StitchingNet study plan | [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) |
| Completed tasks, limitations and blockers | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) |
| Manuscript/figure notes | [`paper/README.md`](paper/README.md) |

Run the software tests from repository root:

```bash
python -m pip install -r code/requirements.txt
python -m pip install -e code/
cd code
python -m pytest -q
```

**13 automated tests passed** on synthetic fixtures in the linked GitHub Actions CI; this is software validation, not a new accuracy result.

## Next scientific steps

1. Audit archive images, labels, generated defects, original photograph and derivative source IDs; reconcile 181 / 9,134 / 10,000 / 20,000 counts.
2. Create source-disjoint training, validation and never-tuned internal-test partitions.
3. Rebuild and train models on verified source groups and report class-wise metrics.
4. Run frozen checkpoints on the strictly normal/broken subset of [StitchingNet](https://github.com/hyungjungkim/StitchingNet), observing its CC BY-NC 4.0 terms.
5. Update the draft paper **only after** results and image/author/venue-policy review.

### Anonymity and intellectual-property notice

This is a publicly attributable repository and may compromise strict double-blind submission if linked from the paper. Human authors must confirm image provenance, ownership, publishing permissions, and ICEEICT policies. **No blanket dataset, image or code license has been granted merely by making these files public.**


## Current critical audit finding and new weights (18 Oct 2026)

[Dataset inventory](audits/original_dataset_inventory_2026_10_18.json) found **7,365 images** with one unreadable file. [Family collision audit](audits/candidate_parent_overlap_2026_10_18.json) found that **all 182 candidate original-image filename families** appear across the original train/validation/test folders; **do not use the original partition as an independent test**.

A [new provisional group-exclusive SmallCNN run](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/exploratory-cnn-candidate-family-v1) was completed from scratch on the actual ZIP (7,364 readable images). Training succeeded and authentic `best.pt` exists, but the **exploratory test** had accuracy 57.38%, macro-F1 36.46%, and normal recall **0%** (all 369 normal test examples misidentified). It is **not** a functional stitch-detector and does **not** reproduce the thesis's historical 94.91% validation accuracy.

See [full current status](docs/CURRENT_STATUS.md) and [model documentation](models/README.md). True photographed-source linkage and independent StitchingNet evaluation remain unsolved.

## 18 October 2026 — New pretrained baseline and FIRST frozen external comparison

The original-data audit now suggests **151 Photoshop-faulty candidate families link to 31 normal-source families** via filename convention; this gives **31 inferred photographed-source parent IDs**, not human-verified originals. See [photo-parent audit](audits/photographed_source_inference/).

A **frozen ImageNet MobileNetV3-Small feature extractor + trained balanced logistic head** achieved validation macro-F1 **0.6483** on the *older 182-family candidate split*. [Model weights & report](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-v3-small-frozen-feature-v1).

We then performed the **first and only untouched StitchingNet binary external evaluation**, with all 2,436 normal and 1,201 broken images, frozen threshold 0.5, no external tuning:

| Frozen model | External macro-F1 | Accuracy | Broken recall | Normal recall |
|---|---:|---:|---:|---:|
| Rebuilt scratch SmallCNN | 0.315 | 35.61% | 91.17% | 8.21% |
| ImageNet-pretrained frozen MobileNetV3-Small | **0.635** | **66.46%** | **57.79%** | **70.73%** |

[External result JSON](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1) · [Complete methods, parentage mapping and limitations](docs/RESULTS_AND_LINEAGE_2026_10_18.md).

**NO-GO for deployment or scientific claims of photo-source-independent accuracy.** MobileNet misses 507/1,201 real broken-stitch images. Also, further tuning evaluated against the now-inspected external StitchingNet benchmark is *not* a new untouched benchmark. The new 31-parent proposed split has **not** been used to fit this checkpoint.

## Photo-parent-inferred pretrained MobileNet checkpoint (separate experiment)

A second ImageNet-pretrained MobileNet-V3-Small frozen-feature + balanced logistic model has **now** been fitted on the **31-photo-parent inferred graph**: 21 candidate photo parents for training (**5,019 valid images** after removal of 1 corrupt file), 5 validation parent groups (**1,259 images**), 5 hold-out parent groups (**1,086 images NOT EVALUATED**). Validation macro-F1 **0.6419**, balanced accuracy **67.25%**, normal recall **75.06%**, broken/faulty recall **59.44%**. [Download trained checkpoint and validation metadata](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1).

This is NOT a new StitchingNet evaluation. StitchingNet has already been inspected using the older 182-family-trained checkpoint, so subsequent tests on it would no longer be **untouched**. Human-source Photoshop provenance is still missing; do not represent photo-parent inference as verified original photography.

## 19 October 2026: FIRST photo-parent-inferred internal holdout

[Reproducibility report](docs/ONE_SHOT_HOLDOUT_2026_10_19.md) and [one-shot GitHub Release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1) contain the **first frozen holdout evaluation** of the **new 31-inferred-photo-parent-trained** ImageNet MobileNetV3-Small. Results on 1,086 augmented test images (only five inferred parent candidates, NOT independently confirmed natural photos): **accuracy 80.20%, macro-F1 0.799, balanced accuracy 81.74%, AUROC 0.914, normal recall 88.63%, faulty recall 74.85%**. Confusion matrix [[374,48],[167,497]]. Source `ak` alone has 94/257 missed synthetic faults; source `az` has 66 normal and zero faulty images. It is not deployment-grade or independently photographed-source-proven. The internal test has now been unsealed once. NO further tuning can cite this same test as untouched.

Research sources: [independent external dataset screening](docs/INDEPENDENT_EXTERNAL_DATASET_SCREEN_2026_10_19.md) and [provenance verification requirements](docs/PHOTO_PROVENANCE_REVIEW_REQUIREMENTS.md). An unrelated textile's broken-end/drop-stitch label must not be used as broken sewing seam. No verified **new** independent binary external dataset was obtained; **NO new external accuracy** claimed for this newer checkpoint.


## New evidence-based empirical paper (19 October 2026)

**[Read the v3 anonymous empirical paper](paper/ICEEICT2027_BrokenStitch_Evidence_Audit_v3_Anonymous.pdf)** · [LaTeX source](paper/evidence_audit_v3.tex) · [Dataset-source screening and publication checklist](docs/RESEARCH_EVIDENCE_AND_PUBLICATION_CHECKLIST_2026_10_19.md)

This manuscript replaces the unexecuted-protocol-only emphasis with **actual measured** original-ZIP augmentation-family overlap, new inferred-photo-parent internal holdout and an earlier model's untouched real StitchingNet cross-domain evaluation, keeping each checkpoint explicitly distinct. The title is *How Reliable Is Broken-Stitch Classification? An Augmentation-Lineage Audit and Cross-Domain Evaluation*. Four IEEE A4 double-column pages, intended as a research draft; image provenance and conference AI-use policies prevent a submission-cleared claim.

**Key caution:** 80.20% internal holdout accuracy / macro-F1 0.799 from **new inferred-photo-parent MobileNet** is not the same model as the older MobileNet's one-shot StitchingNet macro-F1 0.635 / broken recall 57.79%. The 94.91% original thesis VGG-16 is historical unreproduced validation only. No second source-independent normal/broken-sewing-seam dataset has been validated; Roboflow leads are under review and must not be scored as clean-negative binary data by assuming an absent defect bounding box means normal sewing.


## Improved v4 empirical paper: controlled split-protocol evidence (19 October 2026)

**Current working manuscript:** [5-page v4 anonymous PDF](paper/ICEEICT2027_BrokenStitch_Evidence_Audit_v4_Anonymous.pdf), [v4 IEEE LaTeX](paper/evidence_audit_v4.tex), [reviewer/readiness checklist](docs/REVIEWER_READINESS_V4_2026_10_19.md).

The new **development-only 5-fold split-protocol diagnostic** used 6,278 images from 26 inferred photographed-source parents, **without accessing** the previously inspected five-parent internal holdout or StitchingNet. Identical frozen MobileNetV3-Small features and logistic classifier produced image-random pooled OOF macro-F1 **0.8714** versus inferred-parent-grouped **0.6879**; observed **18.35 percentage-point difference**. In every random fold, all 26 inferred parents cross fold-training/validation; in every grouped fold, zero cross. This is an **exploratory observational protocol sensitivity** finding, not a causal estimate of historical 94.91% overoptimism or verified real-photo independence.

[Full CV JSON and GitHub Action](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/development-split-sensitivity-v1). The previous frozen holdout, external StitchingNet and all raw parentage caveats remain unchanged. **This paper remains NOT submission-cleared**, especially due to unverified original image ownership/provenance and ICEEICT's strict declared LLM-use rule.


## Leakage-controlled dataset v2 — now published

**[Download physically reorganized train/validation/test ZIPs](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/leakage-controlled-dataset-v2-inferred-31-parents)** · [Data cleaning report](docs/DATASET_V2_LEAKAGE_CONTROL_2026_10_19.md)

The original augmented-image split leaked all 182 edit families. Version 2 repartitions **7,364 readable images by 31 inferred photographed-source roots** (train 5,019 / val 1,259 / test 1,086, with 21/5/5 candidate roots) and discards one corrupt original. Verified **0 cross-split candidate parents and 0 cross-split exact-byte duplicates**. The source-photo mapping is still *inferred*, and validation/test still consist of historical augmented images; we have **not** recovered true unedited raw photos. Existing held-out scores are already inspected, and no new accuracy was generated just by cleaning ZIP folders.

Please use v2 instead of the original leaky train/val/test directories for reproducible future work.


## New model experiment on cleaned dataset v2

An **ImageNet-pretrained EfficientNet-B0 frozen image encoder** with a **newly supervised-trained, inferred-photo-parent/class-balanced logistic head** is configured in [`code/scripts/train_clean_effnet.py`](code/scripts/train_clean_effnet.py). It uses only clean v2 `train.zip` (5,019) and `val.zip` (1,259), 21/5 inferred photo parents, verifies archive/member checksums and **does not download test.zip or StitchingNet**. The older inspected test and external metrics MUST NOT be transferred to this model. Inference code: [`code/scripts/predict_clean_effnet.py`](code/scripts/predict_clean_effnet.py). See the dedicated GitHub Actions workflow; when successful, the separately versioned training-report and authentic checkpoint will be made available as a GitHub Release. The dataset is still composed of historical augmented Photoshop-derived images, not independently verified raw photos.


## 20 October 2026 — Controlled training intervention outcome

[**Exact 3-arm same-backbone source-aware training results**](docs/SOURCE_AWARE_ABLATION_RESULTS_2026_10_20.md) · [verified grouped-fold methodology](docs/SOURCE_AWARE_ABLATION_PROTOCOL_2026_10_20.md) · [corrected structured results](docs/source_aware_ablation_results_v1_corrected.json).

MobileNetV3 class-balanced standard training grouped-CV macro-F1 **0.74347**, additional filename-parent-balanced loss **0.74343**, source-balanced + training augmentation **0.73781**. Paired inferred-parent bootstrap 95% descriptive difference intervals for B–A **[-0.00978, +0.00810]** and C–B **[-0.02386, +0.01562]** cross zero; do **not** claim a demonstrated macro-F1 gain from source weighting or online augmentation. B faulty recall **77.31%** vs A **75.88%** trades off normal recall. These were 26 inferred parents/6,278 augmented records in **development-only grouped CV**, not a new untouched external test. A Release v1 *wording-only* erratum corrects its parent-capped description to identical **globally class-stratified 800+800 examples**. Original released metrics/checkpoints have NOT changed.
