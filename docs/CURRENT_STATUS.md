# Research status — verified on 18 October 2026

**Checkpoint created and publicly released**, but not validated for publication or deployment.

| Item | Verified evidence | Status |
|---|---|---|
| Original ZIP | 2,146,784,353 bytes; [Release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/original-thesis-dataset-v1) | Published |
| Archive audit | [JSON](../audits/original_dataset_inventory_2026_10_18.json) | 7,365 images; 7,364 decoded; 1 corrupt |
| Original three-way partitions | [Candidate parent audit](../audits/candidate_parent_overlap_2026_10_18.json) | **NO-GO: 182 of 182 candidate filename families appear across splits; all 7,365 images potentially affected by augmentation-family leakage** |
| Candidate families | 31 normal, 151 faulty inferred from UUID-suffixed image filenames | Unverified true underlying photograph lineage |
| Replacement grouped partition | [Exploratory release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/exploratory-cnn-candidate-family-v1) | Fresh train 5,167, val 1,120, test 1,077; old folders ignored |
| PyTorch reconstructed SmallCNN | [trained best.pt](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/exploratory-cnn-candidate-family-v1) | **Trained from scratch on the actual original archive** |
| Automated code tests | [CI](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/workflows/tests.yml) | 13/13 pass |
| Model evaluation | [Workflow](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39133504317) | Completed; details below |
| Source photography / Photoshop-derived parentage | No direct original-photo mapping | **BLOCKED** |
| Independent StitchingNet evaluation | Not run | **BLOCKED** |
| Submission-ready new accuracy | None | **NO-GO** |

## Exploratory candidate-family-held-out result — NOT paper evidence

The team asked to fit a new model; this is a real fitting run, **not** the 2022 historical weights. Rebuilt SmallCNN / 100×100 RGB / Adam 0.001 / class-weighted cross entropy / batch 32 / train-time affine augment / CPU. Best epoch **1** selected using validation macro-F1 among 3 epochs (early stopping patience 2).

| Metric | Provisional candidate-family test result |
|---|---:|
| Test N | 1,077 (369 actual, 708 faulty) |
| Accuracy | 0.573816 |
| Balanced accuracy | 0.436441 |
| Macro-F1 | 0.364602 |
| AUROC (faulty as positive) | 0.446083 |
| Normal/actual recall | **0.000000** |
| Faulty recall | 0.872881 |
| Confusion matrix row actual → predicted actual/faulty | 0 / 369 |
| Confusion matrix row faulty → predicted actual/faulty | 90 / 618 |

**Interpretation:** All 369 normal images in the candidate-family-held-out test were flagged as faulty. The model performs worse by accuracy than always predicting the majority faulty class (708/1,077 ≈ 65.74%). Despite legitimate trained weights, it is **not usable** as a garment inspection detector. True photo-source contamination could remain even across candidate filename groups. Metrics must not be presented as reliable real-world performance.

## Critical next gates

1. **Source linkage:** Manually trace every normal/Photoshop-faulty candidate back to its underlying real photograph; 182 filename families are not proof of 182 independent photo sources.
2. **Data quality:** Inspect label correctness, one corrupt original (`dataset/train/actual/actual_original_AB.png_579370ba-65d0-4793-a3c0-0f73fbf6c367.png`), train augmentation and distributions.
3. **Improved protocol:** Rebuild photo-source-grouped partitions, retain untouched independent test and external StitchingNet subset, predeclare validation metric and experiments; do not repeatedly tune to this already-inspected exploratory test.
4. **New baseline:** After photo-source audit, train genuinely pretrained VGG-16/MobileNet; compare group-held-out balanced accuracy, class-wise recall and macro-F1, and independently test real StitchingNet normal/broken images.
5. **Paper:** Historic 94.91% remains thesis **validation**; do not claim the rebuilt checkpoint reproduces it.

The released archive's original 2022 reported 181 image sources and the observed 182 filename groups need reconciliation, not invented explanation. Publication rights and conference AI/anonymity policies remain separate unresolved gates.

## 18 October update — pretrained models and FIRST external evaluation

The previous scratch-CNN checkpoint remains available. Newly [pretrained MobileNetV3-Small frozen-feature checkpoint](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-v3-small-frozen-feature-v1) has been trained on the older 182-family train/val grouping (MobileNet validation macro-F1 0.6483). [One-shot external report](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1): 3,637 StitchingNet images; MobileNet macro-F1 0.6354, normal recall 70.73%, broken recall 57.79%; old SmallCNN macro-F1 0.3146. All models and 0.5 threshold were frozen BEFORE viewing StitchingNet labels for evaluation.

[Photoshop parent inference](../audits/photographed_source_inference/) maps 151 faulty candidate families to 31 normal candidate photo names (filename match rate 100%); human visual/source-document verification still outstanding. **The MobileNet was NOT trained on the newly inferred 31-photo-parent split.** Full results/limitations: [research report](RESULTS_AND_LINEAGE_2026_10_18.md).

The StitchingNet benchmark has now been examined. Do **not** reuse it for post-hoc model tuning and claim any subsequent assessment is untouched. Still **NO-GO** for submission-ready independent-source performance, device use or a verified photo-parent classification benchmark.

## 31-photo-parent-INFERRED ImageNet MobileNet internal training finished

[Successful GitHub Actions run](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39136623853) and [frozen weights](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1). Training (actual/faulty) 1,878/3,141 = **5,019** after excluding exactly one corrupt normal PNG. Validation (433/826) = **1,259**. Holdout 422/664 = **1,086** was **not accessed** by this training workflow.

Validation ONLY: accuracy 0.648133, balanced accuracy **0.672504**, macro-F1 **0.641908**, normal recall **0.750577**, faulty recall **0.594431**, AUROC **0.708058**; matrix [[325,108],[335,491]]. This is not directly comparable with older 182-family validation or StitchingNet external scores due to different partitions, and true underlying photographs are not independently confirmed.

No further StitchingNet evaluation was made. Human inspection and separate unseen-domain test remain required.

## 19 October frozen holdout (not photographed-source independent)

[Single successful holdout run](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39264164430) / [result JSON](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1): 1,086 photo-parent-inferred held-out synthetic/augmented images; only 5 parent roots; MobileNet 0.8020 accuracy, 0.8174 balanced accuracy, 0.7995 macro-F1, 0.9135 ROC-AUC. Faulty recall 497/664 = 0.7485. A 31-photo-parent mapping remains inference only and original image rights/source verification missing. Full limitations: [19 Oct report](ONE_SHOT_HOLDOUT_2026_10_19.md). New genuinely independent real normal/broken sewing-stitch data [not located](INDEPENDENT_EXTERNAL_DATASET_SCREEN_2026_10_19.md).

The previously published StitchingNet 0.635 macro-F1 belongs to the older 182-family model; it must NOT be assigned to the newer photo-parent model. The old StitchingNet and **now the 5-parent holdout** are both inspected, and cannot be called untouched after subsequent selection/tuning. No new external benchmark was run today.
