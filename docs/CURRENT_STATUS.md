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
