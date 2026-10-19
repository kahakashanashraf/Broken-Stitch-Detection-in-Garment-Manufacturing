# FIRST frozen internal photo-parent holdout — 19 October 2026

**Completed one time** without training, threshold recalibration or checkpoint selection on holdout images.

- Reproducible GitHub Actions: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39264164430
- Immutable frozen result JSON: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1
- Model prerelease: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1
- Model SHA-256: `02fa682d0648aa2521d50bb5f402ba4c8452ee71fd079333d135af9178435253`
- Dataset ZIP SHA-256: `e7aed1355a712c14bbe6560fefbb3dfab546a38970f836d129f5721e0b046979`
- Parent-split CSV SHA-256: `769d10dc6f6651f24d655ae0ed6bc1f780a50f24882c0b7fd5073fec1a51d2cb`
- Grouping used: candidate source base stems, 31 parent candidates (21 train / 5 val / 5 held-out test), derived from filename matching, NOT independently verified original photograph provenance.

The frozen ImageNet-pretrained MobileNetV3-Small features with logistic classification head were trained on 5,019 augmented images and validated on 1,259 images. They had not accessed the 1,086 candidate-parent heldout images before the one-time test. **No StitchingNet evaluation has been performed for this newer 31-parent model.**

## Aggregate internal holdout (augmented original-archive images, 5 inferred photo parents)

| Metric | Result |
|---|---:|
| Test images | 1,086 |
| Normal images | 422 |
| Faulty images | 664 |
| Accuracy | 0.80202578 |
| Balanced accuracy | 0.81737495 |
| Macro-F1 | 0.79945322 |
| ROC-AUC, faulty positive | 0.91350354 |
| Average precision, faulty positive | 0.94795779 |
| Normal recall | 0.88625592 |
| Faulty recall | **0.74849398** |
| Confusion matrix (true normal/faulty by predicted normal/faulty) | **[[374,48],[167,497]]** |

**Interpretation:** Missed **167/664** Photoshop-faulty images (25.15% missed defects). These cannot be considered held-out independent natural broken-seam photographs. In addition, the effective number of holdout *photo parent candidates* is **5**, not 1,086 independent observations. No statistical generalization claim should use naïve independent-image confidence intervals.

## Source-level variation (five inferred parents, one parent contains only normal-class examples)

| Candidate parent | Images | Normal / Faulty | TN | FP | FN | TP | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `photo_candidate:ag` | 213 | 89 / 124 | 86 | 3 | 8 | 116 | 94.84% |
| `photo_candidate:ak` | 333 | 76 / 257 | 63 | 13 | 94 | 163 | 67.87% |
| `photo_candidate:ay` | 245 | 99 / 146 | 86 | 13 | 36 | 110 | 80.00% |
| `photo_candidate:az` | 66 | 66 / **0** | 66 | 0 | 0 | 0 | 100.00% |
| `photo_candidate:bb` | 229 | 92 / 137 | 73 | 19 | 29 | 108 | 79.04% |

The large performance variation, especially the 94 missed defects under source `ak`, should be investigated on a future **new** verification set rather than tuning to this inspected holdout.

## Mandatory limitations and research consequences

1. The unpublished thesis (Section 4.3) describes 31 normal images and Photoshop generation of ~150 faulty examples, but its collection description refers both to garment-site collection and internet-sourced normal images. Original RAW/JPGs, Photoshop editor files and per-photo license/consent records were not located in accessible project files. The 151 faulty filename families vs ~150 reported is not resolved.
2. The **photo-parent relationship is inferred**, not independently confirmed. Independent photographed-source classification accuracy cannot be asserted.
3. All 1,086 holdout images arise from augmentation/synthetic defect transformations in the original archive. No trustworthy new camera/site generalization is established.
4. The old internal 182-family split was contaminated by likely cross-family photo ancestry; old MobileNet's one-shot StitchingNet external macro-F1 0.635 belongs **only to that old model**. Do NOT transfer it to this 31-photo trained checkpoint.
5. **This test is now inspected**. No revised architecture/threshold chosen using this score can claim a new untouched result on these exact parents.
6. [New independent external dataset screening](INDEPENDENT_EXTERNAL_DATASET_SCREEN_2026_10_19.md) found no fully qualified **independently sourced normal/broken sewing stitch** public benchmark for a new, comparable binary evaluation. StitchingNet-Seg descends from the already examined StitchingNet. Knitting-machine drop stitch or woven-fabric broken end labels are NOT equivalent to broken sewing seams.

### Verdict
- **Engineering:** GO — frozen test completed and result/provenance released.
- **Scientific independence:** NO-GO — requires provenance proof and genuinely independent newly sourced genuine stitch images.
- **Deployment:** NO-GO — 25% of synthetic faulty holdout samples missed and no production validation.

Next: request source photo files and Photoshop lineage/permission documentation from original collectors. Obtain independently labelled *real* normal/broken sewing stitch data for one future externally frozen evaluation. No further benchmarking using the now-inspected holdout or StitchingNet as untouched data.
