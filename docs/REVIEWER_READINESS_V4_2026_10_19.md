# Broken-Stitch paper v4 — reviewer defense and reproducibility gates

Updated 19 October 2026. **Research working draft, not submission-cleared.**

## What is new relative to the historical 2022 thesis?

The old thesis compared convolutional classifiers and reported 94.91% **validation**, not independent test accuracy. The current study instead reconstructs the surviving data lineage and quantifies evaluation-protocol sensitivity. It does **not** claim a novel MobileNet architecture nor that the exact historical VGG-16 model has been reproduced.

The core new forensic result is that all **182/182** UUID-stripped augmentation families appear across the original ZIP's old train/val/test folders. Zero byte-identical SHA-256 duplicates does not contradict augmentation-family contamination. All **151** `faulty_original_XXN` families map by filename pattern to **31** `actual_original_XX` candidate photo parents, but no independent Photoshop edit file / underlying camera original has verified the mapping; the number of independent physical garments is not known.

## Strongest new controlled numerical finding

The **first** development-only five-fold partition-sensitivity experiment used the same 6,278 images from 26 inferred parent candidates, same frozen ImageNet MobileNetV3-Small features, same StandardScaler+balanced logistic regression, and fixed C=1 and threshold 0.5. The 5 previously inspected internal-test parents and external StitchingNet are excluded.

| Fixed-protocol measure | Image-random stratified CV | Inferred-photo-parent-grouped CV |
|---|---:|---:|
| Pooled out-of-fold macro-F1 | **0.8714044** | **0.6879128** |
| Pooled out-of-fold balanced accuracy | **0.8777301** | **0.6900719** |
| Photo-parent group cross-over in every fold | 26 | 0 |

Observed macro-F1 protocol difference: **0.18349 (18.35 percentage points)**. This is **not** a paired causal estimate of the historical 94.91% score, because fold composition changes and identifiers are inferred; it is an exploratory sensitivity study conducted after other test scores were inspected.

Source and report:
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/code/scripts/development_cv_split_sensitivity.py
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/development-split-sensitivity-v1
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39271250163

## Two different MobileNet checkpoints: never conflate

1. **New 31-photo-candidate grouped model**, trained on 5,019 derived images, validated on 1,259 and frozen-tested once on 1,086 derived images from just **five** inferred parent photos: accuracy **0.8020**, macro-F1 **0.7995**, balanced accuracy **0.8174**, synthetic-faulty recall **0.7485**. Confusion matrix [[374,48],[167,497]]. Original/photo lineage and natural defect authenticity unverified. One parent has normal images only.

2. **Older 182-family-trained model**, evaluated once without adaptation on official StitchingNet normal/broken images (2,436/1,201): accuracy **0.6646**, macro-F1 **0.6354**, natural broken-stitch recall **0.5779**, confusion matrix [[1723,713],[507,694]]. It misses **507 of 1,201** real broken stitches. Always-normal accuracy on this dataset is **66.98%**; thus model accuracy alone is not an achievement. **This external score belongs ONLY to the older model.**

Reproducible reports:
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1

## Major reviewer questions and how the v4 draft handles them

**Q: Is data leakage itself a new discovery?** No. The literature shows it is a well-known cross-domain problem. Novelty is narrower: an auditable source-lineage reconstruction in a sewing-stitch dataset plus within-pipeline split-policy sensitivity and a qualified actual sewing-stitch cross-domain test. Cite Kapoor and Narayanan (2023), Yagis et al. (2021), Gerz and Jelali (2026).

**Q: Did you prove the 2022 94.91% was inflated by 18.35 points?** No. Original training manifest and old model weights are absent; the controlled diagnostic uses a newly reconstructed fixed backbone/classifier.

**Q: Are 1,086 independent test samples?** No. Only five inferred photographed sources; records are correlated transformations. Report parent-level confusion counts, no naïve image-level narrow confidence intervals.

**Q: Does the result support industrial deployment?** No. The real external model missed 42.21% of broken stitches and the newer model has no truly independent real-image external test.

**Q: Are random-fold metrics genuine validation?** No. Deliberately contaminated image-level folds are used exclusively to demonstrate protocol sensitivity. Grouped folds are also source-inferred, not verified physically independent garment groups.

**Q: Are the images and code original/redistributable?** The public repo contains the original archive at the user's request. Some thesis source images were sourced from the internet, and per-image ownership/license must be established by human authors before further reuse or conference publication.

**Q: Does the paper comply with ICEEICT's generative-AI rule?** Not yet confirmed. ICEEICT requires disclosure and says LLM use must be solely for *non-intellectual* purposes; this workflow involved research code, literature assessment and manuscript drafting. Human authors must seek written confirmation from the TPC chair that the actual use is permissible and must make no false certification. Official rule: https://iceeict.mist.ac.bd/authors/paper-submission/

**Q: What about double-blind review?** Public GitHub carries identifiable author ownership. The manuscript must not include named repository references, and the program committee's rules on anonymized data/code supplements must be confirmed; author metadata and self-identification should be checked before upload.

## Stop/Go for submission

Technical paper: five-page IEEE A4, PDF under 3 MB, embedded fonts (Type 1), anonymous text check, ten scholarly references, real results with distinctions. **Engineering GO** after clean GitHub compile/CI.

Submission: **NO-GO** until (a) permission for historical figures and internet images is documented, (b) all authors approve exact text and contributions, (c) the conference approves actual substantive LLM involvement, (d) scientific claims are sufficiently hedged around inferred physical sources and inspected test sets, and (e) similarity/IEEE PDF verification is performed.

The project is a **reproducibility and evaluation case study**, not a superior industrial detector. Do not change that framing merely to maximize acceptance probability.