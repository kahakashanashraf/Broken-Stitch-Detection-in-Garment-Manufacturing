# Real broken-stitch Grad-CAM: thesis-only versus held-out-fabric supervised model

**Status:** read-only qualitative XAI study, version 1 (21 October 2026); code published with automatic release workflow. Do **not** treat the XAI figures as defect localization accuracy or a fresh independent test. Once the workflow completes, use the linked release for actual case outputs. Until then no new artifacts or results are claimed.

## Question

When the thesis-derived MobileNetV3 fails on a genuinely acquired StitchingNet broken-stitch image, does its Grad-CAM response differ from the response of a model trained on *other* StitchingNet fabrics? Is a heatmap's highest-response region important to the thesis model's broken-class logit under a simple removal perturbation?

## Data/model sources and deliberately fair constraints

- **Left:** already released **thesis-only** ImageNet→thesis-derived MobileNetV3-Small, `224×224`, `12` epochs, fitted on 26 filename-*inferred* parent groups. Locked model SHA256: `0c1eaf25ad7daea5370fd16492b811c0b90408adb3a561f47d3a5c5c48a83a81`. It was never trained on StitchingNet.
- **Right:** already published Grad-CAM from **StitchingNet-supervised** models initialized from that thesis checkpoint, trained on other fabric groups and evaluated on held-out fabric A–K groups. Reuses exact earlier Grad-CAM PNGs and report identifiers, **without retraining those models**.
- **Images:** ten selected *broken* examples from five previously evaluated fabric-group folds (one supervised true positive and one supervised false negative each). Their selection is explicitly conditional on the previously observed **supervised** model errors. This is **not a randomly sampled representative set**, cannot support prevalence, impact estimates or quantitative external scores.
- **Source:** https://zenodo.org/records/20352813; ZIP MD5 `51b228d9572b925035e9e9af01c96569` (official source, CC BY-NC 4.0 indicated by StitchingNet authors). The workflow never publishes raw photos or raw photo overlays from this dataset.
- **Threshold:** fixed 0.5 for descriptions only. No fitting, backprop training, threshold sweep, tuning, or recomputing the supervised folds occurs in this study.

## XAI implementation

1. Verify model SHA, official ZIP MD5, all five source reports and **no fabric identity overlap** in their training/evaluation split. For every selected example verify the exact source member path, real broken ground-truth label, report category, CAM dimensions and supervised model's frozen probability.
2. Load real image in memory and compute broken-class Grad-CAM through MobileNetV3 convolutional feature tensor's activations and true autograd gradients. Save only independent grayscale heatmaps. Construct **left/right same-image, different-model** grayscale panels for qualitative interpretation. Heatmaps are normalized separately per case and per model: their intensities are not directly comparable as absolute activation amounts. The corresponding released source report is linked for each image.
3. Run an optional, *descriptive* sensitivity check on thesis-only model: replace the **top 20%** highest Grad-CAM pixels with ImageNet-channel mean RGB (zero in normalized space) and compare the broken-class logit drop to **bottom 20%** removal. Record signed drops, including negative deltas, without cherry picking.
4. Save per-case provenance, scores, signed changes, identical image identifiers, ten thesis CAMs, ten paired panels and CSV/JSON evidence in one immutable release. Original StitchingNet photos remain only inside official ZIP and never get distributed in figure assets.

## How to read it

A brighter region means greater gradient-weighted positive evidence for the broken logit **under this approximate Grad-CAM calculation**, not a verified defect mask, causal proof, nor physically precise stitch localization. The occlusion perturbation creates an out-of-distribution image and cannot validate lesion/defect localization. There are no bounding-box or segmentation annotations for these exact ten cases used to score region alignment. The previously inspected StitchingNet and the retrospectively selected cases mean this analysis is descriptive, not confirmation of an unbiased improvement.

**Do not mistake zero-shot and target-domain adaptation as the same evaluation protocol.** StitchingNet-trained fabric-heldout model **does** use target-domain labels from *other fabrics*; thesis-only model does **not**. Differences may be driven by domain adaptation, training schedules and other recipe details. No causal claims are justified from the ten heatmaps.

## GitHub code and artifacts

- Analysis code: `code/scripts/xai_thesis_vs_fabric_holdout.py`
- Synthetic fixture/sanity tests: `code/tests/test_xai_same_image.py`
- Automatic single-run workflow: `.github/workflows/xai-thesis-vs-fabric.yml`
- Earlier genuinely trained supervised Grad-CAM evidence: `docs/FABRIC_HELDOUT_DOMAIN_ADAPTATION_2026_10_20.md`
- Once published: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/thesis-only-vs-fabric-heldout-gradcam-v1

## Additional independent validation needed

Before claiming reliable, interpretable real garment inspection: physically independent, newly acquired normal and broken sewing photos; actual annotated bounding boxes/masks of defects to evaluate localization; assess Grad-CAM parameter/label randomization sanity, robustness to augmentations, fabric-wise failure correlations and expert review. The source parent lineage and third-party image permission remain unverified; obey conference policies on AI involvement in research and writing.