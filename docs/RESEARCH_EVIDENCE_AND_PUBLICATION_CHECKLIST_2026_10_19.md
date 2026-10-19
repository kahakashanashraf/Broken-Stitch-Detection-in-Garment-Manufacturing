# Broken-stitch evidence status and reproducibility checklist (19 October 2026)

**This IEEE paper is an evidence-based working research draft, not cleared for conference submission.**

## Measured new results (do not conflate checkpoints)

- Original ZIP: 7,365 image records; 7,364 readable; 31 candidate normal photographed-parent names and 151 candidate Photoshop faulty families. All 182 UUID-derived augmentation families cross at least two original train/val/test directories. This is code-verified filename lineage, **not** human-verified photographic source lineage.
- NEW 31-photo-parent-INFERRED MobileNetV3-Small checkpoint, frozen internal group holdout: 5 inferred photo parents / 1,086 augmented records; accuracy 80.20%, balanced accuracy 81.74%, macro-F1 0.799, faulty recall 74.85%, ROC-AUC 0.914; matrix [[374,48],[167,497]]. **No external StitchingNet result exists for this checkpoint.**
- OLDER 182-augmentation-family pretrained MobileNet, first external StitchingNet zero-adaptation evaluation: 3,637 images, accuracy 66.46%, balanced accuracy 64.26%, macro-F1 0.635, broken recall 57.79%, matrix [[1723,713],[507,694]]. The older scratch CNN external macro-F1 was 0.315. The real StitchingNet result is not transferable to the newer photo-parent-trained model.
- Historical thesis VGG-16 94.91% was a reported **validation** accuracy with unrecovered code and partition history, not reproduced and not comparable to either test above.

## Reproducibility sources (full raw URLs)
- Original project: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing
- Original ZIP release: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/original-thesis-dataset-v1
- Original archive audit: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/audits/original_dataset_inventory_2026_10_18.json
- Leakage audit: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/audits/candidate_parent_overlap_2026_10_18.json
- 31 photo candidate graph: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/tree/main/audits/photographed_source_inference
- NEW photo-inferred pretrained checkpoint: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1
- NEW frozen internal holdout: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1
- OLDER pretrained checkpoint: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-v3-small-frozen-feature-v1
- OLDER one-shot StitchingNet external: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1

## New public-data leads / explicit reasons not yet an independent binary benchmark
- Roboflow **Sewing defects** (Fabric Defects, 1,778 images, Broken/Skip *bounding-box* classes): https://universe.roboflow.com/fabric-defects-dxiz7/sewing-defects-p0d83-32vo7 — online listing validated; cannot interpret absence of a Broken bbox as a clean normal-stitch negative without the dataset's explicit normal labels. Audit independent source and image overlap with StitchingNet before treating it as new external test. Public webpage listing is **not proof** downloaded images are suitable.
- Roboflow **Fabric Defect Detection** (Yolo for ar; 2,196 images; Broken-Stitch, Fabric-defects, Overlapped stitch, Stains and Damages): https://universe.roboflow.com/yolo-for-ar/fabric-defect-detection-rjvau — object-detection labels, class mapping of nonbroken images to pristine normal is NOT established; underlying image provenance/duplication unknown.
- Roboflow **Textile defect datasts** (DS, 271 images, 28 heterogeneous defect classes including broken stitch): https://universe.roboflow.com/ds-gfhaj/textile-defect-datasts — missing verified matched independent normal class and unknown origin.
- Roboflow **Garment Defect Detection** (shasti, 66 images, 62 labels including broken stitch): https://universe.roboflow.com/shasti/garment-defect-detection-hekka — too small/heterogeneous for defensible normal/broken benchmark without manual curation, possible mislabelling.
- 2026 **Vina-Embroidery** (1,080 normal/abnormal embroidered-surface images; broken-thread localization among 4 defect types): https://data.mendeley.com/datasets/sp99tbxchp/1 — independent embroidery application, but **NOT broken sewing seam classification**. It can support a separately labelled out-of-domain anomaly stress test, not a replacement binary normal/broken-seam benchmark.
- StitchingNet-Seg: https://github.com/hyungjungkim/StitchingNet-Seg — derived from already examined StitchingNet; cannot count as a second untouched external source.
- 2024 original broke-stitch 500-image study: https://doi.org/10.1109/ICISET62123.2024.10939324 — public article reports dataset, but accessible downloadable, independently verified image archive not established.
- 2023 on-machine 880-image broken-stitch study: https://doi.org/10.1093/jcde/qwad069 — peer-reviewed method/data reported, public raw image archive not verified.

**No second independent, source-traced, label-compatible image set has been confirmed or scored.** Do not manufacture external metrics, relabel dropped knit stitches as broken sewing seams, or treat Roboflow annotated defect boxes as labelled clean-negative images by default.

## Manuscript final gates
1. Correct link/citation/metadata audit. Current 4-page anonymous IEEE source and figures compile; no author identities in PDF.
2. Obtain actual parent photographs, per Photoshop defect's edit chain and consent/ownership / license. Historical images may include internet-origin photographs and publication rights must be signed off.
3. Confirm all coauthor scientific conclusions and authorship/AI-use declaration. Original ICEEICT LLM policy must be checked; the drafting/research aid was substantive, not merely spell-checking.
4. Do not call the now-observed 5-photo holdout or already-observed StitchingNet test 'untouched' after making changes informed by their scores.
5. If collecting new data, use real broken sewing stitches, independent factory/camera/garments, at least 2 independent label checks, physical parent IDs, clean negatives, consent, and frozen checkpoint+threshold prior to scoring.
6. Do not describe any checkpoint as production-safe; in old cross-domain test MobileNet missed 507 of 1,201 broken stitches and new augmented-internal holdout missed 167 of 664 faulty examples.

Verdict: **finished empirical write-up and reproducibility evidence; substantive publication ethics/provenance gates remain**. The novelty should be positioned as a *reproducibility/leakage audit with properly separated external-domain failure evidence*, not as a superior garment detector.