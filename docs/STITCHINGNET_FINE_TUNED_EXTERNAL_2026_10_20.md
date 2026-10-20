# Six genuinely fine-tuned models on different dataset: StitchingNet (20 Oct 2026)

## Research question
How well do classifiers trained **only on the original thesis augmented/Photoshop-derived stitch data** transfer to **StitchingNet's real sewing-machine images** when classifying *normal stitch vs broken stitch*?

This is **external-domain transfer but not a new blind or untouched benchmark**. StitchingNet was already inspected previously using an older model. No external label-based model tuning, threshold selection, calibration, training, or fine-tuning was performed in the evaluation workflows. Models were frozen as published prior to the new transfer runs.

- Source: https://zenodo.org/records/20352813
- Official archive MD5: `51b228d9572b925035e9e9af01c96569`
- Binary subset: 2,436 normal class 00, 1,201 broken class 02 = 3,637 images across eleven fabric-thread combinations.
- All other classes excluded; 0.5 threshold pre-fixed on each of six model checkpoints. No raw StitchingNet images committed to this repository.
- Prior project's earlier exact-byte hash intersection against original 2.15-GB thesis ZIP was zero. Exact-byte checking does not exclude transformed/cropped duplicates, though different named data acquisition sources are documented.
- All previously published original-source photo parent IDs were filename-inferred, NOT independently authenticated.

## Actual verified external results

| Frozen development-trained checkpoint | Accuracy | Macro-F1 | Balanced accuracy | Broken recall | Broken misses (of 1,201) | False broken alarms (of 2,436) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: class-balanced MobileNetV3 | **75.34%** | **0.72163** | **72.20%** | 62.95% | 445 | 452 | 0.77686 |
| B: class-and-parent-balanced MobileNetV3 | 74.26% | 0.71091 | 71.23% | 62.28% | 453 | 483 | 0.76539 |
| C: B + train-only augmentation | 71.82% | 0.69926 | **71.55%** | **70.77%** | **351** | 674 | 0.78304 |
| MobileNetV3 architecture-CV refit (different training recipe) | 74.43% | 0.67629 | 66.54% | 43.30% | 681 | 249 | 0.78290 |
| DenseNet121 architecture-CV refit | 73.93% | 0.71144 | 71.63% | 64.86% | 422 | 526 | **0.80749** |
| VGG16 architecture-CV refit | 75.12% | 0.61940 | 62.32% | **24.65%** | **905** | 0 | 0.79401 |

**Interpretation:** Within the controlled A/B/C training ablation, source balancing did NOT improve external macro-F1; C trades 102 fewer missed broken stitches than B for 191 extra false positive alarms. **A gives the best observed macro-F1**, not necessarily the best application-specific cost profile. More expensive pretrained backbones did not automatically give better external macro-F1; VGG16 misses 905 real broken images at its pre-fixed .5 threshold. No tuning was allowed after examining these external labels.

A trivial **always-normal classifier** gets 2,436/3,637 ≈ **66.98%** accuracy and **zero broken recall**. Thus use macro-F1, balanced accuracy and broken recall, not accuracy alone, for deployment discussions.

## Caveats, uncertainty and publication status

1. No real factory deployment study was conducted. StitchingNet and StitchingNet-Seg are NOT independent of one another. StitchingNet has been examined previously in this project, so results are non-blind descriptive model transfer, not untouched confirmatory evidence.
2. Both original/augmented thesis sources and the 31 inferred photo parents remain unverified by original camera/edit logs.
3. Two families of results are distinct: the older *architecture-screening CV* 3 models and the matched **controlled source-balance ablation A/B/C**. Cross-family differences involve different training recipes and are not causal architectural ablations.
4. The [six-model comparison release](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-comparison-v1) holds immutable per-image predictions, fabric-stratified metrics, and paired fabric-bootstrap differences. The 11 fabric groups used for the bootstrap are not 11 independently random source-photography datasets. Intervals are descriptive, not guaranteed generalization significance.
5. This dataset is under the creators' CC BY-NC 4.0 license; this project did NOT republish StitchingNet raw images.
6. Neither this descriptive model transfer nor internal 26-inferred-parent CV alone permits a state-of-the-art industrial accuracy claim.

## Download exact results and reproduce

Six-model single evaluation [successful workflow](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39441584489) (all jobs completed success).

Paired fabric comparison [successful workflow](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39444557394).

[Release summary, comparisons and uncertainty](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-comparison-v1).

- A: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-class_only-v1
- B: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-parent_class-v1
- C: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-parent_class_aug-v1
- Architecture MobileNet: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-mobilenetv3-v1
- DenseNet: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-densenet121-v1
- VGG16: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-vgg16-v1

Source scripts: `code/scripts/evaluate_stitchingnet_finetuned.py` and `code/scripts/compare_stitchingnet_transfer.py`. Pretrained checkpoints were not modified.

## Alternative public datasets screened

[Roboflow Stitching Full](https://universe.roboflow.com/tts-workspace/stitching-full-4t6ge) lists classes `Normal`, `BrokenStitch` etc., but public metadata does not establish capture provenance, class semantics, or independence from StitchingNet. Therefore no new independent results can be claimed on it from this run. [StitchingNet-Seg](https://github.com/hyungjungkim/StitchingNet-Seg) explicitly derives from StitchingNet, so it is not a new independent test.
