# Models and pretrained weights

## Newly trained exploratory checkpoint (18 Oct 2026)

[**GitHub prerelease: trained SmallCNN best.pt**](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/exploratory-cnn-candidate-family-v1)

The real `best.pt` model was trained using 7,364 decoded images from the team's original 2.15 GB ZIP with new candidate UUID-filename family grouping (the original leaked splits were ignored). Exact artifact: `best.pt` **418,747 bytes**, SHA-256 `f66b1623ebce9b07c98ef7e6550c98bf42298b831d1d918debc5b15059696991`.

Artifacts published together:
- `candidate_group_splits_portable.csv` — all train/validation/test assignments
- `candidate_split_report.json` — group and class distribution, image error
- `history.json`, `run_config.json`, `checkpoint_provenance.json`
- `exploratory_test_results.json`, `MODEL_CARD.md`

**Safety and scientific limitations:** SmallCNN normal recall was 0/369 in the exploratory test. It must not be deployed or reported as successful research. The grouping is based on stripped UUID filename suffix, **not verified real-photo ancestry**; images derived from the same underlying photo can still cross groups. After observing these test numbers, any tuning on this same test invalidates it as a fresh independent test.

## Not recovered

The original 2022 TensorFlow/Keras `.h5`/`.keras` model and original 2022 training code have **not** been found. The new PyTorch checkpoint is NOT those historical weights.

If/when a fully audited model is produced, release an entirely separate checkpoint with trained architecture, seeds, training code SHA, photo-source-disjoint split, class map, training provenance, validation selection and independent metrics.

## Pretrained MobileNetV3-Small baseline (18 October 2026)

[Download frozen ImageNet-pretrained MobileNet features + trained balanced logistic head](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-v3-small-frozen-feature-v1). This checkpoint includes the pretrained feature network and fitted classifier; it is NOT the historical VGG-16 2022 model. The 5,167 train/1,120 validation images used the *older* candidate-family grouping.

[First frozen external StitchingNet test](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-frozen-first-eval-v1): MobileNet macro-F1 **0.6354**, normal recall **70.73%**, broken-stitch recall **57.79%** across 3,637 external normal/broken images. No external label tuning; this has now consumed StitchingNet as a *first untouched* benchmark and must not be repeated as independent after tuning. Still unverified photographed-photo lineage and NOT deployable.
