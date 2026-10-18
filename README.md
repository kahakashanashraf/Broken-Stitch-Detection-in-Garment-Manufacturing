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
| Original/reconstructed trained model weights | [`models/README.md`](models/README.md) | **Unavailable — no genuine checkpoints were found** |

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

**12 automated tests passed** on synthetic fixtures in the linked GitHub Actions CI; this is software validation, not a new accuracy result.

## Next scientific steps

1. Audit archive images, labels, generated defects, original photograph and derivative source IDs; reconcile 181 / 9,134 / 10,000 / 20,000 counts.
2. Create source-disjoint training, validation and never-tuned internal-test partitions.
3. Rebuild and train models on verified source groups and report class-wise metrics.
4. Run frozen checkpoints on the strictly normal/broken subset of [StitchingNet](https://github.com/hyungjungkim/StitchingNet), observing its CC BY-NC 4.0 terms.
5. Update the draft paper **only after** results and image/author/venue-policy review.

### Anonymity and intellectual-property notice

This is a publicly attributable repository and may compromise strict double-blind submission if linked from the paper. Human authors must confirm image provenance, ownership, publishing permissions, and ICEEICT policies. **No blanket dataset, image or code license has been granted merely by making these files public.**
