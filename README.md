# Broken Stitch Detection in Garment Manufacturing

[![Unit tests](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/workflows/tests.yml/badge.svg)](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/workflows/tests.yml)

**Research project for an ICEEICT 2027 working paper — NOT a verified deployment model or publication-ready result.**

This repository reconstructs the *families of experiments* described in an unpublished 2022 undergraduate thesis on normal-vs-broken sewing stitches. It adds **source-disjoint dataset handling**, reproducible evaluation scaffolding, and a planned, untouched external benchmark using StitchingNet. The 2022 original code/checkpoints are unavailable; this code is newly implemented in PyTorch, rather than an exact reproduction of TensorFlow/Keras architecture or weights.

## Start here

| Component | Location | What it contains |
|---|---|---|
| Rebuilt training code | [`code/`](code/) | CNN, frozen/finetuned VGG-16, data filters, ensemble, metrics, CLI |
| Regression and safety tests | [`code/tests/`](code/tests/) | Synthetic-data smoke checks; no measured garment accuracy |
| Colab audit notebook | [`notebooks/`](notebooks/) | Authenticated Drive ZIP inventory; lineage must be verified by humans |
| Research protocol | [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) | Source-disjoint split, internal & external test rules |
| Evidence and open blockers | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) | Explicit completed/pending work |
| Paper LaTeX source | [`paper/`](paper/) | IEEEtran working paper, anonymous, **prospective** experiments marked as unrun |
| Data rights and licenses | [`docs/DATA_AND_LICENSE.md`](docs/DATA_AND_LICENSE.md) | Why raw images are not redistributed here |

## Local installation and tests

Python 3.10–3.12. Install compatible PyTorch and torchvision packages for your hardware first. From repository root:

```bash
python -m pip install -r code/requirements.txt
python -m pip install -e code/
cd code
python -m pytest -q
python -m broken_stitch.cli --help
```

At repository upload (18 October 2026), **12/12 tests passed locally**. GitHub Actions separately runs them on every push. These are **software functionality tests on synthetic examples** and do not demonstrate any real dataset accuracy.

## Dataset experiment gates

1. Audit the team's original ~2.15 GB ZIP. Identify each original photograph and all Photoshop/augmentation descendants by a *human-verified* `source_id`. Never guess parent IDs.
2. Create group-exclusive train/validation/test partitions before on-the-fly augmentation, verify class labeling and exact/near-duplicate leakage risks.
3. Train CNN and officially ImageNet-pretrained VGG-16 on the same frozen splits, select by validation metrics, then evaluate the untouched internal test.
4. Download [StitchingNet](https://github.com/hyungjungkim/StitchingNet) under its **CC BY-NC 4.0** terms. Use **only normal and broken-stitch** classes for the external binary evaluation. Do not fit or tune on that external test.
5. Update manuscript results only after actual measurements and audit approval; report macro-F1, balanced accuracy, per-class recall, AUROC, average precision and confusion matrices.

See [the detailed CLI instructions](code/README.md).

## What is and isn't a research result

- **Historical thesis observation:** final VGG-16 **94.91% reported validation accuracy**, 98.71% reported training accuracy. Not independently reproduced.
- **Current new training on original images:** **not run**.
- **External StitchingNet evaluation:** **not run**.
- **Source-disjoint generalization finding:** **not established**.
- **Code tests:** 12 passed on synthetic fixtures only.

The thesis mentions 181 initial images, but its 9,134 / 10,000 / 20,000 augmented-image counts are inconsistent. Our code must not silently resolve these discrepancies.

## ICEEICT manuscript and figures

The paper source is [`paper/main.tex`](paper/main.tex) with background, methods, historical measurements and a **proposed** new validation protocol. It is **not submission-cleared**. The original historical thesis figures are kept outside this public repo pending provenance and image-use permissions; [`paper/restore_figures.py`](paper/restore_figures.py) can extract approved assets from a separately provided `figures_original_thesis.zip`. The `.tex` also compiles with visible placeholders in their absence.

**Double-blind caution:** A public repository bearing author attribution can make an anonymous manuscript deanonymizable. Verify the venue's policies before citing or releasing this repository as supplementary submission material.

## Provenance and reuse

Raw thesis images and model weights are intentionally excluded, as rights and provenance have not been certified. Attribution or licensing of third-party StitchingNet data remains with its publishers. No source-code license has yet been explicitly approved by the research authors; do not presume public visibility authorizes third-party reuse.
