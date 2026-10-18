# Reconstruction technical verification — 17 October 2026

## What was actually run in this workspace

- Syntax/import test: `python -m broken_stitch.cli --help` — PASS.
- Unit/integration suite: `python -m pytest -q` — **8 passed**.
- CNN one-epoch smoke training on 20 toy image groups, with 2 image variants/group — PASS.
- Trained CNN checkpoint → held-out split evaluation — PASS.
- Two trained CNN checkpoints → probability ensemble evaluation — PASS.
- Group split with same parent between normal/edited variants — PASS.
- Rejection tests: blank source_id, source crossing splits, cross-split exact duplicates, modified evaluation manifest — PASS.
- VGG16 feature/linear-head forward smoke, no pretrained weights — PASS.
- VGG16 one-epoch smoke train and checkpoint reload/evaluation on locally generated toy images — PASS.
- VGG smoke marked `model_weights_provenance_ok = False`, `publication_ready = False` — PASS.

## Hardware and libraries used for smoke tests
- CPU: PyTorch 2.10.0+cpu, torchvision package paired with installed torch.
- TensorFlow/Keras is **not** installed, and the 2022 thesis training was not rerun in its original framework.
- Original data `dataset.zip` is present in user's Drive at approx 2.15 GB but was not downloadable through the connected Drive API (hard limit 256 MiB). No authentic stitch-image result has been computed here.

## Unresolved / NOT verified

1. Content, lineages, class counts, actual image quality and legal reusability of real original dataset.
2. Original TensorFlow architecture; exact VGG16 fine-tune stages; data processing settings and historical best epochs.
3. Numerical agreement with reported 85.35%, 91.48%, 93%, 94.91% etc. It would be academically improper to expect identical historical metrics when the code and splits are unrecoverable.
4. Pretrained VGG weight download and official ImageNet state_dict provenance on a real GPU. No 500 MB VGG weights were downloaded.
5. Training resource consumption on 9,134+ images and model throughput on Kaggle GPU.
6. Sensitivity to near-duplicate transformed versions not caught by SHA; lineage groups remain mandatory.
7. Statistical uncertainty across independent photographed seam sources; robust validation is still required before replacing paper results.

## Verdict
- **GO for code review, dataset preparation, and GPU pilot once source IDs audited.**
- **NO-GO for claiming reproduced original results or submission-cleared new experiments.**

Historical reported values remain in `examples/HISTORICAL_THESIS_RESULTS_DO_NOT_COMPARE.json` as editorial context only.