# Raw data, figures and licensing status

- **Original thesis dataset** is now distributed as `dataset.zip` (2,146,784,353 bytes) through [GitHub Release `original-thesis-dataset-v1`](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/original-thesis-dataset-v1). Metadata with SHA-256 and ZIP-integrity validation is included alongside the release asset.
- **Historical thesis figures**: 10 extracted image files are available in [`paper/figures/`](../paper/figures/). Images were extracted from the team's thesis material and were posted at the repository owner's express request.
- **No actual historical or newly trained weights** have been recovered. See [`models/README.md`](../models/README.md).
- **Provenance and legal status remain unverified.** The thesis describes Photoshop-generated defects and image sourcing that may include third-party content. Publication of these archives is not a declaration of sole ownership, consent from every depicted/source author, public-domain status or a blanket redistribution license. Request permission before third-party reuse.
- External public [StitchingNet](https://github.com/hyungjungkim/StitchingNet) carries its publishers' CC BY-NC 4.0 license. It has **not** been copied into this repository or used for new test metrics.
- Rebuilt PyTorch code is not the historical TensorFlow/Keras source. No general open-source licensing grant has been confirmed for this repository.

## Scientific use warning

ZIP CRC/size validation alone does not prove original-image identity, non-overlapping training/test splits, dataset label validity, lack of near-duplicate leakage or generalizability. Follow [research protocol](RESEARCH_PROTOCOL.md) before reporting accuracy.
