# Leakage-controlled broken-stitch dataset v2 (19 October 2026)

## Download

**Verified public Release:** https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/leakage-controlled-dataset-v2-inferred-31-parents

Download all three assets, `train.zip`, `val.zip` and `test.zip`, along with `clean_manifest.csv`, `quality_report.json`, `evaluation_representative_candidates.csv`, and `README.txt`. Extract the three ZIPs into the **same clean folder**, resulting in `train/actual`, `train/faulty`, `val/actual`, `val/faulty`, `test/actual`, `test/faulty`.

## What is fixed

Original 2022 image-level splits placed derivatives from all 182 UUID-stripped edit families into multiple partitions. We discarded the **old split labels**, not the original intact data, and used the established **31 inferred photographed-source parent IDs**, which combine corresponding normal and Photoshop-faulty edit families. Each complete candidate parent group is assigned to exactly one partition.

| Split | Actual | Faulty | Total | Inferred photo parents |
|---|---:|---:|---:|---:|
| Train | 1,878 | 3,141 | 5,019 | 21 |
| Validation | 433 | 826 | 1,259 | 5 |
| Test | 422 | 664 | 1,086 | 5 |
| All | 2,733 | 4,631 | **7,364** | **31** |

Exactly **one corrupted original normal image** was excluded. All 7,364 remaining historical images were image-opened and verified, CRC validated, and copied to the new split archive. The workflow checks **zero inferred-parent overlap across splits** and **zero exact-byte image hashes crossing splits**; each archive was validated by ZIP integrity and has a recorded SHA-256 checksum in `quality_report.json`.

Build workflow: https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/actions/runs/39275264689

Original dataset digest: `e7aed1355a712c14bbe6560fefbb3dfab546a38970f836d129f5721e0b046979`

Canonical parent-manifest digest: `769d10dc6f6651f24d655ae0ed6bc1f780a50f24882c0b7fd5073fec1a51d2cb`

## Honest scientific limitations

**This is a validated split correction, not full photographic provenance proof.** The 31 parents and their 151 Photoshop-variant links are **inferred from filenames plus candidate visual matching**; unaltered raw photos, Photoshop source-edit histories and rights/consent logs have not been recovered. Exact duplicate checking does not prove absence of cross-parent semantic near duplicates.

The dataset still contains **previously augmented and synthetically edited images in validation and test**, not newly acquired raw natural images. `evaluation_representative_candidates.csv` selects one EXISTING derived image for each edit family; these representatives are **not guaranteed raw originals**. Do not represent v2 as a newly unaugmented, independently sourced real-stitch dataset. Apply no *additional* augmentation to validation/test.

**The five-parent test set has already been evaluated** for the earlier trained 31-parent MobileNet; its score is no longer a fresh untouched test for any model subsequently tuned using that result. This rematerialization did **not** retrain a model, and it did **not** compute new accuracy. Historical 94.91% remains a separate, unreplicated validation report.

The [existing 31-parent MobileNet checkpoint](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/mobilenet-photo-parent-inferred-v1) was trained using these **same manifest assignments** from the original archive, so do not label it a second or new model. The one-shot [prior holdout results](https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/frozen-photo-parent-holdout-first-v1) remain historical to that fixed protocol, not independent evidence obtained by publishing these cleaned ZIPs.

## Reproducibility

Script: [`code/scripts/build_clean_grouped_dataset.py`](../code/scripts/build_clean_grouped_dataset.py)

Workflow: [`.github/workflows/build-clean-grouped-dataset.yml`](../.github/workflows/build-clean-grouped-dataset.yml)

Data-rights note: The original archive may include internet-sourced and manipulated images. Public release does not guarantee permission to reuse all images in downstream publications.
