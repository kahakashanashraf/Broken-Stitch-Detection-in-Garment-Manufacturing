# ICEEICT 2027 experiment protocol — October 18 update

## What is and is not complete

- **Known:** the original thesis reports 181 initial records (31 normal, 150 faulty) and 94.91% final VGG-16 *validation* accuracy. The source/derivative lineage has not been proven.
- **Verified externally:** [StitchingNet](https://github.com/hyungjungkim/StitchingNet), 2025, contains 14,565 images in 11 classes. Its original publication tabulates 2,436 normal and 1,201 broken-stitch images. Dataset license: CC BY-NC 4.0.
- **Not done:** thesis ZIP image inventory, sample provenance audit, GPU training, external inference, new metric generation, subgroup/bootstrapped CI, or novel-result validation.

## RQs and protocols

1. **RQ1 (split sensitivity):** parent-source-disjoint vs random image-level split diagnostic, identical models and predefined seeds; latter only as negative control. Never report random split as primary performance.
2. **RQ2 (synthetic-to-real):** train/validate/select solely on original thesis data, freeze checkpoint and 0.5 threshold, evaluate once on untouched StitchingNet normal/broken subset. Other nine defective classes are excluded.
3. **RQ3 (baselines):** compare custom CNN and ImageNet-pretrained VGG-16 on exactly the same splits with validation macro-F1 model selection. Compact MobileNetV3 baseline desirable but pending implementation.

## Must-have gates

- **G0 — Provenance:** original archive opened, exact inventory/labels/rights, authentic versus Photoshop defects established, parent IDs audited; STOP if grouping unknown.
- **G1 — Split:** all variants confined to one source group, train-only augmentation, inspect near-duplicates. A source could contain *both* labels.
- **G2 — Baselines:** rerun >= 3 seeds if feasible; report macro-F1, balanced accuracy, class recalls, confusion matrix, PR-AUC and ROC-AUC with sample counts and uncertainty. Tune only on validation.
- **G3 — External:** review StitchingNet directories and license, verify 2,436/1,201 image counts post-extraction, freeze model, SHA overlap audit, one final evaluation. Report domain shift caveat.
- **G4 — Paper:** replace historical-only emphasis with **measured** new results (if available), check figure sourcing, literature, authorship, similarity and LLM-use policy, cite exact code/config/weights. If unavailable by venue deadline, mark **NO-GO for novel-results submission**.

## Workbook of planned output (blank until experiment)

| Model | Split | n images | Macro F1 | Balanced Acc | Broken Recall | PR-AUC | Note |
|---|---|---:|---:|---:|---:|---:|---|
| CNN | Source-disjoint thesis | — | — | — | — | — | NOT RUN |
| VGG16 | Source-disjoint thesis | — | — | — | — | — | NOT RUN |
| Frozen winner | StitchingNet binary | 3,637 expected* | — | — | — | — | NOT RUN |

\*Source publication total, not inventory count confirmed from the downloaded dataset.

## 2026 literature context
- Kim et al. 2022: DOI 10.1016/j.eswa.2021.116014 — VGG feature maps, synthetics and 92.3% on their own test. Not directly rankable.
- Jung et al. 2025: DOI 10.1093/jcde/qwaf037 — StitchingNet and deep transfer learning.
- Kim et al. 2026: DOI 10.1038/s41598-026-73203-5 — PatchCore + fine-tuned ResNet on sewing anomalies.
- Park et al. 2026: DOI 10.1038/s41597-026-08496-2 — StitchingNet-Seg localization dataset.

**ICEEICT deadline:** October 25, 2026, official https://iceeict.mist.ac.bd/authors/important-dates/ . Review requires <= 6 A4 IEEE pages, <=3 MB PDF, anonymized, original research. ChatGPT preparation/LLM-use declaration is required by the organizers. Manuscript intellectual work must remain authors' responsibility; check policy before submission.