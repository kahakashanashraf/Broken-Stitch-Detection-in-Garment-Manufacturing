> **Wording erratum (20 October 2026):** The original Release v1 JSON incorrectly described the training subset as *per-parent capped*. The actual trained A/B/C runs all used the **same globally class-stratified 800 normal + 800 faulty training images per fold, without a parent-specific cap**. Only class-vs-source loss weighting and online augmentation changed. All scores, predictions and paired bootstrap intervals below are unchanged. See [corrected structured report](source_aware_ablation_results_v1_corrected.json).

# Controlled source-weighting and train-only augmentation ablation

**Exploratory development-only 5-fold grouped-CV results**; 26 inferred photo parents, 6,278 augmented image records.

| Arm | Grouped OOF macro-F1 | Balanced accuracy | Faulty recall | Parent bootstrap descriptive 95% |
|---|---:|---:|---:|---|
| class_only | 0.7435 | 0.7526 | 0.7588 | [0.7022, 0.7830] |
| parent_class | 0.7434 | 0.7498 | 0.7731 | [0.7022, 0.7835] |
| parent_class_aug | 0.7378 | 0.7545 | 0.7172 | [0.6945, 0.7792] |

## Paired differences (macro-F1, same OOF images and candidate parents)
- **parent_class_minus_class_only**: -0.0000, descriptive interval [-0.0098, +0.0081]
- **parent_class_aug_minus_parent_class**: -0.0056, descriptive interval [-0.0239, +0.0156]
- **parent_class_aug_minus_class_only**: -0.0057, descriptive interval [-0.0222, +0.0125]

**Caveat:** Within identical global class-stratified sampling, this ablation tests **additional loss weighting by parent**, not the benefit of parent-grouped splits in general. Grouped splits were held fixed throughout.
The previously inspected five-parent holdout and StitchingNet were NOT accessed. This is not a new external result.
Inferred photographed-source identities have not been human verified.
