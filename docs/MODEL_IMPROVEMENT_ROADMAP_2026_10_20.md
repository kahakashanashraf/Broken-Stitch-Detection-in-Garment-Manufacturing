# Model improvement: evidence, bottlenecks and next controlled experiment
**20 October 2026 · Working plan (no new experiment run by this document)**

## What our actual data show

We have a released leakage-controlled split constructed from **31 filename-inferred photographic parent IDs** and 7,364 readable historical augmented / Photoshop-edited images. Parent provenance has *not* been authenticated. The existing architecture comparison and matched ablation use **6,278 development images from 26 inferred parents**, with **five prior-inspected holdout parents excluded**.

* Latest matched source-balancing ablation, source-grouped 5-fold pooled OOF: Standard class-only MobileNetV3 macro-F1 **0.7435**; source-weighted **0.7434**; source-weighted + train augmentation **0.7378**. These are development-only and not evidence of a new independent test.
* Cross-domain, frozen 0.5 threshold, StitchingNet binary normal-vs-broken N=3,637: class-only MobileNetV3 macro-F1 **0.7216**, broken recall **0.6295** (445/1,201 broken stitches missed); source-weighted model macro-F1 **0.7109**; source-weighted + augmentation macro-F1 **0.6993**, broken recall **0.7077** (351 missed, 674 false alarms among 2,436 normals).
* A separate architecture-screening MobileNetV3 checkpoint returned external macro-F1 **0.6763** and broken recall **0.4330**. Training recipes differed; this is not a controlled proof of architecture superiority or causal recipe improvement.
* StitchingNet evaluation was previously inspected. All these external numbers are descriptive **non-blind** transfer, not an untouched confirmatory set. We must not tune or choose thresholds on StitchingNet while continuing to call it an independent evaluation.

Source artifacts:
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/docs/STITCHINGNET_FINE_TUNED_EXTERNAL_2026_10_20.md
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/blob/main/docs/SOURCE_AWARE_ABLATION_RESULTS_2026_10_20.md
- https://github.com/kahakashanashraf/Broken-Stitch-Detection-in-Garment-Manufacturing/releases/tag/stitchingnet-descriptive-finetuned-comparison-v1

## Specific bottlenecks and evidence status

| Issue | Evidence | Interpretation |
|---|---|---|
| Few underlying physical sources | 31 *inferred* parents, not 7,364 independent photographs | Strong limitation on real-world photographic and synthetic-defect diversity; source genealogy remains unverified |
| Weak training budget | Earlier grouped-CV recipes used **128×128**, **2 epochs**, either a per-parent/class cap of 40 or 800 samples/class global cap | **Hypothesis**, not yet a demonstrated cause of low score. A training loss decrease between epochs 1 and 2 is not proof that longer fitting will generalize |
| Model threshold and clinical/industrial costs | VGG16 ROC-AUC 0.7940 but broken recall 24.65% at threshold .5; 905/1,201 broken images missed | Operating point can matter; select any new threshold on grouped development predictions only, evaluate once on a genuinely new source |
| Backbone size | More complex VGG16 did not outperform standard MobileNetV3 in non-blind StitchingNet macro-F1 | Larger architecture alone is not a demonstrated solution |
| Domain mismatch | Historical edited stitches vs externally acquired sewing-machine defects, different image capture and fabric patterns | Plausible but **not isolated causally** in available experiments |
| GPU device routing bug | In *grouped_finetune_cv.py*, \`run()\` detects CUDA but previously called \`fit_once()\` without \`device=device\` | **Fixed in new source:** both CV folds and development-only refit now receive detected device; legacy scores/checkpoints are immutable and unchanged |

## Immediate reproducible code repair

The GPU-routing bug is corrected in \`code/scripts/grouped_finetune_cv.py\`. New regression tests validate that both calls use the auto-detected device, tensors/model move to it, and the locked data policy and historic recipe are not accidentally changed.

**This is a compute-device wiring fix, NOT a performance improvement result.** Standard \`ubuntu-latest\` runners are CPU-only; enabling CUDA support in code does **not** allocate GPU hardware. GPU run needs a separately provisioned GPU runner. CPU and GPU operation may have numeric reproducibility differences, so compare matched seeds and avoid silently overwriting v1 releases.

## Proposed new experiment, not yet executed

To test whether under-training is responsible for weak detection, freeze a **new v2 experiment specification**, and keep v1 artifact names untouched.

1. **Development-only:** 26 inferred parents; 5-fold grouped CV with predeclared seed and identical fold assignments for all arms. Keep the previously inspected 5-parent holdout and previously used StitchingNet out of selection and threshold tuning.
2. **Backbone:** prefer one MobileNetV3 model first, rather than unconstrained architecture search. Compare a 128px/2-epoch reproduced baseline with a **224px/10–20-epoch** gradual-unfreezing recipe. Use a predeclared common training sample policy (do not covertly change sample count while claiming only resolution/epochs changed). Separate ablations if training budget, resolution, and unfreeze depth each need attribution.
3. **Selection:** group-aware validation macro-F1 as primary metric; broken recall and normal false positive rate as explicit secondary metrics. Any early stopping must be fitted using only training-fold inner groups, not by reusing the final OOF validation fold epoch-by-epoch for model choice.
4. **Safety checks:** fail closed on parent overlaps, archive digests, label inconsistencies, class leakage, overly sparse folds, and test/external download requests. Log GPU/CPU choice, all seeds, epoch history, per-group counts, checkpoint SHA, and per-fold predictions.
5. **Fresh evaluation:** obtain newly sourced physically independent normal/broken stitched-photo groups, verify license and ancestry, lock selected model/threshold first, and conduct a single truly new external test. StitchingNet-Seg derives from StitchingNet and is NOT independent. Other discovered public stitch datasets require independence audit before being described as fresh.
6. **Interpretation:** if grouped-CV or new independent-source performance does not improve, publish it as negative evidence; do not claim source-aware method or industrial readiness based on training accuracy.

**No 0.80+ score is claimed, no new weights were fit for this note, and historical v1 benchmark outputs are not retroactively modified.** The priority is a truthful matched-protocol experiment, not score chasing on already seen external labels.

## Venue and integrity gates

The thesis data's derivative source chain and right to republish third-party imagery remain human-unverified; original 2022 source code/weights unrecovered. A research draft's results must not be silently replaced with training outcomes or development-only OOF metrics. ICEEICT authorship, AI-use and image rights policies require explicit human review before submission.
