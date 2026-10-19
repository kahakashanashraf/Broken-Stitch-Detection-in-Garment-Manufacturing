# Photographed-source lineage: what is known vs unknowable yet

## Computational evidence verified
- Original thesis 2,146,784,353-byte archive has 7,365 image files; 7,364 decode and one is corrupt.
- The filename graph contains 31 `actual_original_XX` augmentation families, 151 `faulty_original_XXN` families, 0 unmatched; the inferred rule maps each faulty family to an `XX` normal candidate photo.
- The original image-level train/val/test directories contain augmented derivatives crossing splits. Old folder-level independent performance is not reliable.
- Ranked computer-vision similarities on candidate normal/faulty family representatives support several matching pairs. This is not exact edit-history provenance.

## What cannot be certified from uploaded files
- Full set of unedited parent photos, Photoshop PSD/edit-history chains, actual acquisition order/source IDs, evidence that a synthetic image of class faulty was derived precisely from the normal image with the same stem. Names and visual similarity do not independently prove this.
- True photograph independence across possible repeated camera shots, even after 31-key grouping.
- Ownership/licensing of every input photo and permission to redistribute.

## Human provenance audit needed
For all 31 parent candidates, obtain one unaltered source photograph (RAW/JPG/PNG) plus capture-site/garment ID, and editor/source file provenance for each of the 151 synthetic faulty variants. A reviewer checks the source and target visually side-by-side, verifies edited location and label, signs off the relation. Preserve original hash and reviewer/time. Unresolved pairs receive status `unknown`, not `verified`.

Model evaluation on the 31-inferred-parent holdout **may be descriptive** but is never described as human-verified independent holdout. The five previously uninspected photo-parent candidates become an *inspected* test as soon as scores are released; future tuning on those scores contaminates that test.

Original inferred mapping and reviewer candidate images: [audits/photographed_source_inference](../audits/photographed_source_inference), [audits/photo_lineage_review](../audits/photo_lineage_review).
