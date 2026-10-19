# Independent broken-sewing-stitch external test: candidate source screening (19 Oct 2026)

**Scope restriction:** Need genuinely distinct acquisition source, image-level **NORMAL SEWING STITCH vs BROKEN SEWING STITCH**, no screenshots/crops from StitchingNet or its derivatives; no overlap with the original thesis Photoshop transformations. Same-label benchmarks must be frozen before scoring; no post-test tuning.

| Public candidate | Source | Why not suitable as unseen broken-sewing-stitch binary test |
|---|---|---|
| StitchingNet-Seg (2026) | https://github.com/hyungjungkim/StitchingNet-Seg | **Derived from original StitchingNet**. Its images overlap or descend from already-tested data; not independent |
| BDFD9 (2026) | https://data.mendeley.com/datasets/z49cc7zvfc/1 | Class `DropStitch` refers to knit/weave fabric fault, **not broken sewing seam**. Dataset lacks directly matched normal/broken sewing stitch labels |
| Fabric Anomaly Recognition (2025) | https://data.mendeley.com/datasets/h2ytgy2pmb/1 | Classes `Hole, Broken End, Oil Stains`; `Broken End` is a weaving/yarn fault rather than broken sewing thread |
| Fabric Defect (2025) | https://data.mendeley.com/datasets/y62b4pfyz2/1 | Contains `Sewing-error`, but not independently labelled **broken vs normal sewing stitch** subset; some images augmented; access may require request |
| TSFabrics (2026) | https://www.nature.com/articles/s41597-026-06748-9 | Knitting-machine dropped/missed stitch and other textile flaws, not the same task |
| StitchingNet on Hugging Face / Mendeley / Kaggle | https://data.mendeley.com/datasets/6tdthsjgfc/1 | Replicas of **already examined** StitchingNet, cannot be a new independent benchmark |

**Result of search:** No verified, downloadable, label-compatible and source-independent new external *broken sewing stitch* binary benchmark established by this screening. **No numeric new-external result** is claimed. Resist remapping raw-fabric 'broken end' or 'drop stitch' into broken sewing seam.

**Acquisition requirement for a real external test:**
- Collect paired / independently labelled normal and genuinely broken (not Photoshop-generated) sewing seams from other cameras, fabrics and operators (with consent/rights), ideally at least 5 independent garments per condition and multiple lots; do not equate image samples with independent garment units.
- Record original file hash, camera/site/date, garment/physical source ID, fabric type, stitching condition, defect confirmation by ≥2 reviewers, acquisition permissions and human-readable label protocol.
- Freeze label mapping, exclusions, groups, classifier checkpoint SHA and threshold 0.5 **before unsealing scores**; keep all descendants of one photograph/garment in same split; verify no duplicate/near-duplicate across domains.
- One evaluative run only. Never reuse the existing StitchingNet split as *untouched* for new models after observing it.

This is a source-screening artifact, not proof that no such dataset exists anywhere.
