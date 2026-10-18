# Current research status — 2026-10-18

| Item | Status | Evidence |
|---|---|---|
| 2022 historical best VGG-16 validation accuracy | Reported **94.91%** | Unpublished thesis; not independently reproduced |
| Rebuilt CNN/VGG-16 PyTorch code | Implemented approximation | `code/` (NOT exact 2022 source recovery) |
| Automated code tests | **12 passed locally** (2026-10-18) | `python -m pytest -q` on available synthetic test cases |
| Original 2.15 GB ZIP audit | **Not completed** | Connector download cap prevented access |
| Verified parent/source image lineage | **Not completed** | Requires image provenance inventory and manual review |
| Source-disjoint real-data training | **Not run** | Requires audited lineage and GPU |
| StitchingNet binary external test | **Not run** | Official dataset licensing and image verification pending |
| 2027 ICEEICT 6-page paper draft | Working draft / protocol | `paper/main.tex` |
| Conference acceptance/submission approval | **Not established** | Requires results, author review, image rights, venue policy |

**Reproducibility rule:** Do not use the code's synthetic smoke-test outputs or the historical validation number as fresh scientific evidence. The 2022 reported 20,000/10,000/9,134 counts must be reconciled rather than silently replaced.

**Caution for double-blind review:** this public repository and its owner identity may make the authors discoverable. Confirm conference anonymization rules before citing this repository in the review manuscript.