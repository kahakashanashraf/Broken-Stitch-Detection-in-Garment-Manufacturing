# Broken-Stitch Detection on Unseen Fabrics — ICEEICT 2027 working paper

**Current manuscript: v13 (21 October 2026).** The paper studies group-based testing, edge fusion and Grad-CAM analysis. This is a **research working draft**, not a cleared submission. Human coauthors must verify the data rights, citations, manuscript content, anonymous submission rules and AI-use policy.

## Latest paper files

- **[IEEE PDF, v13](ICEEICT2027_BrokenStitch_v13_Introduction_Flow.pdf)** — 5 A4 IEEE two-column pages; 20 cited references; model and experimental figures inside the paper.
- **[Editable LaTeX, v13](ICEEICT2027_BrokenStitch_v13_Introduction_Flow.tex)** — contains an Introduction with a continuous problem → gap → positioning → RQ1–RQ4 → contributions flow.
- **[Reproduced figures](figures_v13/)** — the actual held-out-fabric F1 chart and deterministic colorized Grad-CAM evidence from released grayscale maps. These are qualitative, not verified defect-localization masks.
- **[Figure generator](reproduce_v13_figures.py)** — verifies both original GitHub release PNG SHA-256 digests before regenerating the figures.
- **[GitHub Actions build](../.github/workflows/build-evidence-v13-paper.yml)** — compiles and checks the PDF, then publishes it together with its figures.

## What the manuscript reports

On StitchingNet's normal/broken subset (3,637 images, 11 fabric settings), five-fold held-out-fabric training yields **macro-F1 0.7969** (ImageNet initialization) and **0.8089** (added edited-image pretraining). A matched hybrid with a small Sobel edge branch reaches **0.8082**, and 30 epochs do not improve macro-F1 relative to 6 epochs in the matched training-budget run. Grad-CAM compares deliberately selected cases; do not infer population recall or precise localization from them. The separate image-lineage audit shows the sensitivity of development scores to related-image splitting.

**Evidence status:** StitchingNet was inspected previously, so the later experiments are retrospective/descriptive rather than independent blind tests. All models in fabric-held-out CV train on StitchingNet labels from *other* fabrics. Do not describe that score as zero-shot external transfer or factory performance.

## Compiling the paper

Clone the repository, install an IEEEtran-capable LaTeX distribution with the listed TeX packages, and run:

```bash
cd paper
pdflatex -halt-on-error -interaction=nonstopmode ICEEICT2027_BrokenStitch_v13_Introduction_Flow.tex
pdflatex -halt-on-error -interaction=nonstopmode ICEEICT2027_BrokenStitch_v13_Introduction_Flow.tex
```

The figures are checked into `figures_v13/`. To reproduce them afresh, use the workflow noted above; it downloads the original released Grad-CAM evidence and checks source SHA-256 hashes.

## Older archived versions

Older files such as `evidence_audit_v4.tex` and `main.tex` are archived working drafts. Their earlier stated results, methodologies, and unfinished research plans must **not** be treated as the current manuscript. The v13 files linked above are the latest editable conference-paper version.

**Submission caveat:** Ensure original-image ownership and publication permissions, all author approvals, and the conference AI-use policy are resolved before submission.
