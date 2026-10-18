# ICEEICT 2027 — anonymous IEEE working manuscript

Files in this directory:

- [`main.tex`](main.tex): editable IEEE manuscript describing *historical* thesis results and a **proposed, unexecuted** source-disjoint and StitchingNet validation protocol.
- [`ICEEICT2027_BrokenStitch_Research_Extension_v2_Anonymous.pdf`](ICEEICT2027_BrokenStitch_Research_Extension_v2_Anonymous.pdf): 6-page PDF compiled in GitHub Actions with original thesis images.
- [`figures/`](figures/): ten historical figures extracted from the unpublished thesis; four are currently referenced by the TeX.
- [`PAPER_SHA256.txt`](PAPER_SHA256.txt): SHA-256 of the compiled PDF.
- [`restore_figures.py`](restore_figures.py): option to restore a local figures ZIP if needed.

Compile with IEEEtran and TeX packages (graphicx, booktabs, cite, amsmath, hyperref, etc.):

```bash
cd paper
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

**Not a submission-ready new-results paper.** No new real-data training, source-disjoint performance or cross-dataset evaluation has been run. Figures and thesis dataset are public at the owner's request but provenance/licensing have NOT been independently cleared. GitHub repository visibility does not grant a reuse license. Anonymity of a public author-associated repository must be considered under the venue's double-blind review rules.
