# ICEEICT 2027 working paper (NOT submission cleared)

`main.tex` is an **anonymous IEEEtran working manuscript** that distinguishes 2022 thesis results from unperformed experiments. It must not be cited as new source-disjoint or external benchmark evidence.

To compile on a system with `IEEEtran`, `booktabs`, `amsmath`, `graphicx`, `hyperref` and other standard TeX packages:

```bash
# From repository root:
python paper/restore_figures.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`restore_figures.py` extracts the original thesis figures from `figures_original_thesis.zip` into `paper/figures/` if the archive exists. These are **historical thesis figures**, not newly generated benchmarks. Only figures appearing in the actual manuscript are used. The ZIP provenance is the original thesis document; confirm image rights before sharing publicly.

Publishing a manuscript in a publicly attributable repository could weaken the anonymity of a double-blind submission. Do not put author identities inside the submitted PDF, its metadata, or identifying self-citations. Independently verify conference rules and obtain author approval.

The training code and raw dataset are not included in the PDF; dataset access is discussed in `docs/CURRENT_STATUS.md`.