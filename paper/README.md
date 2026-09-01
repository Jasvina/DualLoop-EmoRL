# Paper Source

`main.tex` is the public preprint source. `Figures/` contains the final assets
referenced by the manuscript. `supplement/` preserves the technical supplement
and recovered appendix source used during submission preparation.

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Compile the technical supplement directly from `paper/supplement/`:

```bash
pdflatex main.tex
```

`TechnicalSupplement.tex` is retained as the identically rendered archival
entry point used during submission preparation.
