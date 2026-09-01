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

Compile the technical supplement from `paper/supplement/` while exposing the
parent directory to TeX:

```bash
TEXINPUTS=..: pdflatex TechnicalSupplement.tex
```
