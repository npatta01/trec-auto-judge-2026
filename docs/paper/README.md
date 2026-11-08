# TREC AutoJudge 2026 notebook paper

The paper describes the final Pairwise, Evidence and Combined submissions for RAG and RAGTIME. Worked scores are illustrative, not official evaluation results.

- [Read the paper](autojudge2026.pdf)
- [LaTeX source](autojudge2026-draft.tex)

## Build

Requires `latexmk` and a TeX Live installation with the packages imported by the source, including TikZ, Latin Modern, microtype and tabularx.

```sh
cd docs/paper
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build autojudge2026-draft.tex
cp build/autojudge2026-draft.pdf autojudge2026.pdf
```

The two PNG assets in `figures/` are required. Keep generated LaTeX auxiliary files in `build/`; do not commit them. The checked-in PDF was inspected page by page for readable labels, table spacing and complete references.
