# TREC AutoJudge 2026 talk

[HTML deck](talk.html) · [PDF slides (30 pages)](talk.pdf)

The PDF includes the complete talk and appendix, with one presentation-layout slide per page and selectable text.

This deck follows the notebook paper’s illustrative four-day-workweek answer through Pairwise, Evidence, and Combined. The main slides show the pairwise judgment counts and how Bradley–Terry turns votes into scores, the shared checklist, coverage and citation judgments, a worked grade, our headline organizer results, and a two-system example of how citation checks change a ranking. The appendix contains the comparison schedule, the detailed example and Bradley–Terry fit, the host-track metrics and citation-order example, the segment-level grounding example, the score formulas, and the organizer’s rank-agreement scores for all 17 reported measures. The workweek answers, judgments, and scores are illustrative. The November 13, 2026 organizer package covers NP Labs runs and identifies our RAGTIME run as top five overall. The layout and palette follow the [final TREC RAG talk](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-talk.html). The illustrations in `art/` were generated for the AutoJudge deck.

- [Submitted notebook paper](../autojudge2026.pdf), especially §§2–5 for the example.
- [Organizer results and meta-evaluation](https://github.com/npatta01/trec-auto-judge-2026/blob/main/docs/organizer-results-2026-11-13.md), with the source [RAG table](https://github.com/npatta01/trec-auto-judge-2026/blob/main/docs/organizer-results/2026-11-13/rag26-scores.tsv) and [RAGTIME table](https://github.com/npatta01/trec-auto-judge-2026/blob/main/docs/organizer-results/2026-11-13/ragtime26-scores.tsv).

From this directory, run the following with Quarto 1.9.

    quarto render talk.qmd

Open `talk.html` in a browser. Click **Edit slides**, choose a slide, and edit its text, table cells, chart values, or speaker notes. **Download HTML** saves the whole deck as a standalone, editable HTML file. Changes live in the current browser tab until you download a copy. Edit the Quarto source to change the slide order or layouts.

Use the arrow keys to present and S for speaker notes. Click the theme button to cycle through Auto, Dark, Light, and Storybook. On narrow screens the slides appear as a scrolling reader.
