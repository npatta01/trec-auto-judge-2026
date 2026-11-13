# TREC 2026 Auto-Judge: Meta-Evaluation Results for NP Labs

This package contains the meta-evaluation results of the auto-judge methods that your
team submitted to the TREC 2026 Auto-Judge track. It lists your own methods only.

## Files

- `rag26-scores.tsv` and `ragtime26-scores.tsv` list every method of your team with its
  `tauap_b` score on the respective host track, sorted from best to worst.
- `rag26-plot.png` and `ragtime26-plot.png` show one column per run tag at its best
  method, colored by LLM family.

A file is present only for a host track with results for your team.

## Methods

A method is one combination of run tag, LLM, and evaluation measure. One run can report
several evaluation measures, and each of them is evaluated as its own method. The
columns `runtag`, `llm`, and `eval_measure` identify the method.

## Meta-Evaluation Methodology

Every method ranks the RAG systems of a host track. We compare this ranking with the
official leaderboard of the host track and report the rank correlation `tauap_b`. This
is the AP rank correlation in its tie-aware b variant, which gives disagreements near
the top of the leaderboard more weight. A value of 1 means that both rankings agree.

The scores of your method are aggregated over exactly the topics that were manually
annotated for the official leaderboard. The correlation is computed over the RAG
systems in the official leaderboard. A system in the official leaderboard that your
method did not score enters the correlation with a score of 0.

### TREC 2026 RAG (`rag26`)

- Official measure: `bt_score`, the Bradley-Terry rating fitted from human side-by-side
  preferences, combined across both annotation batches
- Annotated topics: 42 (17 in batch 1 and 25 in batch 2)
- RAG systems in the official leaderboard: 41

The TREC 2026 RAG track overview notebook describes the annotation process and the
measure.

### TREC 2026 RAGTIME (`ragtime26`)

- Official measure: `f1_llmfilled` of the mostly human assessment
- Annotated topics: 17 mostly human topics
- RAG systems in the official leaderboard: 27

The TREC 2026 RAGTIME track overview notebook describes the annotation process and the
measure.

## Top-5 Markers

For each host track we compile three leaderboards of all participating teams. Each
leaderboard lists every run tag once, represented by its best method.

- `top5_overall` refers to the leaderboard over all LLMs.
- `top5_gpt-oss-120b` refers to the leaderboard over methods that use the LLM `gpt-oss-120b`.
- `top5_jev` refers to the leaderboard over methods that use a Jev LLM.

A marker column contains `X` for the method that represents a run tag among the best
5 of that leaderboard, and is empty otherwise. The markers don't state the position
within the best 5.
