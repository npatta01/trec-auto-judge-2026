# NP Labs organizer results · November 13, 2026

The November 13, 2026 organizer email to NP Labs contained Auto-Judge
meta-evaluation results for the RAG and RAGTIME host tracks.

## Best reported method per host track

| Host track | Run tag | Measure | `tauap_b` | Overall top five |
| --- | --- | --- | ---: | --- |
| RAG | `ntp-combined-v1` | `RAG_REQUEST_COVERAGE_PROXY` | 0.573318 | No marker |
| RAGTIME | `ntp-combined-v1` | `RAGTIME_SENTENCE_SUPPORT_PROXY` | 0.584562 | Yes |

All 19 method rows (11 RAG, 8 RAGTIME) list `gpt-6-luna`. The only top-five
marker in the package is the RAGTIME overall marker above. The organizer marks
one best method per run tag in each leaderboard and the marker gives top-five
membership without an exact position. The package covers NP Labs only, so it
cannot establish exact ranks or gaps to other teams.

The standalone `ntp-pairwise-10-v1` reports 0.505701 on RAG and −0.009917 on
RAGTIME. Its score matches the pairwise measure reported under `ntp-combined-v1`
on each track. The package has no standalone evidence run tag. The
[submission ledger](submission-ledger.md) records the TIRA receipts
under the track-specific submission names.

## How to read the scores

`tauap_b` measures how closely each Auto-Judge method reproduces the host
track's official ranking, weighting disagreements near the top more heavily.
RAG uses `bt_score`, a Bradley–Terry rating fitted to human preferences
over 42 annotated topics and 41 systems.
RAGTIME uses mostly-human `f1_llmfilled` over 17 annotated topics and 27 systems.
A missing official system is scored as zero before the correlation is computed.

## Source files

- [Organizer README](organizer-results/2026-11-13/README.md)
- RAG [score table](organizer-results/2026-11-13/rag26-scores.tsv) and [plot](organizer-results/2026-11-13/rag26-plot.png)
- RAGTIME [score table](organizer-results/2026-11-13/ragtime26-scores.tsv) and [plot](organizer-results/2026-11-13/ragtime26-plot.png)

The circularity archive is stored locally under `results/organizer/2026-11-13/`.
