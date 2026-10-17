# PROGRESS

Running log. Final state when the overnight run stopped.

## Phase 0 - setup (DONE)
Project at `company_debate/`, venv + packages installed, network to SEC/HuggingFace/Yahoo confirmed
working on this machine. Whiteboard PDF read and matches the brief. `debate_prototype.py` and
`HANDOFF_for_new_chat.md` were NOT found anywhere on this machine — engine rebuilt from brief + whiteboard
(see DECISIONS.md).

## Phase 1 - companies (DONE)
30 companies, 10 sectors, real CIKs from SEC's own ticker table. `data/companies.csv`. XOM CIK overridden
to the real Exxon Mobil Corp registrant (see DECISIONS.md).

## Phase 2 - 10-K text (DONE)
All 30 companies: both Item 1 and Item 1A extracted and verified by hand (head/tail spot checks).
GE needed a manual heading-pair override (its filing doesn't use "Item 1A" as a standalone heading).
`data/raw_text/*.txt` + `text_metadata.csv`.

## Phase 3 - labels (DONE)
Finance + Risk pulled live from Yahoo Finance; ESG from a Kaggle Sustainalytics snapshot (Yahoo's own
live ESG field was empty for every company — confirmed directly). No missing data for any of the 30
companies. `data/labels.csv`, `data/labels_log.txt`.

## Phase 4 - embeddings (DONE)
`all-MiniLM-L6-v2` (primary) + `BAAI/bge-small-en-v1.5` (comparison), both on GPU. Item 1 / Item 1A /
both embedded separately. Sanity checks (nearest-neighbour same-sector rate, PCA map, similarity
heatmap) all produced and well above the random baseline. `data/embeddings/`, `outputs/pca_map_*.png`,
`outputs/similarity_heatmap_*.png`, `outputs/nearest_neighbours_*.csv`.

## Phase 5 - directions (DONE)
Nested cross-validation (companies held out, not pairs) across 2 models x 3 variants x 3 perspectives,
plus a ridge/mean-difference comparison, a sector-only baseline, a C-sensitivity sweep, and a 100-sample
bootstrap. Item 1A chosen as the final embedding variant (best held-out accuracy). Final G matrix and
directions saved to `data/directions.npz` / `directions_meta.json`. All results in
`outputs/direction_*.csv/.png`, `G_matrix.*`, `label_vs_sector_R2.csv`.

## Phase 6 - investor weights (PARTIAL - see final note)
Three presets (finance_heavy, esg_heavy, balanced) built and saved to `data/investors.json`. The
interactive 18-pair questionnaire (`p06_investor.py --interview`) is built, using the real learned
directions to pick the pairs where perspectives most disagree, but NOT run — it needs the user's own
live answers, which were not available overnight.

## Phase 7 - debate engine, real data (DONE)
Engine (`src/engine.py`) unit-tested against the whiteboard's exact numbers first (`tests/test_engine.py`,
8/8 passing), then run on the real 30-company world for all three investors: full-agent ranking, one
fully worked pair debate each (full human-readable trace + bar/line plot), a full 30-company tournament,
baseline comparison (majority vote, single-agent, top-pick plurality), and an investor-ranking
correlation check. Every debated pair's identity (`m_F+m_E+m_R = S_ALL gap`) verified automatically;
zero mismatches. `outputs/debate_trace_*.txt`, `tournament_*.json`, `full_agent_ranking_*.csv`,
`baseline_comparison.csv`, `investor_ranking_correlation.csv`.

## Phase 8 - stress tests (DONE)
Noisy agents (7 noise levels x 120 pairs x 200 trials, debate vs. majority vote), missing agent (each
perspective dropped in turn), early stopping (loose / Cauchy-Schwarz / tight bounds, messages saved +
winner-mismatch check — 0 mismatches everywhere). LLM-argument layer (brief's explicitly-optional
extension) skipped — no API key configured for this pass, and it was marked lowest priority.
`outputs/stress_*.csv/.png`.

## Phase 9 - RESULTS.md (DONE)
`reports/RESULTS.md` - every section pulled from the actual run's output files, numbers checked against
the CSVs/JSONs directly, not retyped from memory.

## Phase 10 - EXPLAIN.md (DONE)
`reports/EXPLAIN.md` - 5-minute talk script, glossary, step-by-step list, honest Q&A, and the full
"things I inferred and need to confirm" list, all using this run's real numbers.

## What is NOT done
- The investor questionnaire was not run with a real person (needs you, see EXPLAIN.md section 5 item 7).
- The optional LLM-argument layer (Phase 8's clearly-optional extension) was skipped - no key configured,
  and the brief ranks it last.
- Scale is 30 companies / 3 agents, as planned for a first pass, not the paper's full scale.

## Final check against the brief's "definition of done"
- [x] companies.csv, raw text + metadata; no failures to log (all 30 succeeded)
- [x] labels.csv with sources + as-of dates
- [x] embeddings saved; PCA map + nearest-neighbour check produced
- [x] three directions learned; held-out accuracy/correlations reported; G heatmap produced
- [x] investor weights for the 3 presets (user's own: built, not yet run - needs you)
- [x] debate engine runs end to end; full trace logged; identity + winner checks pass
- [x] baselines compared; stress tests run (noise, missing agent, early stopping)
- [x] RESULTS.md and EXPLAIN.md written from real numbers
- [x] README with the single rerun command; PROGRESS.md and DECISIONS.md complete
- [x] this final message
