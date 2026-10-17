# RESULTS

Every number below comes from an actual run of this pipeline (see `outputs/` for the source file of
each table/plot). Nothing here is a placeholder.

## 1. Companies

30 US-listed companies across 10 sectors (`data/companies.csv`): Technology (AAPL, MSFT, NVDA),
Oil & Gas (XOM, CVX, COP), Banks (JPM, BAC, GS), Utilities (NEE, DUK, SO), Retail (WMT, COST, HD),
Pharma & Healthcare (JNJ, PFE, UNH), Consumer Staples (PG, KO, PEP), Industrials (CAT, GE, BA, UPS),
Automotive (TSLA, F, GM), Telecom (T, VZ). All 30 have both 10-K sections extracted successfully — none
dropped (`data/raw_text/text_metadata.csv`).

## 2. Data sources and dates

| Label | Source | As-of |
|---|---|---|
| Finance (3y revenue CAGR + margin) | Yahoo Finance annual statements via `yfinance` | pulled live, run date |
| ESG (Sustainalytics Total ESG Risk score) | Kaggle `pritish509/s-and-p-500-esg-risk-ratings` | file dated 2024-06-07 |
| Risk (volatility + max drawdown) | Yahoo Finance daily prices via `yfinance`, 3-year window | pulled live, run date |
| 10-K text (Item 1, Item 1A) | SEC EDGAR, each company's latest 10-K | filing dates in `text_metadata.csv` (mostly Jan–Mar 2026; COST Oct 2026) |

No company is missing any label (`data/labels_log.txt`: "no missing data").

## 3. Embedding sanity checks

Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), 150-word chunks, mean-pooled, L2-normalised.
A second model (`BAAI/bge-small-en-v1.5`) was run in parallel for comparison.

**Nearest-neighbour check** (does cosine similarity mostly find same-sector companies?), MiniLM:

| Section embedded | top-1 same-sector | top-3 same-sector | random baseline |
|---|---|---|---|
| Item 1 (Business) | 93.3% | 62.2% | 7.1% |
| Item 1A (Risk Factors) | 76.7% | 50.0% | 7.1% |
| Both | 93.3% | 55.6% | 7.1% |

All three variants are far above the random baseline — the embeddings are capturing real business
similarity, not noise. Full nearest-neighbour table: `outputs/nearest_neighbours_all_minilm_l6_v2.csv`.
Example: GS's nearest neighbour is BAC (cosine 0.939); XOM's is COP (0.856); AAPL's top neighbour is JNJ
(0.86) rather than another tech company — a real, reportable limitation (see §7).

PCA map: `outputs/pca_map_all_minilm_l6_v2.png`. Similarity heatmap: `outputs/similarity_heatmap_all_minilm_l6_v2.png`.
Chunk-size check: under 10% of chunks exceed MiniLM's 256-token window for every company
(`outputs/*chunk_stats.csv` — e.g. JNJ 10%, PFE/GM/PEP 5%, most companies 0%).

## 4. Direction fit (Phase 5)

Nested cross-validation, companies held out (never pairs) — `outputs/direction_cv_results.csv`:

| Perspective | Variant | Held-out pair accuracy | Held-out Spearman |
|---|---|---|---|
| Finance | Item 1A | **0.862** ± 0.030 | 0.613 |
| ESG | Item 1 | **0.840** ± 0.052 | 0.732 |
| Risk | Item 1A | **0.748** ± 0.044 | 0.452 |

Chosen embedding for the whole pipeline: **Item 1A alone** (mean pair accuracy 0.794 across the three
perspectives, vs. 0.765 for "both sections" and 0.753 for Item 1 alone) — a little counter-intuitive,
since the labels aren't literally about risk-factor text, but it's what the held-out evaluation shows.

Logistic regression beat ridge regression and mean-difference on every perspective on the same folds
(e.g. Finance: logistic 0.862 vs ridge 0.778 vs mean-diff 0.739).

**Regularisation sensitivity** (`outputs/direction_C_sensitivity.png`/`.csv`): accuracy is fairly flat
for C≥1 on Finance and Risk, but the *direction itself* keeps changing — see the G-vs-C table below.
Final C per perspective (highest held-out accuracy): Finance=10, ESG=0.1, Risk=100.

**Sector baseline, for context**: predicting a held-out company's label from its sector's mean label
alone already gets 0.822 (Finance) / 0.730 (ESG) / 0.641 (Risk) pair accuracy, and sector explains
55–73% of each label's variance in-sample (`outputs/direction_sector_baseline.csv`,
`label_vs_sector_R2.csv`). The learned directions beat this baseline on Finance and Risk but are close
to it on ESG — with only 30 companies, a meaningful share of what the directions learn is "which sector",
not a finer distinction. Stated plainly in §7.

**Agreement matrix G** (real, learned — not the whiteboard's illustrative numbers):

|   | Finance | ESG | Risk |
|---|---|---|---|
| **Finance** | 1.00 | +0.19 | +0.14 |
| **ESG** | +0.19 | 1.00 | **−0.16** |
| **Risk** | +0.14 | −0.16 | 1.00 |

Only ESG and Risk are natural challengers here (negative G); Finance sits mildly positively correlated
with both. Bootstrap stability (`outputs/direction_bootstrap.json`): resampled directions have cosine
0.77–0.87 with the full-data direction — reasonably stable but not rock solid at n=30. G itself shifts
with C (e.g. E–R goes from −0.22 at C=0.01 to −0.28 at C=1000) — reported honestly, not hidden.

## 5. Investor weights (Phase 6)

Three presets used throughout (`data/investors.json`): `finance_heavy` (0.70/0.15/0.15),
`esg_heavy` (0.15/0.70/0.15), `balanced` (0.40/0.35/0.25 — the whiteboard's own numbers). The
interactive questionnaire (`p06_investor.py --interview`) is built and ready but needs a person to
answer it — not run in this pass (see the final note in EXPLAIN.md).

## 6. One fully worked debate (balanced investor)

Full agreement matrix and identity check both verified live — `outputs/debate_trace_balanced.txt`,
plot `outputs/worked_debate_balanced.png`:

```
Debate: A=NVDA vs B=VZ
  speaker 1: Finance  beta=0.40 [highest weight speaks first]        margin -0.0599  running -0.0599
  speaker 2: Risk     beta=0.25 [strongest challenger of Finance, G=+0.14]  margin +0.0430  running -0.0170
  speaker 3: ESG      beta=0.35 [remaining agent]                     margin -0.0003  running -0.0173
  final evidence -0.0173  =>  NVDA wins, VZ eliminated
  identity check: m_F+m_E+m_R = -0.0173  vs  S_ALL(VZ) - S_ALL(NVDA) = -0.0173   OK
```

**Full-agent ranking, balanced investor** (`outputs/full_agent_ranking_balanced.csv`), top 5:
1 MSFT · 2 NVDA · 3 VZ · 4 AAPL · 5 KO.

**Tournament** (all 30 companies, queue method, `outputs/tournament_balanced.json`):
winner = **MSFT**, 87 agent messages (29 pairs × 3 agents), bracket confirmed to end on the same
company the full-agent ranking puts first.

## 7. Baseline comparison

`outputs/baseline_comparison.csv` — does each method's tournament winner agree with the full agent's?

| Method | finance_heavy | esg_heavy | balanced |
|---|---|---|---|
| Majority vote | ✓ MSFT | ✓ MSFT | ✓ MSFT |
| Single agent — Finance only | ✓ MSFT | ✓ MSFT | ✓ MSFT |
| Single agent — ESG only | ✓ MSFT | ✓ MSFT | ✓ MSFT |
| Single agent — Risk only | ✗ DUK | ✗ DUK | ✗ DUK |
| Top-pick plurality | ✓ MSFT | ✓ MSFT | ✓ MSFT |

MSFT wins under every investor and every method except "Risk alone" — it is a genuinely dominant
company in this set (strong on all three perspectives). The Risk-only agent consistently prefers DUK
instead, which is the clearest single baseline difference in the whole run.

**All three investors happen to agree on the overall winner (MSFT)** — but their *full* rankings do
not (`outputs/investor_ranking_correlation.csv`, Spearman correlation of the complete 30-company
ranking):

| | finance_heavy | esg_heavy | balanced |
|---|---|---|---|
| finance_heavy | 1.00 | **0.57** | 0.83 |
| esg_heavy | 0.57 | 1.00 | 0.90 |
| balanced | 0.83 | 0.90 | 1.00 |

Finance-heavy and ESG-heavy investors only correlate at 0.57 on the full ranking — meaningfully
different priorities — even though, by coincidence in this specific 30-company set, they land on the
same #1. This is the real, honest answer to "do different investors get different winners?" (see
EXPLAIN.md's "things inferred" section on I001/I002): yes in general, but not guaranteed, and not in
this particular run's top spot.

## 8. Stress tests

### Noisy agents (`outputs/stress_noise_*.csv/.png`, 120 sampled pairs × 200 trials per noise level)

The debate (all 3 agents, full information) is consistently **more robust to noise than majority
vote** at every noise level above zero, for every investor. Example (balanced investor):

| noise σ | debate mismatch rate | majority-vote mismatch rate |
|---|---|---|
| 0.00 | 0.000 | 0.167 |
| 0.02 | 0.103 | 0.209 |
| 0.05 | 0.216 | 0.277 |
| 0.10 | 0.322 | 0.353 |
| 0.40 | 0.449 | 0.460 |

(At σ=0 majority vote is still wrong 16.7% of the time even with NO noise, because majority vote is a
different decision rule from the full agent, not just a noisy version of it — a vote can be 2-1 against
the direction the weighted sum actually favours.)

### Missing agent (`outputs/stress_missing_agent_*.csv`)

Dropping an agent entirely hurts more the more that investor weights it. Balanced investor
(beta = 0.40/0.35/0.25): dropping ESG flips 30.0% of pair winners, dropping Finance flips 14.2%,
dropping Risk flips 5.8% — roughly tracks the beta ordering (E and F carry more weight than R here, and
cause more damage when missing), though not perfectly proportionally.

### Early stopping (`outputs/stress_early_stop_*.csv`)

The "tight" bound (uses the real G matrix and what's already been said) saves more messages than the
looser Cauchy–Schwarz bound, which saves more than the loosest bound — as expected:

| investor | loose | Cauchy–Schwarz | tight |
|---|---|---|---|
| finance_heavy | 0% | 10.3% | 19.2% |
| esg_heavy | 3.9% | 23.9% | **33.3%** |
| balanced | 0% | 0% | 0% |

**Winner mismatches under early stopping: 0, for every bound, for every investor** — exactly as the
theory requires (confirmed again on 600 random pairs in `tests/test_engine.py`). The **balanced**
investor saves nothing: its weights (0.40/0.35/0.25) are close enough together that, on this question
set, the running evidence rarely clears the bound before all three agents have spoken — a genuine,
reportable finding, not a bug: early stopping's payoff depends on how *concentrated* an investor's
weights are, not just on the question set.

## 9. Limitations

See `reports/EXPLAIN.md` → "What to admit before they ask" for the full list, summarised: label scale
(30 companies only), ESG snapshot is from 2024 while Finance/Risk are pulled live today, directions are
shared across investors (an assumption, not verified), sector explains a large share of each label, and
the clean-case debate-equals-full-agent result is an algebraic identity, not evidence that debate beats
a single model — the stress tests are what actually test that claim.
