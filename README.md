# Company Debate: When Do Partial-Information Agents Converge to a Full-Information Debate?

**Ishwarya Mohan**¹, under the supervision of **Prof. Mahesh Mohan M R**¹ and **Kunal**¹
¹ Indian Institute of Technology Kharagpur

*Working research, part of the "Financial Agentic AI Debate" project. This repository accompanies an
internal research brief (8 Oct 2026) and is not yet a formal publication — see [Status](#status) and
[Citation](#citation).*

📄 [EXPLAIN.md](reports/EXPLAIN.md) — plain-language walkthrough &nbsp;·&nbsp;
📊 [RESULTS.md](reports/RESULTS.md) — full results with sources &nbsp;·&nbsp;
📓 [PROGRESS.md](PROGRESS.md) / [DECISIONS.md](DECISIONS.md) — reproducibility log

---

## Abstract

We study a multi-agent debate setting for company selection in which three perspective agents —
**Finance**, **ESG**, and **Risk** — each observe a company only through a single learned direction in
embedding space, while a reference *full agent* observes all three directions jointly, weighted by an
investor's stated preferences. We show that when agent scores are linear in a shared embedding, the
debate's outcome is provably identical to the full agent's ranking once every agent has spoken — an
algebraic identity, not an empirical finding. We instead use this identity as a correctness check and
study three regimes where the guarantee is no longer trivial: (i) **noisy** agent evidence, where the
full 3-agent debate is consistently more accurate than majority voting; (ii) a **missing** agent, whose
damage scales with that agent's weight in the investor's preferences; and (iii) **early stopping**,
where a provably valid bound on undisclosed evidence saves up to one third of agent messages for
concentrated investor preferences and none for diffuse ones. We instantiate and test this framework on
**30 real US-listed companies**, embedding their actual SEC 10-K filings with a pretrained sentence
encoder, learning each perspective direction from real financial, ESG, and market-risk labels with
held-out (company-level) cross-validation, and verifying the core identity on every debated pair with
zero violations.

<p align="center"><img src="outputs/pca_map_all_minilm_l6_v2.png" width="500" alt="PCA projection of company embeddings, coloured by sector"></p>
<p align="center"><sub>Figure 1. Companies embedded from SEC 10-K text and projected to 2-D (PCA);
colour = sector. Same-sector companies cluster together far above a random baseline (§4.1).</sub></p>

---

## 1. Introduction

Multi-agent LLM debate has been proposed as a way to improve reasoning by having models critique one
another. A less-examined question is *informational*: what happens when the agents debating do not all
have access to the same information? We study this in a company-selection setting motivated by
investment decision-making, where a Finance analyst, an ESG analyst, and a Risk analyst each have
genuine, non-overlapping expertise, and a portfolio manager (the investor) must weigh their input.

We formalise "expertise" as a single direction in a shared embedding space (Def. 1–2, §2), and show that
the debate-vs-full-agent agreement result is a direct consequence of linearity (§2.3), not a property
that needs to be demonstrated empirically — the interesting questions are instead what breaks this
equivalence, and by how much (§2.4, §4.3).

## 2. Method

### 2.1 Company representation

Each company $i$ is represented by an embedding $z_i \in \mathbb{R}^d$ ($d=384$), obtained by encoding
its SEC 10-K Risk Factors section with a pretrained sentence encoder
(`sentence-transformers/all-MiniLM-L6-v2`), chunking to respect the model's context window, mean-pooling
chunk vectors, and L2-normalising.

### 2.2 Perspective directions and the full agent

For perspective $k \in \{F, E, R\}$ (Finance, ESG, Risk), we learn a unit direction $d_k$ by fitting an
L2-regularised logistic classifier on company **pairs**, with target "does company $a$ score higher than
company $b$ on perspective $k$" and feature $z_a - z_b$ (Def. 2). An investor is a weight vector
$\beta = (\beta_F, \beta_E, \beta_R)$, $\beta_k \ge 0$, $\sum_k \beta_k = 1$. The **full agent**'s
direction and score are

$$d_{\text{ALL}} = \sum_k \beta_k\, d_k \qquad\qquad S_{\text{ALL}}(i) = z_i \cdot d_{\text{ALL}}$$

while **partial agent** $k$ knows only $\beta_k d_k$ — in particular, none of the three partial agents
individually knows $d_{\text{ALL}}$ or $S_{\text{ALL}}$.

### 2.3 The debate, and why it must agree with the full agent

For a pair $(A, B)$, agent $k$'s margin is $m_k = \beta_k\,(\Delta z \cdot d_k)$ where
$\Delta z = z_B - z_A$. Agents speak in turn — highest-weight agent first, then at each subsequent step
the *strongest challenger* of the previous speaker, defined as $\arg\min_j G_{kj}$ among agents who
have not yet spoken, where $G_{kl} = \cos(d_k, d_l)$ is the inter-agent **agreement matrix** — and the
sign of the cumulative "running evidence" after all agents have spoken decides the winner. Because
$d_{\text{ALL}}$ is a $\beta$-weighted sum of the $d_k$, this quantity is, by construction, identical to
the full agent's score gap:

$$m_F + m_E + m_R \;=\; \Delta z \cdot d_{\text{ALL}} \;=\; S_{\text{ALL}}(B) - S_{\text{ALL}}(A)$$

so **the debate winner equals the full-agent winner whenever all agents have spoken** — a property of
linear scoring, independent of speaker order. We verify this identity automatically on every debated
pair (§4.3; zero violations across all runs) and use it as a correctness test for the implementation,
not as the paper's finding.

### 2.4 Where the identity breaks: noise, missing agents, early stopping

Three departures from the clean setting make the comparison non-trivial, and are where we report actual
empirical results (§4.4):

- **Noise.** Each margin is perturbed by $\mathcal{N}(0, \sigma^2)$ before the debate decides; we
  compare the resulting mismatch rate against the full-agent winner to the mismatch rate of a simple
  majority vote under the same noise.
- **A missing agent.** One perspective is dropped entirely (its margin is never heard); $\beta$ is
  renormalised over the remaining two.
- **Early stopping.** A debate halts once a provable bound on the undisclosed agents' combined margin
  — derived from $\|\Delta z\|$, the spoken margins, and $G$ — guarantees the sign cannot flip (three
  bounds of increasing tightness are compared; see `engine.py:_bounds`).

### 2.5 Tournament

Given a candidate set, companies are paired from a queue, the loser of each debate is eliminated, and
the winner rejoins the back of the queue, until one company remains (a standard single-elimination
queue tournament).

## 3. Experimental setup

**Companies.** 30 US-listed companies across 10 sectors (Technology, Oil & Gas, Banks, Utilities,
Retail, Pharma & Healthcare, Consumer Staples, Industrials, Automotive, Telecom), selected to induce
genuine disagreement between perspectives (e.g. energy majors score well on Finance, poorly on ESG).

**Labels.** Finance = z-scored blend of 3-year revenue CAGR and operating/pretax margin (Yahoo Finance).
ESG = Sustainalytics Total ESG Risk score, sign-flipped (Kaggle snapshot, 2024-06-07 — the only free
source found with full coverage; Yahoo Finance's live ESG field was empty for every company tested).
Risk = z-scored blend of annualised volatility and maximum drawdown over 3 years of daily prices
(Yahoo Finance), sign-flipped so higher is safer. All three winsorised at $\pm2.5\sigma$.

**Text.** SEC 10-K Item 1 (Business) and Item 1A (Risk Factors), extracted directly from EDGAR filings
(heading-based extraction with fallbacks for non-standard filers; see `DECISIONS.md`).

**Evaluation protocol.** Direction quality is evaluated with **company-level** (not pair-level) nested
cross-validation — holding out whole companies prevents a held-out company's embedding from leaking into
training via other pairs. Regularisation strength is selected by an inner cross-validation on the
training fold only.

**Investors.** Three preset weight profiles (`finance_heavy`, `esg_heavy`, `balanced`); an interactive
18-question pairwise elicitation tool is implemented (`p06_investor.py --interview`) but not yet
administered to a human subject (§5).

## 4. Results

### 4.1 Embedding quality

Nearest-neighbour same-sector rate (cosine similarity), vs. a random baseline of 7.1%:

| Variant | top-1 same-sector | top-3 same-sector |
|---|---|---|
| Item 1A (Risk Factors) | 76.7% | 50.0% |
| Item 1 (Business) | 93.3% | 62.2% |
| Both | 93.3% | 55.6% |

### 4.2 Direction fit (held-out, company-level CV)

| Perspective | Pair accuracy | Spearman $\rho$ | Sector-only baseline (accuracy) |
|---|---|---|---|
| Finance | **0.862** $\pm$ 0.030 | 0.613 | 0.822 |
| ESG | **0.840** $\pm$ 0.052 | 0.732 | 0.730 |
| Risk | **0.748** $\pm$ 0.044 | 0.452 | 0.641 |

All three exceed chance (0.5) and their respective sector-only baseline, though for ESG the margin is
small — at $n=30$ companies, a non-trivial share of what each direction learns is attributable to sector
membership rather than finer distinctions (sector explains 55–73% of in-sample label variance; full
table in RESULTS.md §4).

Learned agreement matrix $G = \cos(d_k, d_l)$:

$$G = \begin{pmatrix} 1.00 & +0.19 & +0.14 \\ +0.19 & 1.00 & -0.16 \\ +0.14 & -0.16 & 1.00 \end{pmatrix}
\begin{matrix}F\\E\\R\end{matrix}$$

ESG and Risk are the only natural challengers (negative $G$); Finance correlates mildly positively with
both — different from the illustrative values on the originating whiteboard, which were not derived
from data (see EXPLAIN.md §5).

### 4.3 Debate correctness

The identity $m_F+m_E+m_R = S_{\text{ALL}}(B)-S_{\text{ALL}}(A)$ was checked automatically on every
debated pair across all tournaments and all three investors: **zero violations**. The tournament winner
matched the full-agent top-ranked company for every investor tested. All investors converged on the
same overall winner (MSFT) in this company set — but their complete 30-company rankings correlate at
only $\rho = 0.57$ (finance-heavy vs. esg-heavy), indicating materially different preferences that
happened to coincide at rank 1 (full correlation matrix in RESULTS.md §7).

### 4.4 Stress tests

**Noise.** The full 3-agent debate is more robust to additive noise on the margins than majority voting
at every noise level tested, for every investor (e.g. balanced investor, $\sigma=0.05$: debate mismatch
rate 0.216 vs. majority-vote 0.277; full curves in `outputs/stress_noise_*.png`).

**Missing agent.** Winner-flip rate scales roughly with the dropped agent's $\beta$ weight (balanced
investor: dropping ESG, $\beta_E=0.35$, flips 30.0% of pairs; dropping Risk, $\beta_R=0.25$, flips
5.8%).

**Early stopping.** The tightest bound saves 19–33% of agent messages for concentrated investors
(finance-heavy, esg-heavy) and 0% for the balanced investor, whose weights are too close together for
the running evidence to lock in before all agents speak — and, consistent with the bound being
provably valid, **zero winner mismatches were observed under early stopping across all conditions
tested** (also verified on 600 synthetic random-world pairs in `tests/test_engine.py`).

## 5. Limitations

- Scale: 30 companies, 3 agents, 2 debate rounds per pair — a first instantiation of the framework, not
  matched to any particular target scale.
- The ESG label is a static 2024 snapshot while Finance and Risk are pulled at run time; a genuine
  temporal mismatch, not reconciled.
- Perspective directions are assumed shared across all investors (only $\beta$ varies) — this follows
  the originating research brief but has not been independently validated.
- The investor-elicitation tool has not yet been administered to a human subject; all reported investors
  are hand-specified presets.
- No comparison yet against a learned (rather than hand-derived) early-stopping bound, or against more
  than three agents.

Full discussion, including every default chosen and why: [DECISIONS.md](DECISIONS.md).

## Reproducing the results

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/pip install scipy scikit-learn sentence-transformers requests beautifulsoup4 lxml yfinance seaborn joblib pytest
export SEC_EMAIL=you@example.com      # required by SEC EDGAR's fair-access policy

cd src && python run_all.py           # full pipeline, §3-4
python -m pytest ../tests -q          # correctness tests: identity, full-agent agreement,
                                       # early-stopping safety, whiteboard reproduction
```
10-K downloads and price/ESG pulls are cached (`data/sec_cache/`, gitignored, ~172MB) so repeat runs are
fast. To add a human-elicited investor profile: `python p06_investor.py --interview`, then re-run
`p07_debate.py` and `p08_stress.py`.

## Repository structure

```
src/engine.py          debate engine: directions, margins, speaker order, early-stopping bounds, tournament
src/directions.py      direction fitting (logistic / ridge / mean-difference) + nested cross-validation
src/p01-p08_*.py       pipeline stages, in order (§3)
tests/test_engine.py   whiteboard reproduction + identity/agreement/early-stopping property tests
data/                  companies, labels, embeddings, learned directions, investor profiles
outputs/                every table and figure cited above
reports/EXPLAIN.md      plain-language walkthrough, glossary, Q&A
reports/RESULTS.md      every number in this README, traced to its source file
```

## Status

This is an active internal research project, not a peer-reviewed publication. Results, labels, and the
company set are expected to change as the project scales up. The repository is maintained as the
record of a specific experimental run, per the accompanying `PROGRESS.md`.

## Citation

No formal write-up exists yet. If referencing this repository in the interim:

```bibtex
@misc{mohan2026companydebate,
  title  = {Company Debate: When Do Partial-Information Agents Converge to a Full-Information Debate?},
  author = {Mohan, Ishwarya and Mohan M R, Mahesh and Kunal},
  year   = {2026},
  note   = {Unpublished research repository, Indian Institute of Technology Kharagpur},
  howpublished = {\url{https://github.com/ishwaryaaaaaaaaa/Phase1_meeting}}
}
```

## Acknowledgments

Research direction and the original whiteboard formulation (8 Oct 2026) from Prof. Mahesh Mohan M R and
Kunal, IIT Kharagpur.
