# DECISIONS

Every default chosen without asking, and why. (Brief: "choose a sensible default, write it down, continue.")

## Setup
- New project folder `company_debate/` next to the existing `madc_completion/` project, not inside it -
  unrelated pipeline, own venv.
- `debate_prototype.py` and `HANDOFF_for_new_chat.md` were NOT found anywhere on this machine (searched
  the whole user profile). The whiteboard PDF the user sent (`Note 8 Oct 2026 (2).pdf`) matches the brief
  closely and was used as the primary source instead; the debate engine was rebuilt from the brief +
  whiteboard, not ported from a missing file.

## Phase 1 - companies
- 30 large US-listed companies across 10 sectors (brief's target), chosen by hand for contrast: Tech,
  Oil & Gas, Banks, Utilities, Retail, Pharma & Healthcare, Consumer Staples, Industrials, Automotive,
  Telecom. Includes high-ESG-risk names (XOM, CVX, BA, GE) and low-ESG-risk names (HD, AAPL, MSFT) so the
  ESG perspective has real spread to learn from.
- SEC changed Exxon's ticker->CIK mapping in 2026 (XOM now officially points to a new holding-company
  registrant with no 10-K filed yet). We keep the ticker "XOM" but use Exxon Mobil Corp's own CIK
  (34088), which has the real, current 10-K. Logged in code with a comment; not silently patched.

## Phase 2 - text extraction
- Item 1 / Item 1A are found by the LAST heading of that type before the NEXT item's heading (not the
  first mention), because filings often reference "Item 1A" inside other sections (e.g. "see Item 1A
  for details") before the real section starts. A heading must stand alone on its own line to count
  (`heading_like`) - this is what stopped those inline references from being matched as section starts.
- `MAX_SECTION_CHARS = 250,000`: if a detected section is longer than this, the true end heading was
  probably missed (banks' Item 1A sections are genuinely long - BAC's real Item 1A is ~117k chars - so
  this is a safety cap, not a normal case) and the text is cut off at the cap rather than run away
  until the next accidental match.
- GE's 10-K does not use "Item 1A" as a heading at all (it uses "RISK FACTORS." as a bare caption, with
  the numbered "Item 1A." line appearing only in the Part III cross-reference table). GE is handled with
  a manual heading pair (`ABOUT GE AEROSPACE.` -> `BUSINESS OVERVIEW AND ENVIRONMENT.` for Item 1, and
  `RISK FACTORS.` -> `LEGAL PROCEEDINGS.` for Item 1A), text still taken verbatim from the real filing.
  This is a one-off override, documented in code (`OVERRIDES` dict in `p02_fetch_10k.py`), not a
  silent guess.
- Result: all 30 companies have both sections extracted; none dropped at this phase.

## Phase 3 - labels
- Finance = mean of z(3-year revenue CAGR) and z(profit margin); margin is Operating Income / Revenue
  where available, else Pretax Income / Revenue for banks (they have no "Operating Income" line in
  Yahoo's statements). Both z-scores are winsorised at +-2.5 before averaging so one extreme company
  (NVIDIA's ~3-year revenue roughly doubling) cannot dominate the whole Finance label.
- ESG = Sustainalytics "Total ESG Risk score" (lower = more risk = worse), sign-flipped so higher is
  always better, like the other two labels. Source: Kaggle dataset `pritish509/s-and-p-500-esg-risk-ratings`
  (a public snapshot of Yahoo Finance's Sustainalytics scores), file dated 2024-06-07. Chosen because
  Yahoo Finance's own live `sustainability` field returned EMPTY for every ticker tested on this machine
  (checked directly before picking this route) - this is a real, current limitation of the free API, not
  a guess. The snapshot's build date (mid-2024) is older than the other two labels (pulled live today);
  flagged as a limitation in EXPLAIN.md.
- Risk = mean of z(-annualised volatility) and z(max drawdown), both over 3 years of daily prices,
  winsorised the same way. Higher = safer, consistent with Finance/ESG's "higher = better" convention.

## Phase 5 - directions
- Primary embedding: `all-MiniLM-L6-v2`. Second model for comparison: `BAAI/bge-small-en-v1.5` (a
  stronger, still-small, free model) rather than a "finance-oriented" model, because no free finance-
  specific sentence embedding model with a stable public checkpoint was readily available; BGE-small is
  a reasonable, well-known alternative general embedding model for the comparison the brief asks for.
- Chunking: 150 words/chunk (comfortably under MiniLM's 256-token window after sub-word tokenisation;
  verified directly - under 10% of chunks exceed the window for any company, see
  `outputs/*chunk_stats.csv`), mean-pooled, then L2-normalised.
- Variant chosen for the main pipeline: Item 1A alone scored highest in nested cross-validation
  (mean pair accuracy 0.794 vs 0.765 for "both" and 0.753 for Item 1 alone) - see
  `outputs/direction_cv_results.csv`. This is a little counter-intuitive (the Finance/ESG/Risk labels
  are not literally about risk-factor text) but it is what the held-out evaluation shows, so it is what
  is used; flagged honestly, not overridden by intuition.
- Direction fitting method: L2-regularised logistic regression on pairs (no intercept), per-perspective
  C chosen by NESTED cross-validation (so the held-out companies in the final report never influenced
  the choice of C). Compared against ridge regression and mean-difference baselines on the same folds;
  logistic regression was equal or better for all three perspectives (see `direction_cv_results.csv`).
- `GAP_THRESHOLD = 0.25` (z-score units): pairs whose two companies' labels differ by less than this are
  dropped from both training and evaluation, since "who is better" is not meaningfully defined for a
  near-tie.
- Companies-held-out (not pairs-held-out) cross-validation throughout, as the brief requires, to avoid
  leakage.

## Phase 6 (planned) - investor weights
- Presets (Finance-heavy, ESG-heavy, Balanced) are built immediately so Phases 7-9 have real investors to
  run with. The interactive pairwise questionnaire is also built and left ready to run - it needs the
  user's own answers, which cannot be collected while she is asleep (see final message / TODO).

## General
- Fixed seed 42 everywhere (`config.set_seeds`).
- Every external number (filing dates, financial figures, price windows, ESG scores) is saved together
  with its source and as-of date in the same CSV row it appears in - never typed from memory.
