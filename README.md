# Company Debate — multi-perspective agent debate for picking companies

Real companies -> real 10-K text -> embeddings -> learned Finance/ESG/Risk directions -> an investor's
weights -> a debate engine where perspective agents argue about which company is better, all checked
against the "full agent" that sees everything.

Start with **`reports/EXPLAIN.md`** if you need to present or explain this. **`reports/RESULTS.md`** has
every number this run produced. This README is just how to run it.

## Setup (already done on this machine)
```bash
python -m venv .venv --system-site-packages
.venv/Scripts/pip install scipy scikit-learn sentence-transformers requests beautifulsoup4 lxml yfinance seaborn joblib pytest
```
Set your SEC contact email once (SEC requires it in every request header):
```bash
export SEC_EMAIL=you@example.com     # or set CLAUDE_CODE_USER_EMAIL
```

## Reproduce everything
```bash
cd src
python run_all.py
```
Runs Phases 1–8 in order (companies -> 10-K text -> labels -> embeddings -> sanity checks -> directions
-> investor presets -> debate engine -> stress tests). Takes a few minutes; 10-K downloads and ESG/price
data are cached on disk so a second run is much faster.

To add **your own** investor profile (15–20 pairwise answers), run separately and then re-run `p07`/`p08`:
```bash
python p06_investor.py --interview
python p07_debate.py
python p08_stress.py
```

## Tests
```bash
python -m pytest tests -q
```
Reproduces the whiteboard's exact numbers (margins +0.12/-0.04/+0.07, running evidence
0.12/0.08/0.15, bracket {A,B,C,D}->{B,C,D}->{B,C}->B) and checks the engine on random worlds: the
identity `m_F+m_E+m_R = S_ALL(B)-S_ALL(A)`, debate-winner == full-agent-winner for every pair and every
tournament, and that early stopping never changes a winner.

## Layout
```
src/            all code (one file per phase, numbered p01.. p08, plus engine.py/directions.py/common.py)
data/           companies.csv, labels.csv, embeddings/, investors.json, directions.npz, sec_cache/
outputs/        every plot and CSV this run produced
reports/        RESULTS.md (numbers) and EXPLAIN.md (talk script, glossary, Q&A)
tests/          pytest suite for the debate engine
PROGRESS.md     phase-by-phase log of what happened
DECISIONS.md    every default chosen and why
```
