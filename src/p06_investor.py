"""Phase 6: investor weights beta = (beta_F, beta_E, beta_R), nonnegative, summing to 1.

Two ways to get an investor:
  1. Presets (Finance-heavy / ESG-heavy / Balanced) - always available, used by the I001/I002-style
     demo and by Phases 7-9 so the pipeline has real investors to run on without waiting for a person.
  2. The user's own 15-20 pairwise answers ("which company would you rather invest in?"), fit with a
     constrained regression (beta >= 0, sum to 1). Run interactively:  python p06_investor.py --interview
"""
import argparse
import json

import numpy as np
from scipy.optimize import nnls

from common import DEFAULT_TAG, DEFAULT_VARIANT, PERSPECTIVES, load_companies, load_embeddings
from config import DATA, set_seeds

PRESETS = {
    "finance_heavy": {"F": 0.70, "E": 0.15, "R": 0.15},
    "esg_heavy": {"F": 0.15, "E": 0.70, "R": 0.15},
    "balanced": {"F": 0.40, "E": 0.35, "R": 0.25},   # the whiteboard's own numbers
}


def load_directions():
    d = np.load(DATA / "directions.npz")
    meta = json.loads((DATA / "directions_meta.json").read_text())
    return d["D"], d["G"], list(d["tickers"]), meta


def pick_disagreement_pairs(Z, D, names, n_pairs=18, seed=0):
    """Choose pairs where the three perspectives genuinely disagree about which company is better -
    these are the pairs whose answer actually reveals an investor's weights. For each pair we look at
    the per-agent vote (sign of dz . d_k) and keep pairs where the votes are NOT unanimous."""
    rng = np.random.default_rng(seed)
    idx = list(range(len(names)))
    candidates = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            dz = Z[j] - Z[i]
            votes = np.sign(D @ dz)
            if len(set(votes.tolist())) > 1:          # at least one agent disagrees with the others
                candidates.append((i, j, int(np.abs(votes.sum()))))   # smaller |sum| = more disagreement
    candidates.sort(key=lambda t: t[2])
    chosen, seen = [], set()
    for i, j, _ in candidates:
        if i in seen and j in seen and len(chosen) > n_pairs // 2:
            continue
        chosen.append((i, j)); seen.add(i); seen.add(j)
        if len(chosen) >= n_pairs:
            break
    return chosen


def company_blurb(ticker, comps):
    row = comps.loc[ticker]
    return f"{ticker} - {row['name']} ({row['sector']})"


def run_interview(n_pairs=18):
    set_seeds()
    comps = load_companies()
    Z, names = load_embeddings(DEFAULT_TAG, DEFAULT_VARIANT)
    D, G, d_names, meta = load_directions()
    assert names == d_names, "embedding / direction company order mismatch"
    pairs = pick_disagreement_pairs(Z, D, names, n_pairs)

    print("\nFor each pair, type 1 if you'd rather invest in the FIRST company, 2 for the SECOND,")
    print("or 0 if you have no preference / want to skip. Short plain-English company summaries only.\n")
    X, y = [], []
    for k, (i, j) in enumerate(pairs, 1):
        print(f"[{k}/{len(pairs)}]")
        print(f"  1) {company_blurb(names[i], comps)}")
        print(f"  2) {company_blurb(names[j], comps)}")
        ans = input("  Your choice (1/2/0): ").strip()
        if ans not in ("1", "2"):
            continue
        dz = Z[j] - Z[i]
        gaps = D @ dz                      # per-perspective score gap (j minus i) along each direction
        X.append(gaps)
        y.append(1.0 if ans == "2" else -1.0)
    if len(X) < 5:
        print("Too few answers to fit weights reliably (need >=5). Nothing saved.")
        return None
    X, y = np.array(X), np.array(y)
    beta = fit_beta(X, y)
    save_investor("user", beta, {"n_answers": len(X), "n_offered": len(pairs)})
    print("\nFitted weights:", {k: round(float(b), 3) for k, b in zip(PERSPECTIVES, beta)})
    return beta


def fit_beta(X, y, n_boot=300, seed=0):
    """Nonnegative least squares: y_pair ~= X_pair . beta, beta>=0, then renormalise to sum to 1.
    (NNLS directly gives beta>=0; the sum-to-1 constraint is applied by rescaling afterwards, which is
    equivalent for this ranking-style objective since only the relative weights matter.)"""
    beta, _ = nnls(X, y)
    if beta.sum() == 0:
        beta = np.ones(X.shape[1])
    beta = beta / beta.sum()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        b, _ = nnls(X[idx], y[idx])
        boots.append(b / b.sum() if b.sum() > 0 else np.ones(len(beta)) / len(beta))
    boots = np.array(boots)
    return beta, boots.std(axis=0)


def save_investor(investor_id, beta_and_sd, meta):
    beta, sd = beta_and_sd if isinstance(beta_and_sd, tuple) else (beta_and_sd, None)
    path = DATA / "investors.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[investor_id] = {"beta": {k: float(b) for k, b in zip(PERSPECTIVES, beta)},
                         "beta_bootstrap_sd": None if sd is None else {k: float(s) for k, s in zip(PERSPECTIVES, sd)},
                         **meta}
    path.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interview", action="store_true", help="run the interactive questionnaire")
    ap.add_argument("--n-pairs", type=int, default=18)
    args = ap.parse_args()

    set_seeds()
    for pid, w in PRESETS.items():
        beta = np.array([w[k] for k in PERSPECTIVES])
        save_investor(pid, beta, {"kind": "preset"})
        print(f"{pid:15s} beta = {dict(zip(PERSPECTIVES, beta.round(3)))}")

    if args.interview:
        run_interview(args.n_pairs)
    else:
        print("\n(presets saved; run with --interview to add your own answers to data/investors.json)")


if __name__ == "__main__":
    main()
