"""Phase 8: stress tests.

The clean debate always agrees with the full agent (that's an algebraic identity, not a finding - see
engine.py's docstring and EXPLAIN.md). Interesting behaviour only shows up once the setup is no longer
perfectly linear / noise-free, which is exactly what these three tests do:

  1. Noisy agents   - add Gaussian noise to each agent's margin; how often does the debate's winner
                       differ from the TRUE (noise-free) full-agent winner, at several noise levels?
                       Compared against majority vote under the same noise.
  2. Missing agent  - drop one perspective entirely (its margin is just never heard); how much does
                       that change the outcome compared with having it?
  3. Early stopping - stop a pair's debate the instant the remaining agents provably cannot flip the
                       sign (engine._bounds, 'tight'); how many agent messages does that save, and does
                       the winner ever change? (It shouldn't - that's the point of a *valid* bound;
                       verified here on real data, already proven on random data in tests/test_engine.py.)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine as E
from common import PERSPECTIVES
from config import OUT, SEED, set_seeds
from p07_debate import build_world, investors

N_TRIALS = 200
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]   # std dev, same units as the margins


def noisy_margins_winner(world, beta, ia, ib, rng, noise_sd, order):
    m = E.agent_margins(world.Z[ia], world.Z[ib], world.D, beta)
    noisy = m + rng.normal(0, noise_sd, size=len(m))
    running = np.cumsum([noisy[k] for k in order])
    return running[-1] > 0, (np.sign(noisy) > 0).sum() > len(noisy) / 2    # (debate winner_is_b, majority winner_is_b)


def test_noise(world, beta, pairs, seed=SEED):
    order = E.speaker_order(beta, world.G)
    rows = []
    for sd in NOISE_LEVELS:
        rng = np.random.default_rng(seed)
        mism_debate = mism_majority = 0
        for ia, ib in pairs:
            s_all_b_wins = (E.full_scores(world, beta)[ib] > E.full_scores(world, beta)[ia])
            for _ in range(N_TRIALS):
                d_b, maj_b = noisy_margins_winner(world, beta, ia, ib, rng, sd, order)
                mism_debate += (d_b != s_all_b_wins)
                mism_majority += (maj_b != s_all_b_wins)
        n = len(pairs) * N_TRIALS
        rows.append({"noise_sd": sd, "debate_mismatch_rate": mism_debate / n, "majority_mismatch_rate": mism_majority / n, "n_trials": n})
    return pd.DataFrame(rows)


def test_missing_agent(world, beta, pairs):
    """Drop each perspective in turn: renormalise the remaining betas, redo the pair decision, compare
    to the true full-agent winner. 'Damage' = fraction of pairs whose winner flips."""
    rows = []
    s_all = E.full_scores(world, beta)
    for drop in PERSPECTIVES:
        keep = [k for k in range(3) if PERSPECTIVES[k] != drop]
        beta2 = beta.copy(); beta2[PERSPECTIVES.index(drop)] = 0.0
        if beta2.sum() == 0:
            continue
        beta2 = beta2 / beta2.sum()
        flips = 0
        for ia, ib in pairs:
            res = E.debate_pair(world, beta2, ia, ib, agents=keep)
            b_wins_true = s_all[ib] > s_all[ia]
            b_wins_now = res.winner == world.names[ib]
            flips += (b_wins_now != b_wins_true)
        rows.append({"dropped_agent": drop, "n_pairs": len(pairs), "winner_flip_rate": flips / len(pairs)})
    return pd.DataFrame(rows)


def test_early_stopping(world, beta, pairs):
    rows = []
    for kind in ("loose", "cs", "tight"):
        saved, mismatches = 0, 0
        for ia, ib in pairs:
            full = E.debate_pair(world, beta, ia, ib)
            short = E.debate_pair_early_stop(world, beta, ia, ib, kind)
            saved += 3 - short["messages"]
            mismatches += (short["winner"] != full.winner)
        rows.append({"bound": kind, "messages_saved": saved, "messages_possible": 3 * len(pairs),
                    "pct_saved": round(100 * saved / (3 * len(pairs)), 1), "winner_mismatches": mismatches})
    return pd.DataFrame(rows)


def main():
    set_seeds()
    world, _meta = build_world()
    inv = investors()
    all_pairs = [(i, j) for i in range(len(world.names)) for j in range(i + 1, len(world.names))]
    rng = np.random.default_rng(SEED)
    sample_pairs = [tuple(p) for p in rng.choice(all_pairs, size=120, replace=False)] if len(all_pairs) > 120 else all_pairs

    for inv_id, beta in inv.items():
        print(f"\n=== Stress tests for investor: {inv_id} ===")

        noise = test_noise(world, beta, sample_pairs)
        noise.to_csv(OUT / f"stress_noise_{inv_id}.csv", index=False)
        print("Noise:\n", noise.round(3).to_string(index=False))
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(noise.noise_sd, noise.debate_mismatch_rate, "o-", label="debate (all 3 speak)")
        ax.plot(noise.noise_sd, noise.majority_mismatch_rate, "s-", label="majority vote")
        ax.set_xlabel("noise std dev (margin units)"); ax.set_ylabel("P(winner != true full-agent winner)")
        ax.set_title(f"Noisy agents - {inv_id}"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / f"stress_noise_{inv_id}.png", dpi=150); plt.close(fig)

        miss = test_missing_agent(world, beta, sample_pairs)
        miss.to_csv(OUT / f"stress_missing_agent_{inv_id}.csv", index=False)
        print("Missing agent:\n", miss.round(3).to_string(index=False))

        early = test_early_stopping(world, beta, sample_pairs)
        early.to_csv(OUT / f"stress_early_stop_{inv_id}.csv", index=False)
        print("Early stopping:\n", early.to_string(index=False))

    print("\nAll stress tests done. Winner mismatches under early stopping must be 0 for every bound -",
          "checked above and already proven for random worlds in tests/test_engine.py.")


if __name__ == "__main__":
    main()
