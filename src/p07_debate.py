"""Phase 7: run the debate engine on REAL company embeddings and directions.

For every investor (presets, plus the user's own if data/investors.json has one):
  1. full-agent ranking (S_ALL)
  2. ONE fully worked pair debate, logged as a human-readable trace
  3. a full tournament over all 30 companies (queue method)
  4. baselines: majority vote, single-agent rankings, top-pick plurality
  5. identity check on every debated pair (fails loudly if broken)

Outputs: outputs/debate_trace_<investor>.txt, outputs/full_agent_ranking_<investor>.csv,
         outputs/tournament_<investor>.json, outputs/baseline_comparison.csv
"""
import json

import numpy as np
import pandas as pd

import engine as E
from common import DEFAULT_TAG, DEFAULT_VARIANT, PERSPECTIVES, load_companies, load_embeddings
from config import DATA, OUT, set_seeds
from p06_investor import load_directions


def build_world():
    Z, names = load_embeddings(DEFAULT_TAG, DEFAULT_VARIANT)
    D, G, d_names, meta = load_directions()
    assert names == d_names, "embedding / direction order mismatch"
    return E.World(names, Z, D), meta


def investors():
    data = json.loads((DATA / "investors.json").read_text())
    return {k: np.array([v["beta"][p] for p in PERSPECTIVES]) for k, v in data.items()}


def run_for_investor(world, inv_id, beta, comps):
    s_all = E.full_scores(world, beta)
    rank = pd.DataFrame({"ticker": world.names, "sector": [comps.loc[t, "sector"] for t in world.names],
                         "S_ALL": s_all}).sort_values("S_ALL", ascending=False).reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    rank.to_csv(OUT / f"full_agent_ranking_{inv_id}.csv", index=False)

    # ---- one fully worked debate: the #2 vs #3 company (close, more interesting than the runaway #1) ----
    ia, ib = world.idx(rank.loc[1, "ticker"]), world.idx(rank.loc[2, "ticker"])
    res = E.debate_pair(world, beta, ia, ib)
    E.check_identity(res)
    trace = [f"Investor: {inv_id}  beta = {dict(zip(PERSPECTIVES, beta.round(3)))}",
            f"Agreement matrix G (this run's REAL learned directions, not the whiteboard's illustration):",
            pd.DataFrame(world.G, index=PERSPECTIVES, columns=PERSPECTIVES).round(3).to_string(), "",
            E.format_pair_trace(world, beta, res, ia, ib, 1)]

    # ---- margins bar chart + running evidence for this worked debate ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    cols = ["#4C72B0" if m > 0 else "#C44E52" for m in res.margins]
    ax1.bar([E.AGENT_NAME[E.AGENTS[k]] for k in range(3)], res.margins, color=cols)
    ax1.axhline(0, color="k", lw=0.8); ax1.set_title(f"Margins: {res.a} vs {res.b}\n(+ favours {res.b})")
    ax2.plot(range(1, len(res.order) + 1), res.running, "o-", color="#4C72B0")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(range(1, len(res.order) + 1))
    ax2.set_xticklabels([E.AGENT_NAME[E.AGENTS[k]] for k in res.order], rotation=20)
    ax2.set_title("Running evidence"); fig.tight_layout()
    fig.savefig(OUT / f"worked_debate_{inv_id}.png", dpi=150); plt.close(fig)

    # ---- tournament over all 30 companies ----
    all_idx = list(range(len(world.names)))
    rng = np.random.default_rng(0)
    queue = list(rng.permutation(all_idx))
    t = E.tournament(world, beta, queue)
    trace += ["", "=" * 70, "TOURNAMENT (queue method, all 30 companies)", "=" * 70,
             E.format_tournament_trace(world, beta, t)]
    for s in t.steps:
        E.check_identity(s)
    assert t.winner == rank.loc[0, "ticker"], \
        f"tournament winner {t.winner} != full-agent #1 {rank.loc[0, 'ticker']}"
    trace.append(f"\nCheck: tournament winner ({t.winner}) == full-agent top-ranked company "
                 f"({rank.loc[0, 'ticker']})  OK")

    (OUT / f"debate_trace_{inv_id}.txt").write_text("\n".join(trace), encoding="utf-8")
    json.dump({"winner": t.winner, "bracket": t.rounds, "messages": t.messages,
              "steps": [{"a": s.a, "b": s.b, "winner": s.winner, "final_evidence": s.final,
                        "margins": dict(zip(PERSPECTIVES, s.margins.round(4).tolist()))} for s in t.steps]},
              open(OUT / f"tournament_{inv_id}.json", "w"), indent=2)
    return rank, t, res


def baselines(world, inv_id, beta, full_rank, comps):
    """(a) full agent [already have], (b) majority vote tournament, (c) single-agent-only rankings,
    (d) top-pick plurality on the whole set. Compare winners and agreement with the full agent."""
    rows = []
    all_idx = list(range(len(world.names)))
    rng = np.random.default_rng(0)
    queue = list(rng.permutation(all_idx))

    t_full = E.tournament(world, beta, queue)
    rows.append({"investor": inv_id, "method": "full_agent", "winner": t_full.winner, "messages": t_full.messages})

    t_maj = E.tournament(world, beta, queue, decide=lambda ia, ib: E.decide_majority(world, beta, ia, ib))
    rows.append({"investor": inv_id, "method": "majority_vote", "winner": t_maj.winner, "messages": t_maj.messages})

    for k, name in enumerate(PERSPECTIVES):
        tk = E.tournament(world, beta, queue, decide=lambda ia, ib, k=k: E.decide_single(world, k, ia, ib))
        rows.append({"investor": inv_id, "method": f"single_{name}", "winner": tk.winner, "messages": tk.messages})

    plurality = E.top_pick_plurality(world, beta, all_idx)
    rows.append({"investor": inv_id, "method": "top_pick_plurality", "winner": plurality, "messages": None})

    agree = {r["method"]: (r["winner"] == t_full.winner) for r in rows}
    for r in rows:
        r["agrees_with_full_agent_winner"] = agree[r["method"]]
    return rows


def main():
    set_seeds()
    world, meta = build_world()
    comps = load_companies()
    inv = investors()
    print(f"World: {len(world.names)} companies, directions from {meta['embedding_tag']}/{meta['variant']}")
    print("Investors:", list(inv.keys()))

    all_baseline_rows = []
    for inv_id, beta in inv.items():
        print(f"\n--- {inv_id}  beta={dict(zip(PERSPECTIVES, beta.round(3)))} ---")
        rank, t, res = run_for_investor(world, inv_id, beta, comps)
        print(rank.head(5).to_string(index=False))
        print(f"Worked debate: {res.a} vs {res.b} -> {res.winner}  (final evidence {res.final:+.4f})")
        print(f"Tournament winner: {t.winner}  ({t.messages} agent messages)")
        all_baseline_rows += baselines(world, inv_id, beta, rank, comps)

    bdf = pd.DataFrame(all_baseline_rows)
    bdf.to_csv(OUT / "baseline_comparison.csv", index=False)
    print("\nBaseline comparison (does the method agree with the full agent's tournament winner?):")
    print(bdf.pivot(index="method", columns="investor", values="agrees_with_full_agent_winner").to_string())

    # how often does the winner change between investors? (I001 / I002 style check). The tournament
    # WINNER can coincide by chance even when investors disagree a lot overall, so also compare their
    # FULL rankings (Spearman correlation) -- this is the more informative, honest comparison.
    from scipy.stats import spearmanr
    winners = {k: json.load(open(OUT / f"tournament_{k}.json"))["winner"] for k in inv}
    ranks = {k: E.full_scores(world, b).argsort().argsort() for k, b in inv.items()}
    ids = list(inv)
    corr = pd.DataFrame(np.eye(len(ids)), index=ids, columns=ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            rho = spearmanr(ranks[ids[i]], ranks[ids[j]]).statistic
            corr.iloc[i, j] = corr.iloc[j, i] = rho
    corr.to_csv(OUT / "investor_ranking_correlation.csv")
    print("\nWinner per investor:", winners)
    print("All investors agree on the WINNER:", len(set(winners.values())) == 1)
    print("But their FULL rankings' Spearman correlation (1.0 = identical ranking):")
    print(corr.round(3).to_string())
    json.dump({"winners_by_investor": winners, "n_distinct_winners": len(set(winners.values())),
              "full_ranking_spearman": corr.round(3).to_dict()},
              open(OUT / "investor_winner_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
