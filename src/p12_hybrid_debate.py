"""Phase 12: hybrid tournament -- math engine for every debate, except when the math margin is close
to zero (a near-coin-flip), where a real LLM debate (llm_debate.run_llm_debate) is called in as a
tiebreaker. This directly answers "can't we use the LLM and math together": cheap/deterministic math
for the whole 30-company bracket, real reasoning only spent on the handful of pairs where the learned
directions can't really tell the two companies apart.

A debate is "close" if |final_evidence| < CLOSE_THRESHOLD. Below that, the math margin is small enough
that embedding noise alone (see README §4.5, MSFT vs NVDA) could plausibly be deciding the winner.

Because the tournament is a queue (engine.tournament), overriding an early winner changes who meets whom
afterwards -- so this can surface close debates that never existed in the pure-math bracket. Every LLM
call made is logged, whether it changed the winner, and the final tournament winner is compared against
the existing pure-math run (outputs/tournament_balanced.json).
"""
import json

import numpy as np

import engine as E
from common import DEFAULT_TAG, DEFAULT_VARIANT, PERSP_NAMES, PERSPECTIVES, load_embeddings, load_labels
from config import DATA, OUT, set_seeds
from llm_debate import run_llm_debate
from p06_investor import load_directions

CLOSE_THRESHOLD = 0.03


def build_world():
    Z, names = load_embeddings(DEFAULT_TAG, DEFAULT_VARIANT)
    D, _G, d_names, _meta = load_directions()
    assert names == d_names, "embedding / direction order mismatch"
    return E.World(names, Z, D)


def margin_hints(res):
    """This debate's math margins, one per agent, phrased as a hint for that agent's own LLM prompt."""
    hints = {}
    for p, m in zip(PERSPECTIVES, res.margins):
        side = res.b if m > 0 else res.a
        hints[PERSP_NAMES[p]] = f"margin {m:+.4f} (favours {side})"
    return hints


def make_hybrid_decider(world, beta, labels, log):
    def decide(ia, ib):
        res = E.debate_pair(world, beta, ia, ib)
        E.check_identity(res)
        if abs(res.final) >= CLOSE_THRESHOLD:
            return res

        print(f"  [close call, |evidence|={abs(res.final):.4f}] running real LLM debate: {res.a} vs {res.b}")
        llm = run_llm_debate(res.a, res.b, labels=labels, model_hints=margin_hints(res))
        overridden = llm["winner"] != res.winner
        log.append({"a": res.a, "b": res.b, "math_winner": res.winner, "math_final_evidence": res.final,
                    "llm_winner": llm["winner"], "llm_final_evidence": llm["final_evidence"],
                    "overridden": overridden})
        if not overridden:
            return res
        new_winner, new_loser = llm["winner"], res.a if llm["winner"] == res.b else res.b
        return E.PairResult(res.a, res.b, new_winner, new_loser, res.margins, res.order, res.running,
                            res.final, res.tie, res.full_diff, res.messages)
    return decide


def main():
    set_seeds()
    world = build_world()
    investors = json.loads((DATA / "investors.json").read_text())
    beta = np.array([investors["balanced"]["beta"][p] for p in PERSPECTIVES])
    labels = load_labels(world.names)

    all_idx = list(range(len(world.names)))
    rng = np.random.default_rng(0)       # same seed p07_debate.py uses -- same starting queue
    queue = list(rng.permutation(all_idx))

    log = []
    t = E.tournament(world, beta, queue, decide=make_hybrid_decider(world, beta, labels, log))

    pure_math = json.loads((OUT / "tournament_balanced.json").read_text())

    print(f"\nPure-math tournament winner:   {pure_math['winner']}")
    print(f"Hybrid tournament winner:      {t.winner}")
    print(f"Close calls sent to the LLM:   {len(log)}")
    for entry in log:
        flag = "OVERRIDDEN" if entry["overridden"] else "confirmed"
        print(f"  {entry['a']} vs {entry['b']}: math={entry['math_winner']} llm={entry['llm_winner']}  [{flag}]")
    changed = t.winner != pure_math["winner"]
    print(f"\nFinal winner changed by using the LLM as tiebreaker: {changed}")

    out = {"winner": t.winner, "pure_math_winner": pure_math["winner"], "winner_changed": changed,
          "close_threshold": CLOSE_THRESHOLD, "llm_checks": log, "messages": t.messages,
          "bracket": t.rounds,
          "steps": [{"a": s.a, "b": s.b, "winner": s.winner, "final_evidence": s.final,
                    "margins": dict(zip(PERSPECTIVES, s.margins.round(4).tolist()))} for s in t.steps]}
    (OUT / "hybrid_tournament_balanced.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSaved outputs/hybrid_tournament_balanced.json")


if __name__ == "__main__":
    main()
