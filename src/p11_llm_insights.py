"""Phase 11 insights: compare the LLM debate's real decisions against the math engine's, on the same
pairs, and visualise where they agree and where they genuinely diverge -- and why.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_companies, load_embeddings, load_labels
from config import DATA, OUT

PCOLORS = {"Finance": "#4C72B0", "ESG": "#55A868", "Risk": "#C44E52"}
PAIRS = [("NVDA", "VZ"), ("XOM", "HD"), ("MSFT", "NVDA")]


def math_engine_margins(a, b, investor="balanced"):
    d = np.load(DATA / "directions.npz")
    D, tickers = d["D"], list(d["tickers"])
    Z, names = load_embeddings()
    assert names == tickers
    inv = json.loads((DATA / "investors.json").read_text())
    beta = np.array([inv[investor]["beta"][k] for k in ["F", "E", "R"]])
    ia, ib = names.index(a), names.index(b)
    m = beta * (D @ (Z[ib] - Z[ia]))
    return {"Finance": m[0], "ESG": m[1], "Risk": m[2]}, m.sum()


def plot_agent_by_agent_comparison():
    """For each pair, each agent: what did the math engine's margin say, vs what did the real LLM
    agent say (its confidence score) -- side by side. This is the single most direct comparison:
    same perspective, same company pair, two completely different ways of forming an opinion."""
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(5 * len(PAIRS), 4.6), sharey=False)
    for ax, (a, b) in zip(axes, PAIRS):
        llm = json.loads((OUT / f"llm_debate_{a}_vs_{b}.json").read_text())
        math_m, math_total = math_engine_margins(a, b)
        agents = ["Finance", "ESG", "Risk"]
        x = np.arange(len(agents))
        width = 0.35

        # normalise the math margins to a -1..+1 scale for visual comparability with LLM confidence
        max_abs = max(abs(v) for v in math_m.values()) or 1.0
        math_norm = [math_m[ag] / max_abs for ag in agents]
        llm_conf = []
        for ag in agents:
            step = next(s for s in llm["transcript"] if s["agent"] == ag)
            c = step["confidence"] if step["confidence"] is not None else 0.0
            llm_conf.append(c)

        ax.bar(x - width / 2, math_norm, width, label="Math engine (scaled)", color="#888888")
        ax.bar(x + width / 2, llm_conf, width, label="Real LLM agent", color=[PCOLORS[ag] for ag in agents])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(agents)
        ax.set_ylim(-1.15, 1.15)
        ax.set_title(f"{a} vs {b}\nmath winner: {b if math_total > 0 else a}  |  LLM winner: {llm['winner']}")
        if ax is axes[0]:
            ax.set_ylabel("Favours A (-1) ↔ Favours B (+1)")
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Math engine vs. real LLM agents: same company pairs, same perspectives, independent opinions", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "llm_vs_math_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved llm_vs_math_comparison.png")


def print_plain_summary():
    print("\n=== PLAIN-LANGUAGE SUMMARY ===")
    for a, b in PAIRS:
        llm = json.loads((OUT / f"llm_debate_{a}_vs_{b}.json").read_text())
        math_m, math_total = math_engine_margins(a, b)
        math_winner = b if math_total > 0 else a
        agree = "AGREE" if math_winner == llm["winner"] else "DISAGREE"
        print(f"\n{a} vs {b}: math says {math_winner}, LLM says {llm['winner']}  -> {agree}")
        for step in llm["transcript"]:
            ag = step["agent"]
            mv = math_m[ag]
            print(f"  {ag:8s} math margin={mv:+.4f}  |  LLM verdict={step['verdict']} conf={step['confidence']}")


if __name__ == "__main__":
    plot_agent_by_agent_comparison()
    print_plain_summary()
