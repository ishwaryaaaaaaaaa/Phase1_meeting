"""Two summary plots over the real tournament results (outputs/tournament_*.json), for the README:

1. tournament_progression.png — as the single-elimination tournament runs (29 debates to go from
   30 companies down to 1), how strong is the company CURRENTLY in the lead, for each investor?
   Plotted as that leader's own full-agent score (S_ALL under that investor's beta) after each debate.
2. final_choice_by_investor.png — which company did each investor's tournament end on, and what was
   its score?

Both are built entirely from files p07_debate.py already produced — no new experiment is run here.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import DATA, OUT

INVESTORS = ["finance_heavy", "esg_heavy", "balanced"]
COLORS = {"finance_heavy": "#4C72B0", "esg_heavy": "#55A868", "balanced": "#C44E52"}
LABELS = {"finance_heavy": "Finance-heavy", "esg_heavy": "ESG-heavy", "balanced": "Balanced"}


def leader_scores(inv_id):
    """For each debate step 1..29, the ticker that just won and ITS OWN full-agent score
    (under this investor's beta) -- i.e. how strong is whoever is currently undefeated."""
    t = json.load(open(OUT / f"tournament_{inv_id}.json"))
    rank = pd.read_csv(OUT / f"full_agent_ranking_{inv_id}.csv").set_index("ticker")["S_ALL"]
    steps, scores, leaders = [], [], []
    for i, s in enumerate(t["steps"], 1):
        steps.append(i)
        scores.append(rank[s["winner"]])
        leaders.append(s["winner"])
    return steps, scores, leaders, t["winner"]


def plot_progression():
    fig, ax = plt.subplots(figsize=(9, 5.6))
    # stagger end-of-line labels vertically so they don't overlap when two investors
    # end on nearly the same score (finance_heavy and balanced both land near 0.15)
    end_offsets = {"finance_heavy": 11, "esg_heavy": 0, "balanced": -13}
    for inv_id in INVESTORS:
        steps, scores, leaders, final_winner = leader_scores(inv_id)
        ax.plot(steps, scores, "o-", color=COLORS[inv_id], label=LABELS[inv_id], markersize=4, linewidth=1.6)
        ax.annotate(final_winner, (steps[-1], scores[-1]), textcoords="offset points",
                   xytext=(8, end_offsets[inv_id]), fontsize=10, fontweight="bold",
                   color=COLORS[inv_id], va="center")
    ax.set_xlabel("Debate number (1 → 29, single-elimination over 30 companies)")
    ax.set_ylabel("Current tournament leader's own full-agent score ($S_{ALL}$)")
    ax.set_title("Tournament progression: how strong is the current leader, debate by debate")
    ax.text(0.5, -0.17,
           "Zigzags because the queue order is randomised, not sorted by strength — a fresh,\n"
           "lower-ranked challenger can enter at any point and temporarily pull the line down.",
           transform=ax.transAxes, ha="center", fontsize=9, color="#555555")
    ax.set_xlim(0, 32)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUT / "tournament_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_final_choice():
    fig, ax = plt.subplots(figsize=(7, 5))
    xs, scores, winners = [], [], []
    for inv_id in INVESTORS:
        rank = pd.read_csv(OUT / f"full_agent_ranking_{inv_id}.csv")
        top = rank.iloc[0]
        xs.append(LABELS[inv_id]); scores.append(top["S_ALL"]); winners.append(top["ticker"])

    bars = ax.bar(xs, scores, color=[COLORS[i] for i in INVESTORS], width=0.55)
    for bar, score, ticker in zip(bars, scores, winners):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, ticker,
               ha="center", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{score:.3f}",
               ha="center", va="center", fontsize=10, color="white")
    ax.set_ylabel("Winner's full-agent score ($S_{ALL}$)")
    ax.set_title("Tournament winner chosen by each investor profile\n(29 debates, 87 agent messages each)")
    ax.set_ylim(0, max(scores) * 1.25)
    fig.tight_layout()
    path = OUT / "final_choice_by_investor.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    plot_progression()
    plot_final_choice()
