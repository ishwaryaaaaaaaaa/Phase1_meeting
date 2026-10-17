"""Phase 4 sanity checks on the embeddings.

1. Nearest neighbours by cosine similarity should mostly be same-sector companies.
2. 2-D PCA map coloured by sector.
3. Cosine-similarity heatmap (companies grouped by sector).

Usage: python p04b_sanity.py [tag]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import DEFAULT_TAG, load_companies, load_embeddings
from config import OUT, set_seeds


def same_sector_rate(Z, sectors, k=3):
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    hits, total = 0, 0
    for i in range(len(Z)):
        nn = np.argsort(-S[i])[:k]
        hits += sum(sectors[j] == sectors[i] for j in nn)
        total += k
    return hits / total


def random_baseline(sectors, k=3):
    """Expected same-sector rate if neighbours were picked at random."""
    n = len(sectors)
    s = pd.Series(sectors).value_counts()
    return float(np.mean([(s[sectors[i]] - 1) / (n - 1) for i in range(n)]))


def main(tag=DEFAULT_TAG):
    set_seeds()
    comps = load_companies()
    res = {}
    for variant in ("item1", "item1a", "both"):
        Z, names = load_embeddings(tag, variant)
        sectors = [comps.loc[t, "sector"] for t in names]
        res[variant] = {"top3_same_sector": same_sector_rate(Z, sectors, 3),
                        "top1_same_sector": same_sector_rate(Z, sectors, 1),
                        "random_baseline": random_baseline(sectors)}
    print(f"[{tag}] nearest-neighbour same-sector rate")
    print(pd.DataFrame(res).T.round(3))

    Z, names = load_embeddings(tag, "both")
    sectors = [comps.loc[t, "sector"] for t in names]
    # nearest neighbours table
    S = Z @ Z.T
    rows = []
    for i, t in enumerate(names):
        order = [j for j in np.argsort(-S[i]) if j != i][:3]
        rows.append({"ticker": t, "sector": sectors[i],
                     "nn1": f"{names[order[0]]} ({sectors[order[0]]})", "cos1": round(S[i, order[0]], 3),
                     "nn2": f"{names[order[1]]} ({sectors[order[1]]})",
                     "nn3": f"{names[order[2]]} ({sectors[order[2]]})"})
    nn = pd.DataFrame(rows)
    nn.to_csv(OUT / f"nearest_neighbours_{tag}.csv", index=False)
    json.dump(res, open(OUT / f"nn_sector_rate_{tag}.json", "w"), indent=2)

    # PCA map
    P = PCA(n_components=2, random_state=0).fit(Z)
    X = P.transform(Z)
    uniq = sorted(set(sectors))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7))
    for si, s in enumerate(uniq):
        idx = [i for i, x in enumerate(sectors) if x == s]
        ax.scatter(X[idx, 0], X[idx, 1], s=70, color=cmap(si), label=s, edgecolor="k", linewidth=0.4)
    for i, t in enumerate(names):
        ax.annotate(t, (X[i, 0], X[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"Company embeddings, 2-D PCA ({tag})\nexplained variance: "
                 f"{P.explained_variance_ratio_[0]:.0%} + {P.explained_variance_ratio_[1]:.0%}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(OUT / f"pca_map_{tag}.png", dpi=150); plt.close(fig)

    # heatmap grouped by sector
    order = sorted(range(len(names)), key=lambda i: (sectors[i], names[i]))
    M = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order], fontsize=7)
    fig.colorbar(im, label="cosine similarity")
    ax.set_title(f"Cosine similarity between companies (grouped by sector) - {tag}")
    fig.tight_layout(); fig.savefig(OUT / f"similarity_heatmap_{tag}.png", dpi=150); plt.close(fig)
    print("saved plots + tables to", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAG)
