"""Phase 5: learn the three perspective directions, test them honestly, show G.

Honest testing = hold out WHOLE COMPANIES (never random pairs: a pair-level split would let the
model see a held-out company in other pairs and look better than it is). The regularisation
strength is chosen by an inner cross-validation on the training companies only (nested CV).

Outputs (outputs/): direction_cv_results.csv, direction_sector_baseline.csv,
direction_C_sensitivity.csv/.png, G_matrix.csv/.png, direction_bootstrap.json, label_vs_sector_R2.csv
Saved model (data/): directions.npz + directions_meta.json  <- used by every later phase
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import directions as dr
from common import (LABEL_COL, PERSP_NAMES, PERSPECTIVES, load_companies, load_embeddings, load_labels)
from config import DATA, EMB, OUT, SEED, set_seeds

warnings.filterwarnings("ignore")
K_FOLDS, REPEATS = 5, 8


def cv_eval(Z, y, method, seed=SEED, hp_fixed=None):
    """Outer CV over companies. Returns mean/sd (over repeats) of same-fold pair accuracy and of the
    pooled out-of-fold Spearman correlation between score and label."""
    fit, _grid = dr.METHODS[method]
    n = len(y)
    accs, rhos, hps = [], [], []
    for r in range(REPEATS):
        rng = np.random.RandomState(seed + r)
        folds = np.array_split(rng.permutation(n), K_FOLDS)
        oof = np.zeros(n)
        good = tot = 0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            hp = hp_fixed if hp_fixed is not None else dr.inner_select(method, Z, y, tr, rng)
            hps.append(hp)
            d = fit(Z, y, tr, hp)
            s = Z @ d
            oof[f] = s[f]
            for p in range(len(f)):
                for q in range(p + 1, len(f)):
                    a, b = f[p], f[q]
                    if abs(y[a] - y[b]) > dr.GAP_THRESHOLD:
                        tot += 1
                        good += (s[a] - s[b]) * (y[a] - y[b]) > 0
        accs.append(good / tot)
        rhos.append(spearmanr(oof, y).statistic)
    return {"pair_acc_mean": np.mean(accs), "pair_acc_sd": np.std(accs),
            "spearman_mean": np.mean(rhos), "spearman_sd": np.std(rhos),
            "hp_modal": (pd.Series([h for h in hps if h is not None]).mode().iloc[0] if any(h is not None for h in hps) else np.nan)}


def sector_baseline(y, sectors, seed=SEED):
    """Reference only: predict a held-out company's label by the mean label of training companies in
    the SAME sector (global mean if none). Shows how much of the labels is 'just sector'."""
    n = len(y)
    accs, rhos = [], []
    for r in range(REPEATS):
        rng = np.random.RandomState(seed + r)
        folds = np.array_split(rng.permutation(n), K_FOLDS)
        oof = np.zeros(n); good = tot = 0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            gm = y[tr].mean()
            sc = np.array([np.mean([y[j] for j in tr if sectors[j] == sectors[i]] or [gm]) for i in range(n)])
            oof[f] = sc[f]
            for p in range(len(f)):
                for q in range(p + 1, len(f)):
                    a, b = f[p], f[q]
                    if abs(y[a] - y[b]) > dr.GAP_THRESHOLD:
                        tot += 1
                        good += (sc[a] - sc[b]) * (y[a] - y[b]) > 0.0     # ties count as wrong
        accs.append(good / tot); rhos.append(spearmanr(oof, y).statistic)
    return np.mean(accs), np.std(accs), np.mean(rhos)


def main():
    set_seeds()
    comps = load_companies()
    tags = {"all_minilm_l6_v2": "MiniLM (baseline)", "bge_small_en_v1.5": "BGE-small (2nd model)"}
    variants = ["item1", "item1a", "both"]
    jobs = []
    for tag in tags:
        for var in variants:
            Z, names = load_embeddings(tag, var)
            lab = load_labels(names)
            for k in PERSPECTIVES:
                jobs.append((tag, var, k, "logistic", Z, lab[LABEL_COL[k]].values))
    print(f"running {len(jobs)} nested-CV jobs in parallel ...")
    res = Parallel(n_jobs=-1)(delayed(cv_eval)(Z, y, m) for (_t, _v, _k, m, Z, y) in jobs)
    rows = [{"model": t, "variant": v, "perspective": PERSP_NAMES[k], "method": m, **r}
            for (t, v, k, m, _Z, _y), r in zip(jobs, res)]
    cv = pd.DataFrame(rows)

    # ---- choose the primary embedding: MiniLM variant with best mean pair accuracy ----
    mini = cv[(cv.model == "all_minilm_l6_v2")].groupby("variant")["pair_acc_mean"].mean()
    best_var = mini.idxmax()
    print("\nMiniLM variant comparison (mean pair-accuracy over F/E/R):\n", mini.round(3).to_string())
    print("-> chosen variant:", best_var)

    # ---- compare the 3 fitting methods on the chosen embedding ----
    Z, names = load_embeddings("all_minilm_l6_v2", best_var)
    lab = load_labels(names)
    sectors = [comps.loc[t, "sector"] for t in names]
    jobs2 = [(m, k) for m in ("ridge", "meandiff") for k in PERSPECTIVES]
    res2 = Parallel(n_jobs=-1)(delayed(cv_eval)(Z, lab[LABEL_COL[k]].values, m) for m, k in jobs2)
    for (m, k), r in zip(jobs2, res2):
        cv.loc[len(cv)] = {"model": "all_minilm_l6_v2", "variant": best_var, "perspective": PERSP_NAMES[k],
                           "method": m, **r}
    cv.to_csv(OUT / "direction_cv_results.csv", index=False)
    print("\nCV results (held-out companies):")
    print(cv.round(3).to_string(index=False))

    # ---- sector-only baseline and share of label variance explained by sector ----
    sb, r2rows = [], []
    for k in PERSPECTIVES:
        y = lab[LABEL_COL[k]].values
        a, s, rho = sector_baseline(y, sectors)
        sb.append({"perspective": PERSP_NAMES[k], "sector_pair_acc": a, "sector_pair_acc_sd": s, "sector_spearman": rho})
        grp = pd.Series(y).groupby(sectors)
        ss_between = sum(len(g) * (g.mean() - y.mean()) ** 2 for _, g in grp)
        r2rows.append({"perspective": PERSP_NAMES[k], "R2_sector_explains_label": ss_between / ((y - y.mean()) ** 2).sum()})
    pd.DataFrame(sb).to_csv(OUT / "direction_sector_baseline.csv", index=False)
    pd.DataFrame(r2rows).to_csv(OUT / "label_vs_sector_R2.csv", index=False)
    print("\nSector-only reference baseline:\n", pd.DataFrame(sb).round(3).to_string(index=False))
    print("\nShare of label variance explained by sector (in-sample):\n", pd.DataFrame(r2rows).round(3).to_string(index=False))

    # ---- regularisation sensitivity (non-nested, per fixed C) + final C ----
    sens = []
    for k in PERSPECTIVES:
        y = lab[LABEL_COL[k]].values
        for C in dr.C_GRID:
            r = cv_eval(Z, y, "logistic", hp_fixed=C)
            sens.append({"perspective": PERSP_NAMES[k], "C": C, "pair_acc": r["pair_acc_mean"],
                         "pair_acc_sd": r["pair_acc_sd"], "spearman": r["spearman_mean"]})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "direction_C_sensitivity.csv", index=False)
    print("\nSensitivity of held-out accuracy to the regularisation C:\n",
          sens.pivot(index="C", columns="perspective", values="pair_acc").round(3).to_string())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p, g in sens.groupby("perspective"):
        ax.errorbar(g["C"], g["pair_acc"], yerr=g["pair_acc_sd"], marker="o", capsize=3, label=p)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xscale("log"); ax.set_xlabel("C (larger = weaker regularisation)")
    ax.set_ylabel("held-out pair accuracy"); ax.legend(); ax.set_title("Regularisation sensitivity")
    fig.tight_layout(); fig.savefig(OUT / "direction_C_sensitivity.png", dpi=150); plt.close(fig)

    best_C = {}
    for k in PERSPECTIVES:
        g = sens[sens.perspective == PERSP_NAMES[k]]
        best_C[k] = float(g.loc[g["pair_acc"].idxmax(), "C"])

    # ---- FINAL directions: fit on all companies with the chosen C ----
    D = np.vstack([dr.fit_logistic(Z, lab[LABEL_COL[k]].values, np.arange(len(names)), best_C[k]) for k in PERSPECTIVES])
    G = D @ D.T
    in_sample = {k: dr.pair_accuracy(Z @ D[i], lab[LABEL_COL[k]].values, range(len(names)))
                 for i, k in enumerate(PERSPECTIVES)}
    np.savez(DATA / "directions.npz", D=D, G=G, tickers=np.array(names), best_C=np.array([best_C[k] for k in PERSPECTIVES]))
    (DATA / "directions_meta.json").write_text(json.dumps(
        {"embedding_tag": "all_minilm_l6_v2", "variant": best_var, "perspectives": PERSPECTIVES,
         "best_C": best_C, "gap_threshold": dr.GAP_THRESHOLD, "in_sample_pair_accuracy": in_sample}, indent=2))
    Gdf = pd.DataFrame(G, index=[PERSP_NAMES[k] for k in PERSPECTIVES], columns=[PERSP_NAMES[k] for k in PERSPECTIVES])
    Gdf.to_csv(OUT / "G_matrix.csv")
    print("\nFinal agreement matrix G = cos(d_k, d_l):\n", Gdf.round(3).to_string())
    print("in-sample (training) pair accuracy, for the overfitting comparison:", {k: round(v, 3) for k, v in in_sample.items()})

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(G, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(3)); ax.set_xticklabels(Gdf.columns); ax.set_yticks(range(3)); ax.set_yticklabels(Gdf.index)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{G[i, j]:+.2f}", ha="center", va="center", color="k", fontsize=11)
    fig.colorbar(im, label="cosine"); ax.set_title("G: direction similarity between agents\n(negative = tend to challenge each other)")
    fig.tight_layout(); fig.savefig(OUT / "G_matrix.png", dpi=150); plt.close(fig)

    # ---- stability: bootstrap over companies; sensitivity of G to C ----
    rng = np.random.RandomState(SEED)
    boot_cos = {k: [] for k in PERSPECTIVES}; boot_G = []
    for b in range(100):
        idx = np.unique(rng.choice(len(names), len(names), replace=True))
        Db = np.vstack([dr.fit_logistic(Z, lab[LABEL_COL[k]].values, idx, best_C[k]) for k in PERSPECTIVES])
        for i, k in enumerate(PERSPECTIVES):
            boot_cos[k].append(float(Db[i] @ D[i]))
        boot_G.append((Db @ Db.T)[np.triu_indices(3, 1)])
    boot_G = np.array(boot_G)
    pairs = ["F-E", "F-R", "E-R"]
    G_by_C = {}
    for C in dr.C_GRID:
        Dc = np.vstack([dr.fit_logistic(Z, lab[LABEL_COL[k]].values, np.arange(len(names)), C) for k in PERSPECTIVES])
        G_by_C[C] = (Dc @ Dc.T)[np.triu_indices(3, 1)].round(3).tolist()
    boot = {"bootstrap_cosine_with_full_direction": {PERSP_NAMES[k]: [float(np.mean(v)), float(np.std(v))] for k, v in boot_cos.items()},
            "G_bootstrap_mean_sd": {p: [float(boot_G[:, i].mean()), float(boot_G[:, i].std())] for i, p in enumerate(pairs)},
            "G_entries_F-E_F-R_E-R_for_each_C": G_by_C}
    (OUT / "direction_bootstrap.json").write_text(json.dumps(boot, indent=2))
    print("\nBootstrap stability (cosine of resampled direction with full-data direction):", boot["bootstrap_cosine_with_full_direction"])
    print("G entries bootstrap mean/sd:", boot["G_bootstrap_mean_sd"])
    print("G entries (F-E, F-R, E-R) for each C:", G_by_C)
    print("\nDONE. chosen variant:", best_var, "| chosen C:", best_C)


if __name__ == "__main__":
    main()
