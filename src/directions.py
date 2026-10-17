"""Learn ONE direction per perspective from pairwise comparisons.

Idea (plain words): a 'direction' d_k is an arrow in the embedding space. Projecting a company's
vector z onto it, z . d_k, gives that company's score on perspective k. We choose the arrow so that
it ranks pairs of companies the way the labels do: if company a beats company b on Finance,
then (z_a - z_b) . d_F should be positive.

Three ways to get the arrow (compared in Phase 5):
  logistic : L2-regularised logistic regression on pairs, no intercept  (the main method)
  ridge    : ridge regression of the label on the embedding
  meandiff : mean embedding of the top third minus mean of the bottom third
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

GAP_THRESHOLD = 0.25        # ignore pairs whose label gap is below this (in z-score units): near-ties
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]       # logistic: bigger C = weaker regularisation
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]     # ridge:    bigger alpha = stronger regularisation


def unit(v):
    return v / np.linalg.norm(v)


def pair_dataset(Z, y, idx, thr=GAP_THRESHOLD):
    """All pairs among companies `idx` with |label gap| > thr. Both orientations are included so the
    two classes are balanced: feature z_a - z_b, target 1 if a is better than b."""
    X, t = [], []
    idx = list(idx)
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            a, b = idx[p], idx[q]
            if abs(y[a] - y[b]) > thr:
                better = 1 if y[a] > y[b] else 0
                X.append(Z[a] - Z[b]); t.append(better)
                X.append(Z[b] - Z[a]); t.append(1 - better)
    return np.array(X), np.array(t)


def fit_logistic(Z, y, idx, C):
    X, t = pair_dataset(Z, y, idx)
    s = np.median(np.linalg.norm(X, axis=1))          # rescale so C means the same thing for any embedding
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000)
    clf.fit(X / s, t)
    return unit(clf.coef_[0])


def fit_ridge(Z, y, idx, alpha):
    idx = list(idx)
    m = Ridge(alpha=alpha, fit_intercept=True).fit(Z[idx], y[idx])
    return unit(m.coef_)


def fit_meandiff(Z, y, idx, _hp=None):
    idx = np.array(list(idx))
    order = idx[np.argsort(y[idx])]
    k = max(2, len(order) // 3)
    return unit(Z[order[-k:]].mean(axis=0) - Z[order[:k]].mean(axis=0))


METHODS = {
    "logistic": (fit_logistic, C_GRID),
    "ridge": (fit_ridge, ALPHA_GRID),
    "meandiff": (fit_meandiff, [None]),
}


def pair_accuracy(scores, y, idx, thr=GAP_THRESHOLD):
    """Fraction of company pairs (among idx, label gap > thr) that the scores order like the labels."""
    idx = list(idx)
    good = tot = 0
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            a, b = idx[p], idx[q]
            if abs(y[a] - y[b]) > thr:
                tot += 1
                good += (scores[a] - scores[b]) * (y[a] - y[b]) > 0
    return (good / tot) if tot else np.nan


def inner_select(method, Z, y, train_idx, rng, n_splits=3, repeats=2):
    """Pick the hyper-parameter using ONLY the training companies (nested CV, no peeking)."""
    fit, grid = METHODS[method]
    if len(grid) == 1:
        return grid[0]
    train_idx = np.array(train_idx)
    best, best_acc = grid[0], -1
    for hp in grid:
        accs = []
        for _ in range(repeats):
            perm = rng.permutation(train_idx)
            folds = np.array_split(perm, n_splits)
            for f in folds:
                tr = np.setdiff1d(train_idx, f)
                d = fit(Z, y, tr, hp)
                accs.append(pair_accuracy(Z @ d, y, f))
        acc = np.nanmean(accs)
        if acc > best_acc:
            best, best_acc = hp, acc
    return best
