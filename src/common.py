"""Loaders shared by every later phase (embeddings, labels, companies)."""
import json

import numpy as np
import pandas as pd

from config import DATA, EMB

PERSPECTIVES = ["F", "E", "R"]
PERSP_NAMES = {"F": "Finance", "E": "ESG", "R": "Risk"}
LABEL_COL = {"F": "finance_label", "E": "esg_label", "R": "risk_label"}

DEFAULT_TAG = "all_minilm_l6_v2"
DEFAULT_VARIANT = "both"        # Item 1 + Item 1A together (chosen in Phase 5, see DECISIONS.md)


def load_embeddings(tag: str = DEFAULT_TAG, variant: str = DEFAULT_VARIANT):
    """Return (Z, tickers): Z has one L2-normalised row per company."""
    Z = np.load(EMB / f"{tag}__{variant}.npy")
    names = json.loads((EMB / f"{tag}__names.json").read_text())
    return Z, names


def load_companies():
    return pd.read_csv(DATA / "companies.csv", dtype={"cik": str}).set_index("ticker")


def load_labels(tickers):
    lab = pd.read_csv(DATA / "labels.csv").set_index("ticker").loc[tickers]
    return lab


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)
