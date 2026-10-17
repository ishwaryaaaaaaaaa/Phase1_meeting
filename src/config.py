"""Shared settings: folders, random seed, SEC request headers.

Everything else imports from here so paths and the seed live in ONE place.
"""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_TEXT = DATA / "raw_text"
LABELS = DATA / "labels"
EMB = DATA / "embeddings"
SEC_CACHE = DATA / "sec_cache"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

for _p in (RAW_TEXT, LABELS, EMB, SEC_CACHE, OUT, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Make every run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def sec_headers() -> dict:
    """SEC fair-access rules require a descriptive User-Agent with a contact email.

    The email is read from the environment (never written to any file):
      SEC_EMAIL  (preferred)  or  CLAUDE_CODE_USER_EMAIL.
    """
    email = os.environ.get("SEC_EMAIL") or os.environ.get("CLAUDE_CODE_USER_EMAIL")
    if not email:
        raise RuntimeError(
            "Set SEC_EMAIL to your email address (SEC requires a contact in the User-Agent)."
        )
    return {"User-Agent": f"Ishwarya Mohan {email}", "Accept-Encoding": "gzip, deflate"}


# SEC allows up to 10 requests/second; we stay far below that.
SEC_MIN_INTERVAL = 0.25  # seconds between requests (= 4 requests / second)
