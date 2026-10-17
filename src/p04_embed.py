"""Phase 4: turn each company's 10-K text into ONE vector (an 'embedding').

Pipeline per company:
  clean text -> cut into chunks of ~CHUNK_WORDS words -> encode every chunk with a Sentence
  Transformer -> average the chunk vectors (mean pooling) -> L2-normalise (length 1).

Why chunk? The model only reads ~256 tokens at a time; longer text would be silently cut off.
We embed Item 1 (Business), Item 1A (Risk Factors) and both together, then check which is best.

Usage:  python p04_embed.py [model_name]      (default: sentence-transformers/all-MiniLM-L6-v2)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from config import DATA, EMB, RAW_TEXT, set_seeds

CHUNK_WORDS = 150           # ~200-230 word-pieces: stays under MiniLM's 256-token window
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def clean(text: str) -> str:
    """Drop page furniture (page numbers, 'Table of Contents') and collapse whitespace."""
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or re.fullmatch(r"\d{1,3}", s) or s.lower() in {"table of contents", "financial table of contents"}:
            continue
        lines.append(s)
    return re.sub(r"\s+", " ", " ".join(lines))


def chunk(text: str, n_words: int = CHUNK_WORDS):
    w = text.split()
    chunks = [" ".join(w[i:i + n_words]) for i in range(0, len(w), n_words)]
    if len(chunks) > 1 and len(chunks[-1].split()) < n_words // 4:   # merge a tiny tail
        chunks[-2] += " " + chunks.pop()
    return chunks


def l2(v):
    return v / np.linalg.norm(v)


def main(model_name: str = DEFAULT_MODEL):
    set_seeds()
    tag = model_name.split("/")[-1].lower().replace("-", "_")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    print(f"model={model_name} device={device} max_seq_length={model.max_seq_length}")

    meta = pd.read_csv(RAW_TEXT / "text_metadata.csv")
    comps = pd.read_csv(DATA / "companies.csv", dtype={"cik": str})
    ok = [t for t in comps["ticker"] if (RAW_TEXT / f"{t}_item1.txt").exists() and (RAW_TEXT / f"{t}_item1a.txt").exists()]
    print(f"{len(ok)} companies with both sections (others dropped): {sorted(set(comps['ticker']) - set(ok)) or 'none'}")

    out = {"item1": [], "item1a": [], "both": []}
    stats = []
    tok = model.tokenizer
    for t in ok:
        parts = {}
        for key in ("item1", "item1a"):
            txt = clean((RAW_TEXT / f"{t}_{key}.txt").read_text(encoding="utf-8"))
            parts[key] = chunk(txt)
        vecs = {k: model.encode(v, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
                                show_progress_bar=False) for k, v in parts.items()}
        out["item1"].append(l2(vecs["item1"].mean(axis=0)))
        out["item1a"].append(l2(vecs["item1a"].mean(axis=0)))
        out["both"].append(l2(np.vstack([vecs["item1"], vecs["item1a"]]).mean(axis=0)))
        # how many chunks would exceed the model window (silent truncation check)?
        sample = parts["item1"][:20] + parts["item1a"][:20]
        n_tok = [len(tok(s, add_special_tokens=True)["input_ids"]) for s in sample]
        stats.append({"ticker": t, "chunks_item1": len(parts["item1"]), "chunks_item1a": len(parts["item1a"]),
                      "pct_chunks_over_window": round(100 * np.mean([n > model.max_seq_length for n in n_tok]), 1)})
        print(f"{t}: chunks {len(parts['item1'])}+{len(parts['item1a'])}  over-window {stats[-1]['pct_chunks_over_window']}%")

    for key, arr in out.items():
        np.save(EMB / f"{tag}__{key}.npy", np.vstack(arr))
    (EMB / f"{tag}__names.json").write_text(json.dumps(ok))
    (EMB / f"{tag}__settings.json").write_text(json.dumps(
        {"model": model_name, "chunk_words": CHUNK_WORDS, "pooling": "mean of chunk vectors, then L2-normalised",
         "device": device, "n_companies": len(ok), "torch": torch.__version__}, indent=2))
    pd.DataFrame(stats).to_csv(EMB / f"{tag}__chunk_stats.csv", index=False)
    print(f"saved embeddings for {len(ok)} companies -> {EMB}  (tag={tag})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
