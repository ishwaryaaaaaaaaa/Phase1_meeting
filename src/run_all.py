"""Single entry point: reproduces the whole pipeline from scratch (except the interactive
investor questionnaire, which needs a human and is run separately: p06_investor.py --interview).

    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ["p01_companies.py"],
    ["p02_fetch_10k.py"],
    ["p03_labels.py"],
    ["p04_embed.py", "sentence-transformers/all-MiniLM-L6-v2"],
    ["p04_embed.py", "BAAI/bge-small-en-v1.5"],
    ["p04b_sanity.py", "all_minilm_l6_v2"],
    ["p04b_sanity.py", "bge_small_en_v1.5"],
    ["p05_directions.py"],
    ["p06_investor.py"],           # presets only; add --interview separately for your own answers
    ["p07_debate.py"],
    ["p08_stress.py"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n{'=' * 70}\n>>> {' '.join(step)}\n{'=' * 70}")
        r = subprocess.run([sys.executable, *step])
        if r.returncode != 0:
            print(f"STEP FAILED: {step} (exit {r.returncode}) - see PROGRESS.md / logged output above.")
            sys.exit(r.returncode)
    print("\nAll steps completed. See outputs/ and reports/.")
