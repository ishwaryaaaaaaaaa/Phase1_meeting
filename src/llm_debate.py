"""Phase 11 (new): the SAME debate idea, but with real agents instead of dot products.

The math engine (engine.py) computes each agent's "opinion" as one number (a dot product) and never
lets agents see each other -- proven order-independent, nothing is ever revised (confirmed in chat and
in tests/test_engine.py). This module answers "what if each agent were an actual LLM instead": a real
model, given a real system prompt defining its perspective, shown the REAL underlying company data (not
an embedding), asked for a verdict + a confidence score, and -- unlike the math version -- shown the
PRECEDING agents' actual written arguments before it answers, so real revision becomes possible.

This is a genuinely different experiment from the math engine, not a reimplementation of it: no
guarantee here that the LLM debate matches any "full agent," order can genuinely change the outcome,
and every call costs real API time. Kept deliberately small (a handful of real pairs) for exactly that
reason.
"""
import json
import os
import re
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from common import load_companies, load_labels
from config import OUT, set_seeds

load_dotenv()

AGENTS = ["Finance", "ESG", "Risk"]

SYSTEM_PROMPTS = {
    "Finance": (
        "You are a Finance analyst agent in a multi-agent investment debate. You evaluate companies "
        "ONLY on financial performance -- revenue growth and profitability. You do not consider ESG "
        "or risk factors; that is not your job, and other agents will cover those."
    ),
    "ESG": (
        "You are an ESG analyst agent in a multi-agent investment debate. You evaluate companies ONLY "
        "on their environmental, social and governance risk profile. You do not consider financial "
        "performance or market risk; other agents will cover those."
    ),
    "Risk": (
        "You are a Risk analyst agent in a multi-agent investment debate. You evaluate companies ONLY "
        "on market/price risk -- volatility and drawdown history. You do not consider financial "
        "performance or ESG; other agents will cover those."
    ),
}

FACTS = {
    "Finance": lambda row: (f"3-year revenue growth (CAGR): {row['rev_cagr_3y']:+.1%}\n"
                            f"Profit margin: {row['profit_margin']:.1%}"),
    "ESG": lambda row: (f"Sustainalytics ESG risk score: {row['esg_risk_score']:.1f} (lower is better)\n"
                        f"Controversy level: {row['esg_controversy']}"),
    "Risk": lambda row: (f"Annualised price volatility (3y): {row['ann_vol']:.1%}\n"
                         f"Maximum drawdown (3y): {row['max_drawdown']:.1%}"),
}

VERDICT_RE = re.compile(r"FINAL:\s*([AB])", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*([+-]?\d*\.?\d+)")


def _client():
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])


def _model():
    return os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def _call(messages, max_tokens=700, max_retries=5):
    client = _client()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(model=_model(), messages=messages, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 15.0
            m = re.search(r"try again in ([\d.]+)s", str(e), re.IGNORECASE)
            if m:
                wait = float(m.group(1)) + 1.0
            print(f"  [retry {attempt + 1}/{max_retries}] waiting {wait:.0f}s after: {e}", flush=True)
            time.sleep(wait)


def agent_prompt(agent, tickerA, tickerB, rowA, rowB, prior_arguments):
    """Build this agent's user prompt. `prior_arguments` is a list of (agent_name, verdict_text) for
    whichever agents already spoke this debate -- empty for the first speaker."""
    facts = FACTS[agent]
    body = (
        f"Evaluate these two companies and say which is better on YOUR perspective only.\n\n"
        f"Company A -- {tickerA}:\n{facts(rowA)}\n\n"
        f"Company B -- {tickerB}:\n{facts(rowB)}\n\n"
    )
    if prior_arguments:
        body += "Other agents have already given their views in this debate:\n\n"
        for name, text in prior_arguments:
            body += f"--- {name} agent said ---\n{text}\n\n"
        body += ("You may agree or disagree with them, but your verdict must be based ONLY on your "
                "own perspective (see your system instructions) -- do not be swayed onto their "
                "perspective, only consider whether their facts change your view of YOUR perspective.\n\n")
    body += (
        "Give your reasoning in 2-3 sentences. Then end your reply with exactly two lines:\n"
        "FINAL: A  (or FINAL: B)\n"
        "CONFIDENCE: <a number from -1.0 to +1.0, negative favours A, positive favours B, "
        "magnitude = how strongly>"
    )
    return body


def parse_verdict(text):
    v = VERDICT_RE.search(text)
    c = CONF_RE.search(text)
    verdict = v.group(1).upper() if v else None
    try:
        conf = float(c.group(1)) if c else None
    except ValueError:
        conf = None
    return verdict, conf


def run_llm_debate(tickerA, tickerB, order=AGENTS, show_prior=True, labels=None):
    """Run one real LLM debate between two companies. Returns a dict with each agent's full text,
    parsed verdict/confidence, and the running tally (count of A-votes vs B-votes, weighted by
    confidence) after each speaker -- the LLM analogue of 'running evidence'."""
    labels = labels if labels is not None else load_labels(load_companies().index.tolist())
    rowA, rowB = labels.loc[tickerA], labels.loc[tickerB]

    transcript = []
    prior = []
    running = 0.0
    for agent in order:
        prompt = agent_prompt(agent, tickerA, tickerB, rowA, rowB, prior if show_prior else [])
        messages = [{"role": "system", "content": SYSTEM_PROMPTS[agent]}, {"role": "user", "content": prompt}]
        text = _call(messages)
        verdict, conf = parse_verdict(text)
        signed = conf if conf is not None else (0.5 if verdict == "B" else -0.5 if verdict == "A" else 0.0)
        running += signed
        transcript.append({"agent": agent, "text": text, "verdict": verdict, "confidence": conf,
                          "running_after": running})
        prior.append((agent, text))
    winner = tickerB if running > 0 else tickerA
    return {"a": tickerA, "b": tickerB, "order": order, "winner": winner, "final_evidence": running,
           "transcript": transcript}


def _safe(s):
    """Console-safe preview text -- Windows' terminal codepage can't print some Unicode
    punctuation an LLM happily generates (em-dashes, non-breaking hyphens, curly quotes)."""
    return s.encode("ascii", errors="replace").decode("ascii")


def main():
    set_seeds()
    pairs = [("NVDA", "VZ"), ("XOM", "HD"), ("MSFT", "NVDA")]
    labels = load_labels(load_companies().index.tolist())
    results = []
    for a, b in pairs:
        print(f"\n=== LLM debate: {a} vs {b} ===")
        res = run_llm_debate(a, b, labels=labels)
        for step in res["transcript"]:
            print(f"  {step['agent']:8s} verdict={step['verdict']} confidence={step['confidence']} "
                 f"running={step['running_after']:+.2f}")
            print(f"    \"{_safe(step['text'][:160].strip())}...\"")
        print(f"  -> LLM debate winner: {res['winner']}  (final evidence {res['final_evidence']:+.2f})")
        results.append(res)
        (OUT / f"llm_debate_{a}_vs_{b}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    pd.DataFrame([{"a": r["a"], "b": r["b"], "llm_winner": r["winner"], "final_evidence": r["final_evidence"]}
                 for r in results]).to_csv(OUT / "llm_debate_summary.csv", index=False)
    print("\nSaved transcripts + summary to outputs/llm_debate_*")


if __name__ == "__main__":
    main()
