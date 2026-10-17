"""Phase 2: download each company's latest 10-K from SEC EDGAR and extract
Item 1 (Business) and Item 1A (Risk Factors) as plain text.

Rules followed: SEC User-Agent with contact, <=4 requests/second, every
download cached on disk, no invented text. If a section cannot be found we
try a documented fallback; if that fails the company is logged and dropped
for the text-dependent steps.
"""
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

import sec
from config import DATA, RAW_TEXT, set_seeds

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MIN_SECTION_CHARS = 3000      # shorter than this = probably a table-of-contents entry
MAX_FALLBACK_CHARS = 60000
MAX_SECTION_CHARS = 250000    # sanity cap: a longer 'section' means the end marker was missed

DASH = r"[\.\s\:\-–—]*"


def loose(word: str) -> str:
    """Regex for a word that tolerates the HTML having split it across lines (e.g. 'RIS' / 'K FACTORS')."""
    return r"\s*".join(re.escape(c) for c in word)


RX_ITEM1 = re.compile(rf"(?im)^items?\s*1(?:\s*(?:and|&)\s*2)?{DASH}{loose('business')}(?:\s*and\s*properties)?")
RX_ITEM1A = re.compile(rf"(?im)^item\s*1a{DASH}{loose('riskfactors')}")
RX_AFTER_1A = re.compile(
    rf"(?im)^item\s*(?:1b{DASH}{loose('unresolvedstaffcomments')}|1c{DASH}{loose('cybersecurity')}"
    rf"|2{DASH}{loose('properties')})")

# A few filers (GE) do not use 'Item 1A' lines at all. For those we name the real headings by hand.
# These are documented in DECISIONS.md; the text is still taken verbatim from the filing.
OVERRIDES = {
    "GE": {
        "item1": (r"(?m)^ABOUT GE AEROSPACE\.\s*$", r"(?m)^BUSINESS OVERVIEW AND ENVIRONMENT\.\s*$"),
        "item1a": (r"(?m)^RISK FACTORS\.\s*$", r"(?m)^LEGAL PROCEEDINGS\.\s*$"),
    },
}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["ix:header", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def heading_like(text, m):
    """True if this match looks like a real section heading, not a cross-reference such as
    'Item 1A. Risk Factors - Global Operations section'. A heading stands alone on its line and the
    next line starts a new sentence (capital letter / digit), not 'section', 'and', '.', ':' ..."""
    nl = text.find("\n", m.end())
    nl = len(text) if nl == -1 else nl
    if text[m.end():nl].strip(" .:") != "":
        return False
    nxt = text[nl + 1: nl + 2]
    return not (nxt and (nxt.islower() or nxt in ".,;:)-\u2013\u2014\ufffd"))


def best_span(text, start_rx, end_rx):
    """For every end marker take the NEAREST heading before it (so a cross-reference cannot stretch
    the section); the real section is the longest such span, because table-of-contents entries are tiny."""
    starts = [m.start() for m in start_rx.finditer(text) if heading_like(text, m)]
    ends = [m.start() for m in end_rx.finditer(text) if heading_like(text, m)]
    best = ""
    for e in ends:
        prev = [s for s in starts if s < e]
        if not prev:
            continue
        seg = text[max(prev):e]
        if len(seg) > len(best):
            best = seg
    return best


def override_span(text, start_pat, end_pat):
    best = ""
    for m in re.finditer(start_pat, text):
        e = re.compile(end_pat).search(text, m.end())
        seg = text[m.start(): e.start() if e else len(text)]
        if len(seg) > len(best):
            best = seg
    return best


def extract_sections(text, ticker=""):
    """Return dict(item1, item1a, m1, m1a) where m1/m1a say HOW each section was found."""
    out = {"item1": "", "item1a": "", "m1": "none", "m1a": "none"}

    for key, mkey in (("item1", "m1"), ("item1a", "m1a")):
        if ticker in OVERRIDES and key in OVERRIDES[ticker]:
            seg = override_span(text, *OVERRIDES[ticker][key])
            if len(seg) >= MIN_SECTION_CHARS:
                out[key], out[mkey] = seg[:MAX_SECTION_CHARS], "manual_heading_override"

    if not out["item1"]:
        s1 = best_span(text, RX_ITEM1, RX_ITEM1A)
        if len(s1) >= MIN_SECTION_CHARS:
            out["item1"] = s1[:MAX_SECTION_CHARS]
            out["m1"] = "standard" if len(s1) <= MAX_SECTION_CHARS else "standard_capped"
        else:  # fallback: 'Business overview' style heading
            seg = ""
            for m in re.finditer(r"(?im)^(business overview|overview)\s*$", text):
                cand = text[m.start(): m.start() + MAX_FALLBACK_CHARS]
                if len(cand) > len(seg):
                    seg = cand
            if len(seg) >= MIN_SECTION_CHARS:
                out["item1"], out["m1"] = seg, "fallback_overview_heading"

    if not out["item1a"]:
        s1a = best_span(text, RX_ITEM1A, RX_AFTER_1A)
        if len(s1a) >= MIN_SECTION_CHARS:
            out["item1a"] = s1a[:MAX_SECTION_CHARS]
            out["m1a"] = "standard" if len(s1a) <= MAX_SECTION_CHARS else "standard_capped"
        else:  # fallback: bare 'Risk Factors' heading
            end = re.compile(r"(?im)^(unresolved staff comments|cybersecurity|properties|legal proceedings)\s*$")
            seg = ""
            for m in re.finditer(r"(?im)^risk\s*factors\s*$", text):
                e = end.search(text, m.end())
                cand = text[m.start(): min(e.start() if e else len(text), m.start() + MAX_FALLBACK_CHARS)]
                if len(cand) > len(seg):
                    seg = cand
            if len(seg) >= MIN_SECTION_CHARS:
                out["item1a"], out["m1a"] = seg, "fallback_risk_factors_heading"
    return out


def latest_10k(cik: str):
    sub = sec.get(f"https://data.sec.gov/submissions/CIK{cik}.json", as_json=True)
    rec = sub["filings"]["recent"]
    for i, form in enumerate(rec["form"]):
        if form == "10-K":
            acc = rec["accessionNumber"][i]
            doc = rec["primaryDocument"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
            return {"accession": acc, "filing_date": rec["filingDate"][i],
                    "report_date": rec["reportDate"][i], "url": url,
                    "company_name_sec": sub.get("name", "")}
    return None


def main():
    set_seeds()
    comps = pd.read_csv(DATA / "companies.csv", dtype={"cik": str})
    rows = []
    for _, c in comps.iterrows():
        tkr, cik = str(c["ticker"]), str(c["cik"])
        meta = {"ticker": tkr, "cik": cik, "status": "ok", "note": ""}
        for stale in RAW_TEXT.glob(f"{tkr}_item*.txt"):
            stale.unlink()                      # never keep text from an earlier extraction attempt
        try:
            info = latest_10k(cik)
            if not info:
                meta.update(status="dropped", note="no 10-K found in SEC submissions")
                rows.append(meta); print(f"{tkr}: NO 10-K"); continue
            meta.update(info)
            text = html_to_text(sec.get(info["url"]))
            s = extract_sections(text, tkr)
            meta.update(item1_chars=len(s["item1"]), item1a_chars=len(s["item1a"]),
                        item1_method=s["m1"], item1a_method=s["m1a"], doc_chars=len(text))
            if s["item1"]:
                (RAW_TEXT / f"{tkr}_item1.txt").write_text(s["item1"], encoding="utf-8")
            if s["item1a"]:
                (RAW_TEXT / f"{tkr}_item1a.txt").write_text(s["item1a"], encoding="utf-8")
            if not s["item1"] and not s["item1a"]:
                meta.update(status="dropped", note="could not extract Item 1 or Item 1A")
            elif not s["item1"] or not s["item1a"]:
                meta.update(status="partial", note="only one section extracted")
            print(f"{tkr}: {info['filing_date']} item1={len(s['item1']):>7} ({s['m1']}) "
                  f"item1a={len(s['item1a']):>7} ({s['m1a']}) -> {meta['status']}")
        except Exception as e:  # keep going; log the failure
            meta.update(status="dropped", note=f"error: {e!r}"[:200])
            print(f"{tkr}: ERROR {e!r}")
        rows.append(meta)
    pd.DataFrame(rows).to_csv(RAW_TEXT / "text_metadata.csv", index=False)
    print("\nSaved metadata ->", RAW_TEXT / "text_metadata.csv")


if __name__ == "__main__":
    main()
