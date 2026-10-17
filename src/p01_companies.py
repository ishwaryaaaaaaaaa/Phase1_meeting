"""Phase 1: choose the companies and write companies.csv (ticker, name, CIK, sector).

CIKs come from SEC's own ticker->CIK table (not typed from memory).
"""
import pandas as pd

import sec
from config import DATA, set_seeds

# 30 large US-listed companies from contrasting sectors, chosen so that
# Finance, ESG and Risk perspectives genuinely disagree (see DECISIONS.md).
COMPANIES = [
    # ticker, short name, sector (our own simple grouping, used for sanity checks)
    ("AAPL", "Apple", "Technology"),
    ("MSFT", "Microsoft", "Technology"),
    ("NVDA", "NVIDIA", "Technology"),
    ("XOM", "Exxon Mobil", "Oil & Gas"),
    ("CVX", "Chevron", "Oil & Gas"),
    ("COP", "ConocoPhillips", "Oil & Gas"),
    ("JPM", "JPMorgan Chase", "Banks"),
    ("BAC", "Bank of America", "Banks"),
    ("GS", "Goldman Sachs", "Banks"),
    ("NEE", "NextEra Energy", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Company", "Utilities"),
    ("WMT", "Walmart", "Retail"),
    ("COST", "Costco", "Retail"),
    ("HD", "Home Depot", "Retail"),
    ("JNJ", "Johnson & Johnson", "Pharma & Healthcare"),
    ("PFE", "Pfizer", "Pharma & Healthcare"),
    ("UNH", "UnitedHealth", "Pharma & Healthcare"),
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("KO", "Coca-Cola", "Consumer Staples"),
    ("PEP", "PepsiCo", "Consumer Staples"),
    ("CAT", "Caterpillar", "Industrials"),
    ("GE", "GE Aerospace", "Industrials"),
    ("BA", "Boeing", "Industrials"),
    ("UPS", "United Parcel Service", "Industrials"),
    ("TSLA", "Tesla", "Automotive"),
    ("F", "Ford", "Automotive"),
    ("GM", "General Motors", "Automotive"),
    ("T", "AT&T", "Telecom"),
    ("VZ", "Verizon", "Telecom"),
]


# Exxon became a holding company in 2026: SEC's ticker table now points XOM at the NEW registrant
# (ExxonMobil Holdings Corp, CIK 2115436, which has no 10-K yet). Its latest 10-K (filed 2026-02-18)
# sits under the old Exxon Mobil Corp CIK, so we use that one.
CIK_OVERRIDE = {"XOM": 34088}


def main():
    set_seeds()
    table = sec.get("https://www.sec.gov/files/company_tickers.json", as_json=True)
    t2cik = {v["ticker"].upper(): int(v["cik_str"]) for v in table.values()}
    rows = []
    for tkr, name, sector in COMPANIES:
        if tkr not in t2cik:
            print(f"!! {tkr} not in SEC ticker table - dropped")
            continue
        cik = CIK_OVERRIDE.get(tkr, t2cik[tkr])
        rows.append({"ticker": tkr, "name": name, "cik": f"{cik:010d}", "sector": sector})
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "companies.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {len(df)} companies -> {DATA / 'companies.csv'}")


if __name__ == "__main__":
    main()
