"""Phase 3: perspective labels (what each company 'is worth' on Finance / ESG / Risk).

Every number is real and comes from a public source; missing stays missing.
  Finance : 3-year revenue CAGR + profitability margin (z-scored, averaged). Source: Yahoo Finance
            annual statements via yfinance (originally from company filings). Higher = better.
  ESG     : Sustainalytics 'Total ESG Risk score' from the Kaggle 'S&P 500 ESG Risk Ratings'
            dataset (a snapshot of Yahoo Finance / Sustainalytics). LOWER risk = better, so we
            flip the sign. Not estimated by us.
  Risk    : 3 years of daily prices -> annualised volatility and maximum drawdown
            (z-scored, averaged, sign flipped so higher = SAFER).
"""
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA, LABELS, set_seeds

warnings.filterwarnings("ignore")


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def finance_raw(tkr: str):
    """Return (cagr3, margin, margin_kind, fiscal_dates) from annual statements, or None."""
    inc = yf.Ticker(tkr).income_stmt
    if inc is None or inc.empty or "Total Revenue" not in inc.index:
        return None
    rev = inc.loc["Total Revenue"].dropna().sort_index()      # oldest -> newest
    if len(rev) < 4:
        return None
    r_new, r_old = rev.iloc[-1], rev.iloc[-4]
    if r_old <= 0 or r_new <= 0:
        return None
    cagr = (r_new / r_old) ** (1 / 3) - 1
    if "Operating Income" in inc.index and not np.isnan(inc.loc["Operating Income"].get(rev.index[-1], np.nan)):
        margin = inc.loc["Operating Income"][rev.index[-1]] / r_new
        kind = "operating_margin"
    elif "Pretax Income" in inc.index and not np.isnan(inc.loc["Pretax Income"].get(rev.index[-1], np.nan)):
        margin = inc.loc["Pretax Income"][rev.index[-1]] / r_new   # banks have no operating income line
        kind = "pretax_margin"
    else:
        return None
    return cagr, margin, kind, f"{rev.index[-4].date()}..{rev.index[-1].date()}"


def risk_raw(tkr: str):
    h = yf.Ticker(tkr).history(period="3y", interval="1d", auto_adjust=True)
    if h is None or len(h) < 400:
        return None
    px = h["Close"].dropna()
    ret = px.pct_change().dropna()
    vol = ret.std() * np.sqrt(252)
    maxdd = (px / px.cummax() - 1).min()                         # negative number
    return vol, maxdd, f"{px.index[0].date()}..{px.index[-1].date()}"


def main():
    set_seeds()
    comps = pd.read_csv(DATA / "companies.csv", dtype={"cik": str})
    esg = pd.read_csv(LABELS / "raw" / "SP 500 ESG Risk Ratings.csv").set_index("Symbol")
    rows, log = [], []
    for _, c in comps.iterrows():
        t = c["ticker"]
        row = {"ticker": t}
        f = finance_raw(t)
        if f is None:
            log.append(f"{t}: finance data unavailable")
        else:
            row.update(rev_cagr_3y=f[0], profit_margin=f[1], margin_kind=f[2], finance_window=f[3])
        r = risk_raw(t)
        if r is None:
            log.append(f"{t}: price history unavailable")
        else:
            row.update(ann_vol=r[0], max_drawdown=r[1], price_window=r[2])
        v = esg["Total ESG Risk score"].get(t, np.nan)
        if pd.isna(v):
            log.append(f"{t}: no ESG score in dataset")
        else:
            row.update(esg_risk_score=float(v), esg_controversy=esg["Controversy Level"].get(t))
        rows.append(row)
        print(t, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items() if k != "ticker"})
    df = pd.DataFrame(rows).set_index("ticker")

    # ---- composites (z-scored over the companies that have the data) ----
    # Winsorise each component z-score at +-2.5 so one extreme company (e.g. NVIDIA's ~100%
    # 3-year revenue CAGR) cannot dominate a whole perspective (see DECISIONS.md).
    def zw(s):
        return zscore(s).clip(-2.5, 2.5)

    df["z_cagr"] = zw(df["rev_cagr_3y"])
    df["z_margin"] = zw(df["profit_margin"])
    df["finance_label"] = zscore(df[["z_cagr", "z_margin"]].mean(axis=1, skipna=False))
    df["esg_label"] = zscore(-df["esg_risk_score"])                      # lower risk score = better
    df["z_vol_safe"] = zw(-df["ann_vol"])
    df["z_dd_safe"] = zw(df["max_drawdown"])                             # less negative = safer
    df["risk_label"] = zscore(df[["z_vol_safe", "z_dd_safe"]].mean(axis=1, skipna=False))  # higher = safer

    df["finance_source"] = "Yahoo Finance annual statements via yfinance"
    df["esg_source"] = ("Sustainalytics Total ESG Risk score (lower=better), Kaggle 'S&P 500 ESG Risk "
                        "Ratings' (pritish509); file dated 2024-06-07, scrape date not stated")
    df["risk_source"] = "Yahoo Finance daily prices via yfinance (3y)"
    df["asof_date_pulled"] = dt.date.today().isoformat()
    df.to_csv(DATA / "labels.csv")
    (DATA / "labels_log.txt").write_text("\n".join(log) or "no missing data", encoding="utf-8")
    print("\nMissing/failed:", log or "none")
    print(df[["finance_label", "esg_label", "risk_label"]].round(2).to_string())
    print("\nCorrelation between the three label columns:")
    print(df[["finance_label", "esg_label", "risk_label"]].corr().round(2))


if __name__ == "__main__":
    main()
