import pandas as pd
import requests
from io import StringIO


def get_sp100():
    """Scrape current S&P 100 tickers from Wikipedia."""
    html = requests.get(
        "https://en.wikipedia.org/wiki/S%26P_100",
        headers={"User-Agent": "Mozilla/5.0"}
    ).text
    tables = pd.read_html(StringIO(html))
    df = tables[2]
    tickers = df["Symbol"].tolist()
    return tickers


def get_failed_companies():
    """Historical failures and dramatic declines to combat survivorship bias."""
    failures = pd.DataFrame([
        {"ticker": "AIG", "name": "AIG", "reason": "Bailout/near-collapse 2008"},
        {"ticker": "FNMA", "name": "Fannie Mae", "reason": "Conservatorship 2008"},
        {"ticker": "BLIAQ", "name": "Blockbuster", "reason": "Bankruptcy 2010"},
        {"ticker": "BBBY", "name": "Bed Bath & Beyond", "reason": "Bankruptcy 2023"},
        {"ticker": "FRCB", "name": "First Republic Bank", "reason": "FDIC seizure 2023"},
        {"ticker": "HTZ", "name": "Hertz", "reason": "Bankruptcy 2020, relisted 2021"},
        {"ticker": "GE", "name": "General Electric", "reason": "Blue chip decline"},
        {"ticker": "T", "name": "AT&T", "reason": "Slow value destruction"},
        {"ticker": "INTC", "name": "Intel", "reason": "Recent dramatic decline"},
    ])
    return failures


def build_universe():
    """Combine S&P 100 and failed companies into one master universe."""
    sp100 = get_sp100()
    sp100_df = pd.DataFrame({
        "ticker": sp100,
        "status": "active",
        "reason": ""
    })
    failures = get_failed_companies()
    failures["status"] = "failure"
    universe = pd.concat([sp100_df, failures], ignore_index=True)
    # Remove any duplicates (AIG is in both S&P 100 and failures)
    universe = universe.drop_duplicates(subset="ticker", keep="first")
    universe.to_csv("data/raw/universe.csv", index=False)
    print(f"Universe saved: {len(universe)} tickers")
    print(f"  Active: {len(universe[universe['status'] == 'active'])}")
    print(f"  Failures: {len(universe[universe['status'] == 'failure'])}")
    return universe


if __name__ == "__main__":
    build_universe()

    