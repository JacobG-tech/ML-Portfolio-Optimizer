import pandas as pd
import requests
from io import StringIO
from pathlib import Path

UNIVERSE_FILE = Path("data/raw/universe.csv")
OUTPUT_FILE = Path("data/raw/sectors.csv")

# Hardcoded sectors for failure tickers not in S&P 100.
# Assigned based on each company's primary business at time of failure.
FAILURE_SECTORS = {
    "AIG":  "Financials",
    "FNMA": "Financials",
    "BBBY": "Consumer Discretionary",
    "FRCB": "Financials",
    "HTZ":  "Industrials",
}


def scrape_sp100_sectors():
    """Scrape S&P 100 tickers and GICS sectors from Wikipedia.

    Uses the same table that universe.py reads, but keeps the Symbol and
    GICS Sector columns. Returns a DataFrame with columns [ticker, sector].
    """
    html = requests.get(
        "https://en.wikipedia.org/wiki/S%26P_100",
        headers={"User-Agent": "Mozilla/5.0"},
    ).text
    tables = pd.read_html(StringIO(html))
    df = tables[2]

    df = df[["Symbol", "Sector"]].rename(
        columns={"Symbol": "ticker", "Sector": "sector"}
    )
    return df

def normalize_sector(sector):
    """Convert GICS sector name to snake_case for use as column suffix.

    Example: "Consumer Discretionary" -> "consumer_discretionary"
    """
    return sector.lower().replace(" ", "_")


def build_sector_table():
    """Combine S&P 100 sectors (from Wikipedia) with failure ticker sectors
    (from hardcoded dict). Normalizes sector names consistently across both
    sources. Returns DataFrame with columns [ticker, sector].
    """
    sp100 = scrape_sp100_sectors()

    failures = pd.DataFrame(
        [{"ticker": t, "sector": s} for t, s in FAILURE_SECTORS.items()]
    )

    combined = pd.concat([sp100, failures], ignore_index=True)
    combined["sector"] = combined["sector"].apply(normalize_sector)

    combined = combined.drop_duplicates(subset="ticker", keep="first")

    return combined

# Canonical GICS sectors (11 total), normalized to match our column naming.
# If Wikipedia reformats, the assertion below will catch it loudly.
CANONICAL_GICS_SECTORS = {
    "communication_services",
    "consumer_discretionary",
    "consumer_staples",
    "energy",
    "financials",
    "health_care",
    "industrials",
    "information_technology",
    "materials",
    "real_estate",
    "utilities",
}


def reconcile_with_universe(sectors_df):
    """Verify every ticker in universe.csv has a sector assigned.

    Loudly fails if any ticker is missing, or if the sector set doesn't
    exactly match the canonical 11 GICS sectors.
    """
    universe = pd.read_csv(UNIVERSE_FILE)
    universe_tickers = set(universe["ticker"])
    mapped_tickers = set(sectors_df["ticker"])

    missing = universe_tickers - mapped_tickers
    if missing:
        raise ValueError(
            f"{len(missing)} universe tickers lack a sector mapping: "
            f"{sorted(missing)}"
        )

    actual_sectors = set(sectors_df["sector"])
    if actual_sectors != CANONICAL_GICS_SECTORS:
        unexpected = actual_sectors - CANONICAL_GICS_SECTORS
        missing_sectors = CANONICAL_GICS_SECTORS - actual_sectors
        raise ValueError(
            f"Sector set does not match canonical GICS.\n"
            f"  Unexpected: {unexpected}\n"
            f"  Missing from data: {missing_sectors}"
        )

    # Keep only tickers that are actually in our universe
    sectors_df = sectors_df[sectors_df["ticker"].isin(universe_tickers)].copy()
    return sectors_df


def save_sectors(sectors_df):
    """Write the sectors table to CSV and print a summary."""
    sectors_df = sectors_df.sort_values("ticker").reset_index(drop=True)
    sectors_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Sectors saved to {OUTPUT_FILE}")
    print(f"  Total tickers: {len(sectors_df)}")
    print(f"  Unique sectors: {sectors_df['sector'].nunique()}")
    print(f"\nSector distribution:")
    print(sectors_df["sector"].value_counts().to_string())


if __name__ == "__main__":
    sectors = build_sector_table()
    sectors = reconcile_with_universe(sectors)
    save_sectors(sectors)