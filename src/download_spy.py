import pandas as pd
import yfinance as yf
from pathlib import Path

OUTPUT_FILE = Path("data/raw/spy.parquet")
START_DATE = "2011-04-18"

def download_spy():
    """Download SPY daily data from inception of our panel window to today."""
    print(f"Downloading SPY from {START_DATE}...")

    df = yf.download(
        "SPY",
        start=START_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError("SPY download returned empty. Check network / yfinance.")

    return df

def clean_spy(df):
    """Normalize SPY DataFrame to match our panel's column conventions."""
    # Flatten multi-level columns if yfinance returned them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    # Normalize column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Move date from index to a regular column
    df = df.reset_index()
    df.columns = df.columns.str.lower()

    return df

def save_spy(df):
    """Save SPY DataFrame to parquet."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE)
    print(f"SPY saved to {OUTPUT_FILE}")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")


if __name__ == "__main__":
    df = download_spy()
    df = clean_spy(df)
    save_spy(df)