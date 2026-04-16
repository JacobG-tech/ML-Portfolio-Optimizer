import pandas as pd
from pathlib import Path

PRICES_DIR = Path("data/raw/prices")
OUTPUT_DIR = Path("data/processed")
PANEL_FILE = OUTPUT_DIR / "panel.parquet"

# Zombie filter: drop rows where adjusted close is below this threshold
ZOMBIE_PRICE_THRESHOLD = 1.00

def load_ticker(path):
    """Load one parquet file, flatten multi-level columns, add ticker column."""
    ticker = path.stem
    df = pd.read_parquet(path)

    # Flatten multi-level columns: yfinance returns (metric, ticker) tuples
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names to lowercase with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Move date from index to a regular column
    df = df.reset_index()
    df.columns = df.columns.str.lower()

    # Add ticker column
    df["ticker"] = ticker

    return df

def apply_zombie_filter(df):
    """Drop a ticker's tail once prices permanently fall below threshold."""
    df = df.sort_values("date").reset_index(drop=True)

    above_threshold = df["adj_close"] >= ZOMBIE_PRICE_THRESHOLD
    if not above_threshold.any():
        return df.iloc[0:0]  # entire ticker is zombie

    # Find the last row where adj_close was above threshold
    last_healthy_idx = above_threshold[::-1].idxmax()
    return df.iloc[:last_healthy_idx + 1]

def consolidate():
    """Load all price files, filter, and combine into one long-format panel."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(PRICES_DIR.glob("*.parquet"))
    print(f"Consolidating {len(parquet_files)} tickers...\n")

    frames = []
    filter_log = []

    for path in parquet_files:
        df = load_ticker(path)
        rows_before = len(df)

        df = apply_zombie_filter(df)
        rows_after = len(df)
        dropped = rows_before - rows_after

        if dropped > 0:
            filter_log.append({
                "ticker": path.stem,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "dropped": dropped,
            })

        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel.to_parquet(PANEL_FILE)

    print(f"Panel saved to {PANEL_FILE}")
    print(f"  Total rows: {len(panel):,}")
    print(f"  Tickers: {panel['ticker'].nunique()}")
    print(f"  Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")

    if filter_log:
        print(f"\nZombie filter dropped rows from {len(filter_log)} tickers:")
        log_df = pd.DataFrame(filter_log)
        print(log_df.to_string(index=False))
    else:
        print("\nZombie filter didn't drop anything.")

    return panel


if __name__ == "__main__":
    consolidate()