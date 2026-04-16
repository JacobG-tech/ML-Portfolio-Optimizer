import pandas as pd
import yfinance as yf
import time
from pathlib import Path

YEARS_OF_HISTORY = 15
OUTPUT_DIR = Path("data/raw/prices")
UNIVERSE_FILE = Path("data/raw/universe.csv")
RETRY_DELAY_SECONDS = 3

def download_ticker(ticker):
    """Download price history for one ticker. Returns DataFrame or None if failed."""
    # yfinance needs hyphens instead of periods for multi-class shares
    yf_ticker = ticker.replace(".", "-")

    try:
        data = yf.download(
            yf_ticker,
            period=f"{YEARS_OF_HISTORY}y",
            auto_adjust=False,
            progress=False,
        )
        if len(data) == 0:
            return None
        return data
    except Exception as e:
        print(f"    Error downloading {ticker}: {e}")
        return None
    
def download_universe():
    """Download price history for every ticker in the universe CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(UNIVERSE_FILE)
    tickers = universe["ticker"].tolist()

    print(f"Downloading {len(tickers)} tickers, {YEARS_OF_HISTORY} years of history...")
    print(f"Output: {OUTPUT_DIR}\n")

    successes = []
    failures = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ")

        data = download_ticker(ticker)

        if data is None:
            print("retrying...", end=" ")
            time.sleep(RETRY_DELAY_SECONDS)
            data = download_ticker(ticker)

        if data is not None:
            output_path = OUTPUT_DIR / f"{ticker.replace('.', '-')}.parquet"
            data.to_parquet(output_path)
            successes.append(ticker)
            print(f"OK ({len(data)} days)")
        else:
            failures.append(ticker)
            print("FAILED")

    print(f"\nDone. {len(successes)} successful, {len(failures)} failed.")
    if failures:
        print(f"Failed tickers: {failures}")

if __name__ == "__main__":
    download_universe()