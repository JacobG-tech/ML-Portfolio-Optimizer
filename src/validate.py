import pandas as pd
from pathlib import Path

PRICES_DIR = Path("data/raw/prices")
OUTPUT_DIR = Path("data/processed")
REPORT_FILE = OUTPUT_DIR / "validation_report.csv"

MIN_HISTORY_DAYS = 252
PRICE_GAP_THRESHOLD = 0.5
FLAT_TAIL_THRESHOLD = 20

def validate_ticker(ticker, data):
    """Run quality checks on one ticker's price data. Returns a dict of metrics."""
    metrics = {
        "ticker": ticker,
        "total_days": len(data),
        "start_date": data.index[0].date(),
        "end_date": data.index[-1].date(),
    }

    # Flat tail: trailing days with zero volume (stock effectively dead)
    volume = data["Volume"].squeeze()
    zero_volume_mask = (volume == 0)
    trailing_zeros = 0
    for v in reversed(zero_volume_mask.values):
        if v:
            trailing_zeros += 1
        else:
            break
    metrics["trailing_zero_volume_days"] = trailing_zeros

    # Price gaps: single-day moves larger than threshold
    adj_close = data["Adj Close"].squeeze()
    daily_returns = adj_close.pct_change()
    large_gaps = (daily_returns.abs() > PRICE_GAP_THRESHOLD).sum()
    metrics["large_price_gaps"] = int(large_gaps)

    # Missing dates: expected trading days that aren't in the data
    expected = pd.bdate_range(start=data.index[0], end=data.index[-1])
    missing = len(expected) - len(data.index)
    metrics["missing_days"] = int(missing)

    # Duplicate dates: same date appearing twice (shouldn't happen but does)
    metrics["duplicate_dates"] = int(data.index.duplicated().sum())

    # Flag summary: boolean indicators for quick filtering
    metrics["flag_too_short"] = len(data) < MIN_HISTORY_DAYS
    metrics["flag_flat_tail"] = trailing_zeros >= FLAT_TAIL_THRESHOLD
    metrics["flag_has_gaps"] = large_gaps > 0

    return metrics

def validate_all():
    """Run validation on all tickers and save a report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(PRICES_DIR.glob("*.parquet"))
    print(f"Validating {len(parquet_files)} tickers...\n")

    all_metrics = []

    for path in parquet_files:
        ticker = path.stem
        data = pd.read_parquet(path)
        metrics = validate_ticker(ticker, data)
        all_metrics.append(metrics)

    report = pd.DataFrame(all_metrics)
    report.to_csv(REPORT_FILE, index=False)

    print(f"Report saved to {REPORT_FILE}")
    print(f"\nSummary:")
    print(f"  Total tickers: {len(report)}")
    print(f"  Too short (<{MIN_HISTORY_DAYS} days): {report['flag_too_short'].sum()}")
    print(f"  Flat tail (>={FLAT_TAIL_THRESHOLD} zero-vol days): {report['flag_flat_tail'].sum()}")
    print(f"  Has price gaps: {report['flag_has_gaps'].sum()}")

    flagged = report[
        report["flag_too_short"] |
        report["flag_flat_tail"] |
        report["flag_has_gaps"]
    ]
    if len(flagged) > 0:
        print(f"\n{len(flagged)} tickers flagged for review:")
        print(flagged[["ticker", "total_days", "trailing_zero_volume_days", "large_price_gaps"]].to_string(index=False))

    return report

if __name__ == "__main__":
    validate_all()