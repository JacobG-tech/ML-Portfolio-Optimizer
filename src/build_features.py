import pandas as pd
from pathlib import Path

from features import (
    add_return_features,
    add_trend_features,
    add_volatility_features,
    add_market_features,
    add_rsi_14,
    add_bb_position_20,
    add_macd_hist,
    add_atr_14_pct,
    add_volume_ratio_20d,
)
from targets import (
    add_target_ret_21d,
    add_target_dd5_21d,
)

PANEL_FILE = Path("data/processed/panel.parquet")
SPY_FILE = Path("data/raw/spy.parquet")
SECTORS_FILE = Path("data/raw/sectors.csv")
OUTPUT_DIR = Path("data/processed")
TRAINING_FILE = OUTPUT_DIR / "features_training.parquet"
PREDICTION_FILE = OUTPUT_DIR / "features_prediction.parquet"

# Canonical GICS sectors. Must match the set produced by add_sector.py.
# Kept here (not imported) to keep build_features.py self-contained.
CANONICAL_GICS_SECTORS = [
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
]


def attach_spy_return(panel):
    """Add spy_ret_1d column to panel by merging SPY daily returns on date."""
    spy = pd.read_parquet(SPY_FILE)
    spy["spy_ret_1d"] = spy["adj_close"].pct_change()
    spy_ret = spy[["date", "spy_ret_1d"]]
    panel = panel.merge(spy_ret, on="date", how="left")
    return panel


def attach_sector(panel):
    """Add sector data to the panel.

    Adds a raw `sector` string column (for LightGBM's native categorical
    handling) and 11 one-hot columns named sector_<name> (for XGBoost,
    which requires numeric features).

    One-hots are generated for ALL 11 canonical GICS sectors regardless
    of which are present in the data, so the column set is fixed across
    runs. This prevents surprises if the universe ever drops the last
    ticker in a sparse sector (e.g., Materials).
    """
    sectors = pd.read_csv(SECTORS_FILE)
    panel = panel.merge(sectors, on="ticker", how="left")

    if panel["sector"].isna().any():
        missing = panel[panel["sector"].isna()]["ticker"].unique()
        raise ValueError(
            f"Sector merge left {len(missing)} tickers without a sector: "
            f"{sorted(missing)}"
        )

    for sector_name in CANONICAL_GICS_SECTORS:
        col_name = f"sector_{sector_name}"
        panel[col_name] = (panel["sector"] == sector_name).astype(int)

    return panel


def build_features():
    """Run the full feature + target pipeline and write output files."""
    print("Loading panel...")
    panel = pd.read_parquet(PANEL_FILE)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers")

    print("Attaching SPY returns...")
    panel = attach_spy_return(panel)

    print("Attaching sector data...")
    panel = attach_sector(panel)

    print("Computing features...")
    panel = add_return_features(panel)
    panel = add_trend_features(panel)
    panel = add_volatility_features(panel)
    panel = add_market_features(panel)      # needs ret_21d and spy_ret_1d
    panel = add_rsi_14(panel)
    panel = add_bb_position_20(panel)
    panel = add_macd_hist(panel)
    panel = add_atr_14_pct(panel)
    panel = add_volume_ratio_20d(panel)

    print("Computing targets...")
    panel = add_target_ret_21d(panel)
    panel = add_target_dd5_21d(panel)

    return panel


def split_and_save(panel):
    """Split the panel into training-ready and prediction-ready outputs.

    Training: rows with all features AND both targets present. Used to fit
    the ML models.

    Prediction: rows with all features but no targets (the last 21 rows
    per ticker, where we can't know forward returns yet). Used at inference
    time to generate signals for the most recent dates.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    numeric_feature_cols = [
        "ret_21d", "ret_63d", "ret_252d", "price_to_sma50",
        "vol_20d", "range_pct_20d", "max_dd_90d",
        "beta_60d", "excess_ret_21d",
        "rsi_14", "bb_position_20", "macd_hist", "atr_14_pct",
        "volume_ratio_20d",
    ]
    sector_onehot_cols = [f"sector_{s}" for s in CANONICAL_GICS_SECTORS]
    feature_cols = numeric_feature_cols + sector_onehot_cols + ["sector"]

    target_cols = ["target_ret_21d", "target_dd5_21d"]

    # Rows must have every feature present to be useful
    feature_complete = panel[feature_cols].notna().all(axis=1)

    # Training set: features complete AND targets present
    target_complete = panel[target_cols].notna().all(axis=1)
    training = panel[feature_complete & target_complete].copy()

    # Prediction set: features complete but at least one target missing
    prediction = panel[feature_complete & ~target_complete].copy()

    training.to_parquet(TRAINING_FILE)
    prediction.to_parquet(PREDICTION_FILE)

    print(f"\nTraining set: {TRAINING_FILE}")
    print(f"  Rows: {len(training):,}")
    print(f"  Date range: {training['date'].min().date()} to {training['date'].max().date()}")
    print(f"  target_dd5_21d positive rate: {training['target_dd5_21d'].mean():.1%}")
    print(f"  Feature columns: {len(numeric_feature_cols)} numeric + {len(sector_onehot_cols)} one-hot + 1 raw sector string = {len(feature_cols)}")

    print(f"\nPrediction set: {PREDICTION_FILE}")
    print(f"  Rows: {len(prediction):,}")
    print(f"  Date range: {prediction['date'].min().date()} to {prediction['date'].max().date()}")


if __name__ == "__main__":
    panel = build_features()
    split_and_save(panel)