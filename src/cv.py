"""Purged walk-forward cross-validation for time-series ML.

Design decisions (see 03_ROADMAP.md and 04_DECISIONS_AND_LESSONS.md):
- Expanding-window annual folds: train through year N, validate year N+1
- 21-day purge before each validation fold (matches target horizon)
- 2023-2025 is untouched holdout, never surfaces in any fold
- Yields 2-tuples of positional integer indices, sklearn-compatible
- Inner early-stopping split lives in a separate utility function
"""

import numpy as np
import pandas as pd


MIN_FOLD_TRAIN_ROWS = 10_000


class PurgedWalkForwardCV:
    """Expanding-window walk-forward CV with purging for financial time-series.

    Parameters
    ----------
    start_year : int
        First year used as a validation fold. Everything strictly before
        this year is the fold-1 training set.
    end_year : int
        Last year used as a validation fold (inclusive). Everything after
        this year is the holdout set and is never returned by split().
    purge_days : int
        Number of trading days of training data to drop immediately
        before each validation fold. Should equal the target horizon
        (21 for a 21-day forward return target).
    """

    def __init__(self, start_year=2015, end_year=2022, purge_days=21):
        if start_year > end_year:
            raise ValueError(
                f"start_year ({start_year}) must be <= end_year ({end_year})"
            )
        if purge_days < 0:
            raise ValueError(f"purge_days must be >= 0, got {purge_days}")

        self.start_year = start_year
        self.end_year = end_year
        self.purge_days = purge_days

    def get_n_splits(self):
        """Number of validation folds. Sklearn-style accessor."""
        return self.end_year - self.start_year + 1

    def split(self, df):
        """Generate (train_idx, val_idx) pairs for each fold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'date' column. Rows do not need to be sorted.

        Yields
        ------
        train_idx : np.ndarray
            Positional integer indices (0-based) of training rows.
            Use with df.iloc[train_idx], not df.loc[].
        val_idx : np.ndarray
            Positional integer indices of validation rows.
        """
        if "date" not in df.columns:
            raise ValueError("df must contain a 'date' column")

        dates = pd.to_datetime(df["date"]).values
        positional = np.arange(len(df))

        for val_year in range(self.start_year, self.end_year + 1):
            val_start = np.datetime64(f"{val_year}-01-01")
            val_end = np.datetime64(f"{val_year + 1}-01-01")

            # Validation: rows in [val_start, val_end)
            val_mask = (dates >= val_start) & (dates < val_end)
            val_idx = positional[val_mask]

            # Training: all rows strictly before val_start, then purged
            pre_val_mask = dates < val_start
            pre_val_positional = positional[pre_val_mask]
            pre_val_dates = dates[pre_val_mask]

            if self.purge_days > 0 and len(pre_val_dates) > 0:
                # Unique trading days in the pre-validation range, sorted ascending
                unique_pre_val_days = np.unique(pre_val_dates)
                # The last `purge_days` trading days get dropped from training
                purge_cutoff = unique_pre_val_days[-self.purge_days]
                train_mask_local = pre_val_dates < purge_cutoff
                train_idx = pre_val_positional[train_mask_local]
            else:
                train_idx = pre_val_positional

            if len(train_idx) < MIN_FOLD_TRAIN_ROWS:
                raise ValueError(
                    f"Fold for val_year={val_year} has only {len(train_idx)} "
                    f"training rows (< {MIN_FOLD_TRAIN_ROWS}). "
                    f"Check start_year and data coverage."
                )
            if len(val_idx) == 0:
                raise ValueError(
                    f"Fold for val_year={val_year} has zero validation rows. "
                    f"Check that your data covers this year."
                )

            yield train_idx, val_idx


def inner_early_stopping_split(df, train_idx, inner_val_frac=0.1):
    """Split a fold's training indices into inner-train and inner-val.

    Used by train.py for XGBoost early stopping. The inner-val is the
    most recent `inner_val_frac` of the training period, kept contiguous
    (by date, not random). The outer validation fold is NEVER touched —
    that's what the overall CV is measuring, and peeking at it for early
    stopping would leak.

    Parameters
    ----------
    df : pd.DataFrame
        The same DataFrame passed to PurgedWalkForwardCV.split().
    train_idx : np.ndarray
        Positional integer indices of this fold's training set,
        as yielded by .split().
    inner_val_frac : float
        Fraction of the training period to reserve as inner validation.
        Default 0.1 (the last 10% of the training date range).

    Returns
    -------
    inner_train_idx : np.ndarray
        Positional integer indices for training during this fold.
    inner_val_idx : np.ndarray
        Positional integer indices for early-stopping validation.
    """
    if not 0 < inner_val_frac < 1:
        raise ValueError(
            f"inner_val_frac must be in (0, 1), got {inner_val_frac}"
        )

    train_dates = pd.to_datetime(df["date"].iloc[train_idx]).values
    unique_days = np.unique(train_dates)

    n_inner_val_days = max(1, int(len(unique_days) * inner_val_frac))
    cutoff_date = unique_days[-n_inner_val_days]

    inner_val_mask = train_dates >= cutoff_date
    inner_val_idx = train_idx[inner_val_mask]
    inner_train_idx = train_idx[~inner_val_mask]

    return inner_train_idx, inner_val_idx


if __name__ == "__main__":
    from pathlib import Path

    TRAINING_FILE = Path("data/processed/features_training.parquet")

    print("Loading training data for smoke test...")
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  {len(df):,} rows, date range {df['date'].min().date()} to {df['date'].max().date()}\n")

    cv = PurgedWalkForwardCV(start_year=2015, end_year=2022, purge_days=21)
    print(f"Running {cv.get_n_splits()} folds:\n")

    for i, (train_idx, val_idx) in enumerate(cv.split(df), start=1):
        train_dates = pd.to_datetime(df["date"].iloc[train_idx])
        val_dates = pd.to_datetime(df["date"].iloc[val_idx])

        inner_train_idx, inner_val_idx = inner_early_stopping_split(
            df, train_idx, inner_val_frac=0.1
        )
        inner_val_dates = pd.to_datetime(df["date"].iloc[inner_val_idx])

        print(f"Fold {i} (val={val_dates.min().year}):")
        print(f"  Outer train: {len(train_idx):>7,} rows, "
              f"{train_dates.min().date()} to {train_dates.max().date()}")
        print(f"  Inner val:   {len(inner_val_idx):>7,} rows, "
              f"{inner_val_dates.min().date()} to {inner_val_dates.max().date()}")
        print(f"  Outer val:   {len(val_idx):>7,} rows, "
              f"{val_dates.min().date()} to {val_dates.max().date()}")