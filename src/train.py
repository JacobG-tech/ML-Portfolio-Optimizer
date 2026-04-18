"""Stage 3 training orchestrator: purged walk-forward CV with XGBoost.

For each of 8 annual validation folds:
1. Split fold's training set into inner-train + inner-val (last 10% by date)
2. Train XGBRegressor for forward 21-day return (MSE loss)
3. Train XGBClassifier for >5% drawdown in next 21 days (log loss)
4. Use inner-val for XGBoost early stopping
5. Predict on the outer validation fold
6. Stash predictions keyed by (ticker, date, fold)

Fold 1 runs ~30 Optuna TPE trials per model to pick hyperparameters.
Folds 2-8 reuse those. Predictions are saved for evaluate.py to consume.

Targets:
- Regression: forward 21-day return, winsorized per-date at 1st/99th pct
- Classification: binary drawdown flag (raw, not winsorized)

Key design decisions documented in 03_ROADMAP.md and 04_DECISIONS_AND_LESSONS.md.
"""

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cv import PurgedWalkForwardCV, inner_early_stopping_split


# Paths
TRAINING_FILE = Path("data/processed/features_training.parquet")
PREDICTIONS_DIR = Path("data/predictions")
PREDICTIONS_FILE = PREDICTIONS_DIR / "cv_predictions.parquet"
MODELS_DIR = Path("results/models")

# Reproducibility
RANDOM_SEED = 42

# Tuning config
N_OPTUNA_TRIALS = 30
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_MAX = 500

# Target winsorization percentiles
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

# Feature columns (25 total). Raw sector string is excluded here — that's
# for LightGBM later. XGBoost uses the one-hot encoded sector columns.
NUMERIC_FEATURE_COLS = [
    "ret_21d", "ret_63d", "ret_252d", "price_to_sma50",
    "vol_20d", "range_pct_20d", "max_dd_90d",
    "beta_60d", "excess_ret_21d",
    "rsi_14", "bb_position_20", "macd_hist", "atr_14_pct",
    "volume_ratio_20d",
]
SECTOR_ONEHOT_COLS = [
    "sector_communication_services", "sector_consumer_discretionary",
    "sector_consumer_staples", "sector_energy", "sector_financials",
    "sector_health_care", "sector_industrials", "sector_information_technology",
    "sector_materials", "sector_real_estate", "sector_utilities",
]
FEATURE_COLS = NUMERIC_FEATURE_COLS + SECTOR_ONEHOT_COLS

TARGET_RETURN = "target_ret_21d"
TARGET_DRAWDOWN = "target_dd5_21d"

# Silence Optuna's verbose trial logging. We'll print our own summary.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def winsorize_per_date(df, target_col, lower_q=WINSOR_LOWER_Q, upper_q=WINSOR_UPPER_Q):
    """Winsorize a target column at per-date percentiles.

    For each date, computes the 1st and 99th percentile of the target
    across all tickers on that date, then clips values outside those
    bounds to the bounds. This tames the meme-stock outliers (+392% in
    21 days, etc.) without losing return magnitude the optimizer needs.

    Per-date (not global) because return distributions shift over time —
    a 10% return is unusual in a calm month but ordinary during COVID.

    Returns a new Series (original column unchanged).
    """
    def clip_group(group):
        lower = group.quantile(lower_q)
        upper = group.quantile(upper_q)
        return group.clip(lower=lower, upper=upper)

    return df.groupby("date")[target_col].transform(clip_group)


def load_and_prep():
    """Load training parquet and prepare targets.

    Returns the DataFrame with an added 'target_ret_21d_winsor' column
    (winsorized regression target). Original columns are preserved so
    evaluation can compare against raw actuals.
    """
    print(f"Loading {TRAINING_FILE}...")
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  {len(df):,} rows, date range {df['date'].min().date()} to {df['date'].max().date()}")

    print(f"Winsorizing {TARGET_RETURN} at per-date [{WINSOR_LOWER_Q:.0%}, {WINSOR_UPPER_Q:.0%}]...")
    df["target_ret_21d_winsor"] = winsorize_per_date(df, TARGET_RETURN)

    raw_std = df[TARGET_RETURN].std()
    winsor_std = df["target_ret_21d_winsor"].std()
    raw_max = df[TARGET_RETURN].max()
    winsor_max = df["target_ret_21d_winsor"].max()
    print(f"  Raw target:      std={raw_std:.3f}, max={raw_max:.3f}")
    print(f"  Winsorized:      std={winsor_std:.3f}, max={winsor_max:.3f}")

    return df

def _suggest_xgb_params(trial, task):
    """Sample XGBoost hyperparameters from the Optuna trial.

    `task` is either 'regression' or 'classification'. Search space is
    essentially identical — XGBoost's core hyperparameters aren't
    task-specific. We branch only on the objective function.
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
    }

    if task == "regression":
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"
    elif task == "classification":
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        raise ValueError(f"Unknown task: {task}")

    return params


def _spearman_ic(y_true, y_pred, dates):
    """Cross-sectional Spearman IC averaged across dates.

    For each date, compute Spearman rank correlation between predictions
    and actuals across that day's tickers. Return the mean of those
    per-date correlations.

    Degenerate dates (fewer than 2 names, or constant predictions/actuals
    producing NaN from scipy) contribute 0.0, not NaN. This reflects the
    fact that a constant-prediction day has zero rank information —
    correctly counted as "no signal" rather than silently dropped, which
    would bias the average upward.
    """
    def daily_corr(group):
        if len(group) < 2:
            return 0.0
        corr = spearmanr(group["y_true"], group["y_pred"]).correlation
        return 0.0 if np.isnan(corr) else corr

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "date": dates})
    daily_ic = df.groupby("date").apply(daily_corr, include_groups=False)
    return daily_ic.mean()


def make_objective(df, train_idx, task):
    """Factory: returns an Optuna objective function for the given task.

    Parameters
    ----------
    df : pd.DataFrame
        Full training DataFrame (with winsorized target).
    train_idx : np.ndarray
        Fold 1's outer training indices from PurgedWalkForwardCV.
    task : str
        'regression' or 'classification'.

    Returns
    -------
    objective : callable
        Function that takes an optuna Trial and returns a float score.
        Higher is better (IC for regression, AUC for classification).
    """
    # Split fold 1's training into inner-train and inner-val ONCE here,
    # so every Optuna trial evaluates on the same inner-val.
    inner_train_idx, inner_val_idx = inner_early_stopping_split(
        df, train_idx, inner_val_frac=0.1
    )

    X_inner_train = df.iloc[inner_train_idx][FEATURE_COLS]
    X_inner_val = df.iloc[inner_val_idx][FEATURE_COLS]
    inner_val_dates = df.iloc[inner_val_idx]["date"].values

    if task == "regression":
        y_inner_train = df.iloc[inner_train_idx]["target_ret_21d_winsor"]
        y_inner_val = df.iloc[inner_val_idx]["target_ret_21d_winsor"]
    else:
        y_inner_train = df.iloc[inner_train_idx][TARGET_DRAWDOWN]
        y_inner_val = df.iloc[inner_val_idx][TARGET_DRAWDOWN]

    def objective(trial):
        params = _suggest_xgb_params(trial, task)

        if task == "regression":
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)

        model.fit(
            X_inner_train, y_inner_train,
            eval_set=[(X_inner_val, y_inner_val)],
            verbose=False,
        )

        if task == "regression":
            preds = model.predict(X_inner_val)
            return _spearman_ic(y_inner_val.values, preds, inner_val_dates)
        else:
            preds = model.predict_proba(X_inner_val)[:, 1]
            # Guard against single-class inner-val (rare but possible for
            # a contiguous 10% block during quiet markets). AUC is
            # undefined in that case; fall back to 0.5 (no-skill baseline)
            # rather than crashing the trial.
            if len(np.unique(y_inner_val)) < 2:
                return 0.5
            return roc_auc_score(y_inner_val, preds)

    return objective

def tune_hyperparameters(df, train_idx):
    """Run Optuna on fold 1 to pick hyperparameters for both models.

    Returns a dict with two sub-dicts: 'regression' and 'classification',
    each containing the best hyperparameters found by Optuna.

    Folds 2-8 reuse these hyperparameters — we don't re-tune per fold
    because that overfits hyperparameters to each fold's noise (standard
    simplification for walk-forward setups).
    """
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters on fold 1 with {N_OPTUNA_TRIALS} trials per model")
    print(f"{'='*60}")

    results = {}

    for task in ["regression", "classification"]:
        print(f"\n  Tuning {task} model...")
        objective = make_objective(df, train_idx, task)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

        print(f"    Best value: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        results[task] = study.best_params

    return results

def train_fold(df, train_idx, val_idx, hyperparams, fold_num):
    """Train both models on one fold and predict on the outer validation set.

    For each model (regression, classification):
    1. Split this fold's training data into inner-train + inner-val (last
       10% by date) for XGBoost early stopping.
    2. Fit the model on inner-train with early stopping on inner-val.
    3. Sanity-check that best_iteration was populated (early stopping bug
       check flagged during research).
    4. Predict on the outer val set.
    5. Save the trained model to disk.

    Returns a DataFrame with one row per outer-val sample containing:
    ticker, date, target actuals, predictions, and fold number.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    inner_train_idx, inner_val_idx = inner_early_stopping_split(
        df, train_idx, inner_val_frac=0.1
    )

    X_inner_train = df.iloc[inner_train_idx][FEATURE_COLS]
    X_inner_val = df.iloc[inner_val_idx][FEATURE_COLS]
    X_outer_val = df.iloc[val_idx][FEATURE_COLS]

    # Prep the output predictions DataFrame
    predictions = df.iloc[val_idx][["ticker", "date", TARGET_RETURN, TARGET_DRAWDOWN]].copy()
    predictions = predictions.rename(columns={
        TARGET_RETURN: "actual_return",
        TARGET_DRAWDOWN: "actual_drawdown",
    })
    predictions["fold"] = fold_num

    # --- Regression: predict return ---
    y_inner_train_reg = df.iloc[inner_train_idx]["target_ret_21d_winsor"]
    y_inner_val_reg = df.iloc[inner_val_idx]["target_ret_21d_winsor"]

    reg_params = {
        **hyperparams["regression"],
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
    }
    reg_model = xgb.XGBRegressor(**reg_params)
    reg_model.fit(
        X_inner_train, y_inner_train_reg,
        eval_set=[(X_inner_val, y_inner_val_reg)],
        verbose=False,
    )
    assert hasattr(reg_model, "best_iteration") and reg_model.best_iteration is not None, \
        "Regression model best_iteration not populated — early stopping may not be working"

    predictions["pred_return"] = reg_model.predict(X_outer_val)
    reg_model.save_model(str(MODELS_DIR / f"fold_{fold_num}_regression.json"))

    # --- Classification: predict drawdown probability ---
    y_inner_train_clf = df.iloc[inner_train_idx][TARGET_DRAWDOWN]
    y_inner_val_clf = df.iloc[inner_val_idx][TARGET_DRAWDOWN]

    clf_params = {
        **hyperparams["classification"],
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
    }
    clf_model = xgb.XGBClassifier(**clf_params)
    clf_model.fit(
        X_inner_train, y_inner_train_clf,
        eval_set=[(X_inner_val, y_inner_val_clf)],
        verbose=False,
    )
    assert hasattr(clf_model, "best_iteration") and clf_model.best_iteration is not None, \
        "Classifier best_iteration not populated — early stopping may not be working"

    predictions["pred_drawdown_prob"] = clf_model.predict_proba(X_outer_val)[:, 1]
    clf_model.save_model(str(MODELS_DIR / f"fold_{fold_num}_classification.json"))

    print(f"  Fold {fold_num}: reg best_iter={reg_model.best_iteration}, "
          f"clf best_iter={clf_model.best_iteration}, "
          f"outer_val_rows={len(val_idx):,}")

    return predictions

def main():
    """Orchestrate the full training pipeline.

    1. Load and prepare training data (winsorize regression target)
    2. Initialize the purged walk-forward CV splitter
    3. Tune hyperparameters on fold 1 only (Optuna, 2 studies)
    4. Train both models on each of the 8 folds, predicting on outer val
    5. Concatenate all fold predictions and save to parquet
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prep()
    df = df.reset_index(drop=True)  # guarantee positional index 0..N-1

    cv = PurgedWalkForwardCV(start_year=2015, end_year=2022, purge_days=21)
    print(f"\nWalk-forward CV: {cv.get_n_splits()} folds from {cv.start_year} to {cv.end_year}\n")

    # Fold 1 is special — tune hyperparameters on it
    first_fold_generator = cv.split(df)
    train_idx_f1, val_idx_f1 = next(first_fold_generator)
    hyperparams = tune_hyperparameters(df, train_idx_f1)

    # Now train all 8 folds using the tuned hyperparameters
    print(f"\n{'='*60}")
    print(f"Training all {cv.get_n_splits()} folds with tuned hyperparameters")
    print(f"{'='*60}")

    all_predictions = []

    # Fold 1 uses the indices we already have
    all_predictions.append(train_fold(df, train_idx_f1, val_idx_f1, hyperparams, fold_num=1))

    # Folds 2-8 come from the generator
    for fold_num, (train_idx, val_idx) in enumerate(cv.split(df), start=1):
        if fold_num == 1:
            continue  # already handled
        all_predictions.append(train_fold(df, train_idx, val_idx, hyperparams, fold_num=fold_num))

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(PREDICTIONS_FILE)

    print(f"\n{'='*60}")
    print(f"Done. Predictions saved to {PREDICTIONS_FILE}")
    print(f"  Rows: {len(predictions):,}")
    print(f"  Folds: {predictions['fold'].nunique()}")
    print(f"  Date range: {predictions['date'].min().date()} to {predictions['date'].max().date()}")
    print(f"  Models saved to {MODELS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()