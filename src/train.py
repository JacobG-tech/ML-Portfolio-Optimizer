"""Stage 3 training orchestrator: purged walk-forward CV with XGBoost.

For each of 8 annual validation folds:
1. Split fold's training set into inner-train + inner-val (last 10% by date)
2. Train XGBRegressor for forward return (MSE loss)
3. Train XGBClassifier for >5% drawdown in next 21 days (log loss)
4. Use inner-val for XGBoost early stopping
5. Predict on the outer validation fold
6. Stash predictions keyed by (ticker, date, fold)

Fold 1 runs ~30 Optuna TPE trials per model to pick hyperparameters.
Folds 2-8 reuse those. Predictions are saved for evaluate.py to consume.

Variants:
- v1:  raw features, 21d return target (baseline)
- v2a: rank-transformed features, 21d return target
- v2b: v2a + sector_excess_ret_21d_rank
- v2c: v2b features, 21d return-rank target
- v2d: v2a features minus sector one-hots
- v3a: v2a features, 63d return target (longer horizon)

Key design decisions in 03_ROADMAP.md and 04_DECISIONS_AND_LESSONS.md.
"""

import argparse
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cv import PurgedWalkForwardCV, inner_early_stopping_split


# Static paths
TRAINING_FILE = Path("data/processed/features_training.parquet")
PREDICTIONS_DIR = Path("data/predictions")
MODELS_BASE_DIR = Path("results/models")

# Reproducibility
RANDOM_SEED = 42

# Tuning config
N_OPTUNA_TRIALS = 30
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_MAX = 500

# Target winsorization percentiles (applied only to variants that use the
# raw winsorized target; rank-transformed targets are already bounded)
WINSOR_LOWER_Q = 0.01
WINSOR_UPPER_Q = 0.99

# Feature column groupings
V1_NUMERIC_FEATURE_COLS = [
    "ret_21d", "ret_63d", "ret_252d", "price_to_sma50",
    "vol_20d", "range_pct_20d", "max_dd_90d",
    "beta_60d", "excess_ret_21d",
    "rsi_14", "bb_position_20", "macd_hist", "atr_14_pct",
    "volume_ratio_20d",
]
V2_RANK_FEATURE_COLS = [f"{c}_rank" for c in V1_NUMERIC_FEATURE_COLS]
SECTOR_EXCESS_RANK_COL = "sector_excess_ret_21d_rank"
SECTOR_ONEHOT_COLS = [
    "sector_communication_services", "sector_consumer_discretionary",
    "sector_consumer_staples", "sector_energy", "sector_financials",
    "sector_health_care", "sector_industrials", "sector_information_technology",
    "sector_materials", "sector_real_estate", "sector_utilities",
]

# Target column names
TARGET_RETURN_RAW = "target_ret_21d"
TARGET_RETURN_63D = "target_ret_63d"
TARGET_RETURN_RANK = "target_ret_21d_rank"
TARGET_DRAWDOWN = "target_dd5_21d"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_variant_config(variant):
    """Return per-variant feature list, target column, winsorize flag,
    purge_days, and output paths.
    """
    if variant == "v1":
        feature_cols = V1_NUMERIC_FEATURE_COLS + SECTOR_ONEHOT_COLS
        regression_target = TARGET_RETURN_RAW
        winsorize_target = True
        purge_days = 21

    elif variant == "v2a":
        feature_cols = V2_RANK_FEATURE_COLS + SECTOR_ONEHOT_COLS
        regression_target = TARGET_RETURN_RAW
        winsorize_target = True
        purge_days = 21

    elif variant == "v2b":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL] + SECTOR_ONEHOT_COLS
        regression_target = TARGET_RETURN_RAW
        winsorize_target = True
        purge_days = 21

    elif variant == "v2c":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL] + SECTOR_ONEHOT_COLS
        regression_target = TARGET_RETURN_RANK
        winsorize_target = False
        purge_days = 21

    elif variant == "v2d":
        # Sector-neutral: no sector one-hots
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL]
        regression_target = TARGET_RETURN_RAW
        winsorize_target = True
        purge_days = 21

    elif variant == "v3a":
        # Longer horizon: 63-day forward return target. Same features as v2a.
        # Purge extended to 63 days to match the target horizon.
        feature_cols = V2_RANK_FEATURE_COLS + SECTOR_ONEHOT_COLS
        regression_target = TARGET_RETURN_63D
        winsorize_target = True
        purge_days = 63

    elif variant == "v3d":
        # Sector-neutral at 63-day horizon: v3a features minus sector one-hots,
        # plus sector_excess_ret_21d_rank to capture within-sector signal.
        # Tests whether the longer-horizon signal is real factor content or
        # sector-memorization in disguise.
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL]
        regression_target = TARGET_RETURN_63D
        winsorize_target = True
        purge_days = 63

    else:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Must be one of v1, v2a, v2b, v2c, v2d, v3a, v3d."
        )

    return {
        "variant": variant,
        "feature_cols": feature_cols,
        "regression_target": regression_target,
        "winsorize_target": winsorize_target,
        "purge_days": purge_days,
        "predictions_file": PREDICTIONS_DIR / f"cv_predictions_{variant}.parquet",
        "models_dir": MODELS_BASE_DIR / variant,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 training for ML Portfolio Optimizer"
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["v1", "v2a", "v2b", "v2c", "v2d", "v3a", "v3d"],
        help="Which model variant to train (affects features, target, purge)",
    )
    return parser.parse_args()


def winsorize_per_date(df, target_col, lower_q=WINSOR_LOWER_Q, upper_q=WINSOR_UPPER_Q):
    """Winsorize a target column at per-date percentiles."""
    def clip_group(group):
        lower = group.quantile(lower_q)
        upper = group.quantile(upper_q)
        return group.clip(lower=lower, upper=upper)

    return df.groupby("date")[target_col].transform(clip_group)


def load_and_prep(config):
    """Load training parquet and prepare the regression target per variant."""
    print(f"Loading {TRAINING_FILE}...")
    df = pd.read_parquet(TRAINING_FILE)
    print(f"  {len(df):,} rows, date range {df['date'].min().date()} to {df['date'].max().date()}")

    target_col = config["regression_target"]

    if config["winsorize_target"]:
        print(f"Winsorizing {target_col} at per-date [{WINSOR_LOWER_Q:.0%}, {WINSOR_UPPER_Q:.0%}]...")
        df["regression_target_used"] = winsorize_per_date(df, target_col)

        raw_std = df[target_col].std()
        used_std = df["regression_target_used"].std()
        raw_max = df[target_col].max()
        used_max = df["regression_target_used"].max()
        print(f"  Raw target:      std={raw_std:.3f}, max={raw_max:.3f}")
        print(f"  Winsorized:      std={used_std:.3f}, max={used_max:.3f}")
    else:
        print(f"Using {target_col} as regression target (no winsorization — already bounded)")
        df["regression_target_used"] = df[target_col]
        used_std = df["regression_target_used"].std()
        used_min = df["regression_target_used"].min()
        used_max = df["regression_target_used"].max()
        print(f"  Target stats:    std={used_std:.3f}, min={used_min:.3f}, max={used_max:.3f}")

    return df


def _suggest_xgb_params(trial, task):
    """Sample XGBoost hyperparameters from the Optuna trial."""
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
    """Cross-sectional Spearman IC averaged across dates."""
    def daily_corr(group):
        if len(group) < 2:
            return 0.0
        corr = spearmanr(group["y_true"], group["y_pred"]).correlation
        return 0.0 if np.isnan(corr) else corr

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "date": dates})
    daily_ic = df.groupby("date").apply(daily_corr, include_groups=False)
    return daily_ic.mean()


def make_objective(df, train_idx, task, feature_cols):
    """Factory: returns an Optuna objective function for the given task."""
    inner_train_idx, inner_val_idx = inner_early_stopping_split(
        df, train_idx, inner_val_frac=0.1
    )

    X_inner_train = df.iloc[inner_train_idx][feature_cols]
    X_inner_val = df.iloc[inner_val_idx][feature_cols]
    inner_val_dates = df.iloc[inner_val_idx]["date"].values

    if task == "regression":
        y_inner_train = df.iloc[inner_train_idx]["regression_target_used"]
        y_inner_val = df.iloc[inner_val_idx]["regression_target_used"]
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
            if len(np.unique(y_inner_val)) < 2:
                return 0.5
            return roc_auc_score(y_inner_val, preds)

    return objective


def tune_hyperparameters(df, train_idx, feature_cols):
    """Run Optuna on fold 1 to pick hyperparameters for both models."""
    print(f"\n{'='*60}")
    print(f"Tuning hyperparameters on fold 1 with {N_OPTUNA_TRIALS} trials per model")
    print(f"{'='*60}")

    results = {}

    for task in ["regression", "classification"]:
        print(f"\n  Tuning {task} model...")
        objective = make_objective(df, train_idx, task, feature_cols)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

        print(f"    Best value: {study.best_value:.4f}")
        print(f"    Best params: {study.best_params}")

        results[task] = study.best_params

    return results


def train_fold(df, train_idx, val_idx, hyperparams, fold_num, config):
    """Train both models on one fold and predict on the outer validation set."""
    models_dir = config["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config["feature_cols"]

    inner_train_idx, inner_val_idx = inner_early_stopping_split(
        df, train_idx, inner_val_frac=0.1
    )

    X_inner_train = df.iloc[inner_train_idx][feature_cols]
    X_inner_val = df.iloc[inner_val_idx][feature_cols]
    X_outer_val = df.iloc[val_idx][feature_cols]

    # Predictions store BOTH 21-day and 63-day raw returns as actuals.
    # evaluate.py picks the right column per variant (21d-horizon variants
    # score against actual_return_21d, v3a scores against actual_return_63d).
    # Storing both lets compare mode evaluate every variant on its own horizon.
    predictions = df.iloc[val_idx][
        ["ticker", "date", TARGET_RETURN_RAW, TARGET_RETURN_63D, TARGET_DRAWDOWN]
    ].copy()
    predictions = predictions.rename(columns={
        TARGET_RETURN_RAW: "actual_return_21d",
        TARGET_RETURN_63D: "actual_return_63d",
        TARGET_DRAWDOWN: "actual_drawdown",
    })
    predictions["fold"] = fold_num

    # --- Regression: predict return ---
    y_inner_train_reg = df.iloc[inner_train_idx]["regression_target_used"]
    y_inner_val_reg = df.iloc[inner_val_idx]["regression_target_used"]

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
    reg_model.save_model(str(models_dir / f"fold_{fold_num}_regression.json"))

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
    clf_model.save_model(str(models_dir / f"fold_{fold_num}_classification.json"))

    print(f"  Fold {fold_num}: reg best_iter={reg_model.best_iteration}, "
          f"clf best_iter={clf_model.best_iteration}, "
          f"outer_val_rows={len(val_idx):,}")

    return predictions


def main():
    """Orchestrate the full training pipeline for the selected variant."""
    args = parse_args()
    config = get_variant_config(args.variant)

    print(f"{'='*60}")
    print(f"TRAINING VARIANT: {config['variant']}")
    print(f"  Features:         {len(config['feature_cols'])}")
    print(f"  Regression target: {config['regression_target']}")
    print(f"  Winsorize target: {config['winsorize_target']}")
    print(f"  Purge days:       {config['purge_days']}")
    print(f"  Output file:      {config['predictions_file']}")
    print(f"  Models dir:       {config['models_dir']}")
    print(f"{'='*60}\n")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prep(config)
    df = df.reset_index(drop=True)

    cv = PurgedWalkForwardCV(
        start_year=2015, end_year=2022, purge_days=config["purge_days"]
    )
    print(f"\nWalk-forward CV: {cv.get_n_splits()} folds from {cv.start_year} to {cv.end_year} "
          f"(purge={config['purge_days']} days)\n")

    first_fold_generator = cv.split(df)
    train_idx_f1, val_idx_f1 = next(first_fold_generator)
    hyperparams = tune_hyperparameters(df, train_idx_f1, config["feature_cols"])

    print(f"\n{'='*60}")
    print(f"Training all {cv.get_n_splits()} folds with tuned hyperparameters")
    print(f"{'='*60}")

    all_predictions = []

    all_predictions.append(
        train_fold(df, train_idx_f1, val_idx_f1, hyperparams, fold_num=1, config=config)
    )

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(df), start=1):
        if fold_num == 1:
            continue
        all_predictions.append(
            train_fold(df, train_idx, val_idx, hyperparams, fold_num=fold_num, config=config)
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(config["predictions_file"])

    print(f"\n{'='*60}")
    print(f"Done. Predictions saved to {config['predictions_file']}")
    print(f"  Rows: {len(predictions):,}")
    print(f"  Folds: {predictions['fold'].nunique()}")
    print(f"  Date range: {predictions['date'].min().date()} to {predictions['date'].max().date()}")
    print(f"  Models saved to {config['models_dir']}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()