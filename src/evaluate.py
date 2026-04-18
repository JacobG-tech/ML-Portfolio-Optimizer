"""Stage 3 evaluation: assess CV predictions from train.py.

Reads out-of-fold predictions and produces:
1. Regression diagnostics — per-date and pooled Spearman IC, ICIR
2. Classification diagnostics — AUC, Brier score, calibration plot
3. Feature importance — per-fold and aggregated, saved as PNG charts
4. Go/no-go recommendation for Stage 4

Inputs: data/predictions/cv_predictions.parquet (from train.py)
        results/models/fold_{N}_{type}.json (from train.py)

Outputs: results/evaluation_summary.csv
         results/calibration_plot.png
         results/feature_importance_regression.png
         results/feature_importance_classification.png

Go/no-go thresholds (research-validated against equity quant literature):
  Green  (proceed):  IC >= 0.03 AND ICIR >= 0.40
  Yellow (marginal): IC >= 0.01 AND ICIR >= 0.20
  Red    (rework):   below yellow
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


PREDICTIONS_FILE = Path("data/predictions/cv_predictions.parquet")
MODELS_DIR = Path("results/models")
RESULTS_DIR = Path("results")
SUMMARY_FILE = RESULTS_DIR / "evaluation_summary.csv"
CALIBRATION_PLOT = RESULTS_DIR / "calibration_plot.png"
FI_REG_PLOT = RESULTS_DIR / "feature_importance_regression.png"
FI_CLF_PLOT = RESULTS_DIR / "feature_importance_classification.png"

# Go/no-go thresholds
IC_GREEN = 0.03
IC_YELLOW = 0.01
ICIR_GREEN = 0.40
ICIR_YELLOW = 0.20

# Feature columns — must match train.py's FEATURE_COLS order so that
# XGBoost's feature_importances_ array lines up correctly.
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


def load_predictions():
    """Load the CV predictions parquet produced by train.py."""
    print(f"Loading {PREDICTIONS_FILE}...")
    df = pd.read_parquet(PREDICTIONS_FILE)
    print(f"  {len(df):,} rows across {df['fold'].nunique()} folds")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def per_date_ic(df):
    """Compute Spearman rank correlation between predictions and actual
    returns for each date separately.

    Returns a Series indexed by date with one IC per date.

    Degenerate dates (fewer than 2 tickers, or constant predictions/actuals
    producing NaN) contribute 0.0, consistent with train.py's IC metric.
    """
    def daily_corr(group):
        if len(group) < 2:
            return 0.0
        corr = spearmanr(group["actual_return"], group["pred_return"]).correlation
        return 0.0 if np.isnan(corr) else corr

    return df.groupby("date").apply(daily_corr, include_groups=False)


def regression_metrics(df):
    """Compute IC, ICIR, and per-fold IC breakdown.

    Returns
    -------
    summary : dict
        Overall metrics: mean_ic_pooled, icir_pooled, mean_ic_per_fold
    per_fold : pd.DataFrame
        One row per fold with mean_ic, ic_std, icir, n_dates, n_obs
    """
    # Pooled: one IC per date across ALL folds (each date in exactly 1 fold
    # since walk-forward creates non-overlapping validation periods)
    pooled_ic = per_date_ic(df)
    mean_ic_pooled = pooled_ic.mean()
    std_ic_pooled = pooled_ic.std()
    icir_pooled = mean_ic_pooled / std_ic_pooled if std_ic_pooled > 0 else np.nan

    # Per-fold breakdown
    per_fold_rows = []
    for fold_num, fold_df in df.groupby("fold"):
        fold_ic = per_date_ic(fold_df)
        mean_ic = fold_ic.mean()
        ic_std = fold_ic.std()
        icir = mean_ic / ic_std if ic_std > 0 else np.nan
        per_fold_rows.append({
            "fold": fold_num,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "icir": icir,
            "n_dates": len(fold_ic),
            "n_obs": len(fold_df),
        })
    per_fold = pd.DataFrame(per_fold_rows)

    summary = {
        "mean_ic_pooled": mean_ic_pooled,
        "std_ic_pooled": std_ic_pooled,
        "icir_pooled": icir_pooled,
        "mean_ic_per_fold_avg": per_fold["mean_ic"].mean(),
        "icir_per_fold_avg": per_fold["icir"].mean(),
    }

    return summary, per_fold


def print_regression_report(summary, per_fold):
    """Print a human-readable regression summary."""
    print("\n" + "=" * 60)
    print("REGRESSION METRICS (forward 21-day return prediction)")
    print("=" * 60)

    print(f"\nPooled across all folds ({len(per_fold)} folds):")
    print(f"  Mean IC:           {summary['mean_ic_pooled']:+.4f}")
    print(f"  Std of daily IC:   {summary['std_ic_pooled']:.4f}")
    print(f"  ICIR:              {summary['icir_pooled']:+.3f}")

    print(f"\nPer-fold breakdown:")
    print(f"  {'Fold':<5} {'Mean IC':>10} {'IC Std':>8} {'ICIR':>8} {'N Dates':>8} {'N Obs':>8}")
    for _, row in per_fold.iterrows():
        print(f"  {int(row['fold']):<5} "
              f"{row['mean_ic']:>+10.4f} "
              f"{row['ic_std']:>8.4f} "
              f"{row['icir']:>+8.3f} "
              f"{int(row['n_dates']):>8} "
              f"{int(row['n_obs']):>8,}")

    print(f"\n  Average of per-fold mean IC:  {summary['mean_ic_per_fold_avg']:+.4f}")
    print(f"  Average of per-fold ICIR:     {summary['icir_per_fold_avg']:+.3f}")


def classification_metrics(df):
    """Compute AUC and Brier score, both pooled and per-fold.

    AUC measures discrimination (can the model rank drawdowns vs non-drawdowns).
    Brier score measures calibration + discrimination combined (MSE on
    probabilities). Lower Brier is better; 0.25 is the no-skill baseline
    for a balanced dataset.

    Returns
    -------
    summary : dict
        Overall metrics: auc_pooled, brier_pooled
    per_fold : pd.DataFrame
        One row per fold with auc, brier, n_obs, pos_rate
    """
    # Defensive guard: evaluate.py expects clean binary labels. If NaNs are
    # present, the input file is the wrong one (e.g. features_prediction.parquet
    # instead of cv_predictions.parquet) and metrics would be silently wrong.
    if df["actual_drawdown"].isna().any():
        n_nan = df["actual_drawdown"].isna().sum()
        raise ValueError(
            f"actual_drawdown has {n_nan} NaN values. evaluate.py should only "
            f"be run on the CV predictions file (data/predictions/cv_predictions.parquet), "
            f"not the prediction set which contains tail rows with unknown labels."
        )

    summary = {}

    # Pooled AUC/Brier across all 8 folds
    y_true_all = df["actual_drawdown"].values
    y_prob_all = df["pred_drawdown_prob"].values

    if len(np.unique(y_true_all)) >= 2:
        summary["auc_pooled"] = roc_auc_score(y_true_all, y_prob_all)
    else:
        summary["auc_pooled"] = np.nan

    summary["brier_pooled"] = brier_score_loss(y_true_all, y_prob_all)
    summary["positive_rate_pooled"] = y_true_all.mean()

    # Per-fold breakdown
    per_fold_rows = []
    for fold_num, fold_df in df.groupby("fold"):
        y_true = fold_df["actual_drawdown"].values
        y_prob = fold_df["pred_drawdown_prob"].values

        if len(np.unique(y_true)) >= 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        per_fold_rows.append({
            "fold": fold_num,
            "auc": auc,
            "brier": brier_score_loss(y_true, y_prob),
            "pos_rate": y_true.mean(),
            "n_obs": len(fold_df),
        })
    per_fold = pd.DataFrame(per_fold_rows)

    return summary, per_fold


def plot_calibration(df, output_path):
    """Save a calibration plot (reliability diagram) as PNG.

    Bins predicted probabilities into deciles (10 quantile-based bins),
    plots mean predicted probability vs actual positive rate per bin.
    Perfect calibration is the diagonal y=x line.
    """
    y_true = df["actual_drawdown"].values
    y_prob = df["pred_drawdown_prob"].values

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual positive rate")
    ax.set_title("Drawdown classifier calibration (10 quantile bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration plot saved to {output_path}")


def print_classification_report(summary, per_fold):
    """Print a human-readable classification summary."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS (>5% drawdown in next 21 days)")
    print("=" * 60)

    print(f"\nPooled across all folds:")
    print(f"  AUC:                {summary['auc_pooled']:.4f}  (0.5 = no skill)")
    print(f"  Brier score:        {summary['brier_pooled']:.4f}  (lower is better)")
    print(f"  Positive rate:      {summary['positive_rate_pooled']:.1%}")

    print(f"\nPer-fold breakdown:")
    print(f"  {'Fold':<5} {'AUC':>8} {'Brier':>8} {'Pos Rate':>10} {'N Obs':>8}")
    for _, row in per_fold.iterrows():
        print(f"  {int(row['fold']):<5} "
              f"{row['auc']:>8.4f} "
              f"{row['brier']:>8.4f} "
              f"{row['pos_rate']:>9.1%} "
              f"{int(row['n_obs']):>8,}")


def compute_feature_importance(task):
    """Load all 8 fold models of a given task and extract feature importances.

    Returns a DataFrame with one row per feature and columns for each fold's
    gain-based importance, plus a 'mean' column averaging across folds.

    Handles both key conventions XGBoost may return from get_score():
    - Named columns ("ret_21d", "sector_energy") if trained with pandas DataFrames
    - Positional names ("f0", "f5") if trained with numpy arrays
    """
    importances_by_fold = {}
    feature_to_idx = {name: i for i, name in enumerate(FEATURE_COLS)}

    for fold_num in range(1, 9):
        model_path = MODELS_DIR / f"fold_{fold_num}_{task}.json"

        if task == "regression":
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        # Request gain-based importance. Features not used in any split are
        # omitted from the returned dict entirely.
        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        gains = np.zeros(len(FEATURE_COLS))
        for key, val in gain_dict.items():
            if key in feature_to_idx:
                # Named column case (pandas-trained)
                gains[feature_to_idx[key]] = val
            elif key.startswith("f") and key[1:].isdigit():
                # Positional fallback ("f0", "f5", etc.)
                idx = int(key[1:])
                if 0 <= idx < len(FEATURE_COLS):
                    gains[idx] = val
                else:
                    raise ValueError(
                        f"Feature index {idx} out of range for {len(FEATURE_COLS)} "
                        f"features (fold {fold_num}, {task})"
                    )
            else:
                raise ValueError(
                    f"Unrecognized feature key '{key}' from XGBoost model "
                    f"(fold {fold_num}, {task}). Expected a FEATURE_COLS name or "
                    f"'f<N>' positional key."
                )

        importances_by_fold[fold_num] = gains

    fi_df = pd.DataFrame(importances_by_fold, index=FEATURE_COLS)
    fi_df.columns = [f"fold_{i}" for i in fi_df.columns]
    fi_df["mean"] = fi_df.mean(axis=1)
    fi_df = fi_df.sort_values("mean", ascending=False)

    return fi_df


def plot_feature_importance(fi_df, task, output_path):
    """Save a horizontal bar chart of aggregated feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))

    features_sorted = fi_df.index.tolist()
    means = fi_df["mean"].values

    ax.barh(range(len(features_sorted)), means, color="steelblue")
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()  # highest importance at top
    ax.set_xlabel("Mean gain across 8 folds")
    ax.set_title(f"Feature importance (gain) — {task}")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance chart saved to {output_path}")


def go_no_go(summary):
    """Print a go/no-go recommendation based on regression IC and ICIR.

    Thresholds sourced from equity quant literature:
      Green  (proceed):  IC >= 0.03 AND ICIR >= 0.40
      Yellow (marginal): IC >= 0.01 AND ICIR >= 0.20
      Red    (rework):   below yellow
    """
    ic = summary["mean_ic_pooled"]
    icir = summary["icir_pooled"]

    print("\n" + "=" * 60)
    print("STAGE 3 → STAGE 4 DECISION")
    print("=" * 60)

    if ic >= IC_GREEN and icir >= ICIR_GREEN:
        verdict = "GREEN — proceed to Stage 4 (portfolio optimization)"
        reason = f"IC={ic:+.4f} >= {IC_GREEN}, ICIR={icir:+.3f} >= {ICIR_GREEN}"
    elif ic >= IC_YELLOW and icir >= ICIR_YELLOW:
        verdict = "YELLOW — marginal; consider adding features before proceeding"
        reason = f"IC={ic:+.4f}, ICIR={icir:+.3f} (below green but above red)"
    else:
        verdict = "RED — signal too weak; add features or redesign approach"
        reason = f"IC={ic:+.4f}, ICIR={icir:+.3f} (below yellow thresholds)"

    print(f"\n  Verdict: {verdict}")
    print(f"  Reason:  {reason}")

    if verdict.startswith("RED"):
        print(f"\n  Suggested next steps:")
        print(f"  1. Review feature importance charts — is any feature dominant?")
        print(f"  2. Consider v2 features: cross-sectional rank-transforms,")
        print(f"     sector-excess returns, additional technical indicators.")
        print(f"  3. Check per-fold breakdown — is signal time-varying?")


def main():
    """Orchestrate the full evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions()

    # Regression metrics
    reg_summary, reg_per_fold = regression_metrics(df)
    print_regression_report(reg_summary, reg_per_fold)

    # Classification metrics
    clf_summary, clf_per_fold = classification_metrics(df)
    print_classification_report(clf_summary, clf_per_fold)

    # Calibration plot
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60 + "\n")
    plot_calibration(df, CALIBRATION_PLOT)

    # Feature importance for both models
    print("\nComputing feature importance...")
    fi_reg = compute_feature_importance("regression")
    fi_clf = compute_feature_importance("classification")
    plot_feature_importance(fi_reg, "regression", FI_REG_PLOT)
    plot_feature_importance(fi_clf, "classification", FI_CLF_PLOT)

    # Sanity-check that regression and classification have the same folds
    # before merging. Catches partial-rerun / corrupted-input scenarios.
    reg_folds = set(reg_per_fold["fold"].astype(int))
    clf_folds = set(clf_per_fold["fold"].astype(int))
    if reg_folds != clf_folds:
        raise ValueError(
            f"Fold mismatch between regression and classification metrics. "
            f"Regression folds: {sorted(reg_folds)}, "
            f"Classification folds: {sorted(clf_folds)}"
        )

    # Save summary CSV with all metrics
    summary_rows = []
    for _, row in reg_per_fold.iterrows():
        summary_rows.append({
            "fold": int(row["fold"]),
            "regression_mean_ic": row["mean_ic"],
            "regression_icir": row["icir"],
            "regression_n_dates": int(row["n_dates"]),
        })
    summary_df = pd.DataFrame(summary_rows).merge(
        clf_per_fold.rename(columns={
            "auc": "classification_auc",
            "brier": "classification_brier",
            "pos_rate": "classification_pos_rate",
            "n_obs": "n_obs",
        }).drop(columns=["n_obs"]).assign(fold=lambda d: d["fold"].astype(int)),
        on="fold",
    )
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\n  Per-fold summary saved to {SUMMARY_FILE}")

    # Save top features to summary too
    print(f"\nTop 5 features (regression, by mean gain):")
    for feat, gain in fi_reg["mean"].head(5).items():
        print(f"    {feat:<35} {gain:>10.2f}")
    print(f"\nTop 5 features (classification, by mean gain):")
    for feat, gain in fi_clf["mean"].head(5).items():
        print(f"    {feat:<35} {gain:>10.2f}")

    # Go/no-go
    go_no_go(reg_summary)


if __name__ == "__main__":
    main()