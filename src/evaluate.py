"""Stage 3 evaluation: assess CV predictions from train.py.

Two modes:

1. Single-variant mode — evaluate one variant end-to-end:
   python src/evaluate.py --variant v2b

   Produces per-variant charts and detailed per-fold breakdown.

2. Compare mode — compare all available variants side-by-side:
   python src/evaluate.py --compare

   Prints a comparison table and saves a per-fold IC chart showing
   all variants as lines.

Each variant is evaluated on its own target horizon (21-day for v1/v2*,
63-day for v3*). Spearman IC is rank-based, so even though v3a's
pred_return is in return-magnitude space and gets compared to raw 63-day
returns, the IC calculation re-ranks both inputs per date and produces
a meaningful cross-sectional ranking skill metric on v3a's own horizon.

Inputs (per variant):
  data/predictions/cv_predictions_{variant}.parquet  (contains both
    actual_return_21d and actual_return_63d columns)
  results/models/{variant}/fold_{N}_{type}.json

Outputs:
  Single-variant mode:
    results/evaluation_summary_{variant}.csv
    results/calibration_plot_{variant}.png
    results/feature_importance_regression_{variant}.png
    results/feature_importance_classification_{variant}.png
  Compare mode:
    results/variant_comparison.csv
    results/variant_comparison_ic.png

Go/no-go thresholds (research-validated against equity quant literature):
  Green  (proceed):  IC >= 0.03 AND ICIR >= 0.40
  Yellow (marginal): IC >= 0.01 AND ICIR >= 0.20
  Red    (rework):   below yellow
"""

import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve


PREDICTIONS_DIR = Path("data/predictions")
MODELS_BASE_DIR = Path("results/models")
RESULTS_DIR = Path("results")

ALL_VARIANTS = ["v1", "v2a", "v2b", "v2c", "v2d", "v3a", "v3d"]

# Go/no-go thresholds
IC_GREEN = 0.03
IC_YELLOW = 0.01
ICIR_GREEN = 0.40
ICIR_YELLOW = 0.20

# Feature column definitions per variant (must match train.py at fit time).
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


def get_variant_config(variant):
    """Return per-variant paths, feature list, evaluation target, and metadata.

    actual_return_col: which column in the predictions file to use as the
    ground truth. 21d-horizon variants use actual_return_21d; v3a uses
    actual_return_63d.
    """
    if variant == "v1":
        feature_cols = V1_NUMERIC_FEATURE_COLS + SECTOR_ONEHOT_COLS
        stage4_ready = "yes"
        notes = "baseline"
        horizon = "21d"
        actual_return_col = "actual_return_21d"
    elif variant == "v2a":
        feature_cols = V2_RANK_FEATURE_COLS + SECTOR_ONEHOT_COLS
        stage4_ready = "yes"
        notes = "rank features"
        horizon = "21d"
        actual_return_col = "actual_return_21d"
    elif variant == "v2b":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL] + SECTOR_ONEHOT_COLS
        stage4_ready = "yes"
        notes = "rank + sector-excess"
        horizon = "21d"
        actual_return_col = "actual_return_21d"
    elif variant == "v2c":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL] + SECTOR_ONEHOT_COLS
        stage4_ready = "partial"
        notes = "rank target; mu-hat needs mapping for Stage 4"
        horizon = "21d"
        actual_return_col = "actual_return_21d"
    elif variant == "v2d":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL]
        stage4_ready = "yes"
        notes = "sector-neutral (no sector one-hots)"
        horizon = "21d"
        actual_return_col = "actual_return_21d"
    elif variant == "v3a":
        feature_cols = V2_RANK_FEATURE_COLS + SECTOR_ONEHOT_COLS
        stage4_ready = "yes"
        notes = "63-day horizon (v2a features)"
        horizon = "63d"
        actual_return_col = "actual_return_63d"
    elif variant == "v3d":
        feature_cols = V2_RANK_FEATURE_COLS + [SECTOR_EXCESS_RANK_COL]
        stage4_ready = "yes"
        notes = "63-day sector-neutral"
        horizon = "63d"
        actual_return_col = "actual_return_63d"
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    return {
        "variant": variant,
        "feature_cols": feature_cols,
        "stage4_ready": stage4_ready,
        "notes": notes,
        "horizon": horizon,
        "actual_return_col": actual_return_col,
        "predictions_file": PREDICTIONS_DIR / f"cv_predictions_{variant}.parquet",
        "models_dir": MODELS_BASE_DIR / variant,
        "summary_csv": RESULTS_DIR / f"evaluation_summary_{variant}.csv",
        "calibration_plot": RESULTS_DIR / f"calibration_plot_{variant}.png",
        "fi_reg_plot": RESULTS_DIR / f"feature_importance_regression_{variant}.png",
        "fi_clf_plot": RESULTS_DIR / f"feature_importance_classification_{variant}.png",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3 evaluation for ML Portfolio Optimizer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--variant",
        choices=ALL_VARIANTS,
        help="Evaluate a single variant end-to-end with per-variant charts",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="Compare all detected variants side-by-side with a summary table",
    )
    return parser.parse_args()


def load_predictions(config):
    """Load a variant's CV predictions parquet and validate it has the
    expected actual-return column for this variant's evaluation horizon."""
    print(f"Loading {config['predictions_file']}...")
    df = pd.read_parquet(config["predictions_file"])
    print(f"  {len(df):,} rows across {df['fold'].nunique()} folds")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    required_col = config["actual_return_col"]
    if required_col not in df.columns:
        raise ValueError(
            f"Predictions file for {config['variant']} is missing column "
            f"'{required_col}' (needed for {config['horizon']} horizon evaluation). "
            f"Re-run train.py --variant {config['variant']} to regenerate with "
            f"both actual_return_21d and actual_return_63d columns."
        )

    return df


# ---- Regression metrics ----

def per_date_ic(df, actual_col):
    """Compute Spearman rank correlation between predictions and actual
    returns for each date separately.

    Returns a Series indexed by date with one IC per date. Degenerate
    dates (fewer than 2 tickers, or NaN spearman output) contribute 0.0.

    actual_col: name of the actuals column ('actual_return_21d' or
    'actual_return_63d' depending on variant horizon).
    """
    def daily_corr(group):
        if len(group) < 2:
            return 0.0
        corr = spearmanr(group[actual_col], group["pred_return"]).correlation
        return 0.0 if np.isnan(corr) else corr

    return df.groupby("date").apply(daily_corr, include_groups=False)


def regression_metrics(df, actual_col):
    """Compute pooled and per-fold IC / ICIR."""
    pooled_ic = per_date_ic(df, actual_col)
    mean_ic_pooled = pooled_ic.mean()
    std_ic_pooled = pooled_ic.std()
    icir_pooled = mean_ic_pooled / std_ic_pooled if std_ic_pooled > 0 else np.nan

    per_fold_rows = []
    for fold_num, fold_df in df.groupby("fold"):
        fold_ic = per_date_ic(fold_df, actual_col)
        mean_ic = fold_ic.mean()
        ic_std = fold_ic.std()
        icir = mean_ic / ic_std if ic_std > 0 else np.nan
        per_fold_rows.append({
            "fold": int(fold_num),
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
        "worst_fold_ic": per_fold["mean_ic"].min(),
        "best_fold_ic": per_fold["mean_ic"].max(),
        "mean_ic_per_fold_avg": per_fold["mean_ic"].mean(),
        "icir_per_fold_avg": per_fold["icir"].mean(),
    }
    return summary, per_fold


def print_regression_report(summary, per_fold, horizon):
    print("\n" + "=" * 60)
    print(f"REGRESSION METRICS (forward {horizon} return prediction)")
    print("=" * 60)

    print(f"\nPooled across all folds ({len(per_fold)} folds):")
    print(f"  Mean IC:           {summary['mean_ic_pooled']:+.4f}")
    print(f"  Std of daily IC:   {summary['std_ic_pooled']:.4f}")
    print(f"  ICIR:              {summary['icir_pooled']:+.3f}")
    print(f"  Worst fold IC:     {summary['worst_fold_ic']:+.4f}")
    print(f"  Best fold IC:      {summary['best_fold_ic']:+.4f}")

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


# ---- Classification metrics ----

def classification_metrics(df):
    """Compute pooled and per-fold AUC / Brier. Classifier target is always
    21-day drawdown regardless of regression variant horizon."""
    if df["actual_drawdown"].isna().any():
        n_nan = df["actual_drawdown"].isna().sum()
        raise ValueError(
            f"actual_drawdown has {n_nan} NaN values. evaluate.py should only "
            f"be run on CV predictions files, not the prediction set which "
            f"contains tail rows with unknown labels."
        )

    summary = {}
    y_true_all = df["actual_drawdown"].values
    y_prob_all = df["pred_drawdown_prob"].values

    if len(np.unique(y_true_all)) >= 2:
        summary["auc_pooled"] = roc_auc_score(y_true_all, y_prob_all)
    else:
        summary["auc_pooled"] = np.nan

    summary["brier_pooled"] = brier_score_loss(y_true_all, y_prob_all)
    summary["positive_rate_pooled"] = y_true_all.mean()

    per_fold_rows = []
    for fold_num, fold_df in df.groupby("fold"):
        y_true = fold_df["actual_drawdown"].values
        y_prob = fold_df["pred_drawdown_prob"].values

        if len(np.unique(y_true)) >= 2:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = np.nan

        per_fold_rows.append({
            "fold": int(fold_num),
            "auc": auc,
            "brier": brier_score_loss(y_true, y_prob),
            "pos_rate": y_true.mean(),
            "n_obs": len(fold_df),
        })
    per_fold = pd.DataFrame(per_fold_rows)
    return summary, per_fold


def print_classification_report(summary, per_fold):
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


# ---- Calibration ----

def plot_calibration(df, output_path):
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


# ---- Feature importance ----

def compute_feature_importance(models_dir, feature_cols, task):
    """Load all 8 fold models of a given task and extract feature importances."""
    importances_by_fold = {}
    feature_to_idx = {name: i for i, name in enumerate(feature_cols)}

    for fold_num in range(1, 9):
        model_path = models_dir / f"fold_{fold_num}_{task}.json"

        if task == "regression":
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()
        model.load_model(str(model_path))

        booster = model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        gains = np.zeros(len(feature_cols))
        for key, val in gain_dict.items():
            if key in feature_to_idx:
                gains[feature_to_idx[key]] = val
            elif key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < len(feature_cols):
                    gains[idx] = val
                else:
                    raise ValueError(
                        f"Feature index {idx} out of range for {len(feature_cols)} "
                        f"features (fold {fold_num}, {task})"
                    )
            else:
                raise ValueError(
                    f"Unrecognized feature key '{key}' from XGBoost model "
                    f"(fold {fold_num}, {task})."
                )

        importances_by_fold[fold_num] = gains

    fi_df = pd.DataFrame(importances_by_fold, index=feature_cols)
    fi_df.columns = [f"fold_{i}" for i in fi_df.columns]
    fi_df["mean"] = fi_df.mean(axis=1)
    fi_df = fi_df.sort_values("mean", ascending=False)
    return fi_df


def plot_feature_importance(fi_df, task, variant, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    features_sorted = fi_df.index.tolist()
    means = fi_df["mean"].values

    ax.barh(range(len(features_sorted)), means, color="steelblue")
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Mean gain across 8 folds")
    ax.set_title(f"Feature importance (gain) — {task} — {variant}")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance chart saved to {output_path}")


# ---- Go/no-go ----

def classify_verdict(mean_ic, icir):
    """Return verdict string based on thresholds."""
    if pd.isna(mean_ic) or pd.isna(icir):
        return "N/A"
    if mean_ic >= IC_GREEN and icir >= ICIR_GREEN:
        return "GREEN"
    elif mean_ic >= IC_YELLOW and icir >= ICIR_YELLOW:
        return "YELLOW"
    else:
        return "RED"


def go_no_go(summary, variant, horizon):
    ic = summary["mean_ic_pooled"]
    icir = summary["icir_pooled"]
    verdict = classify_verdict(ic, icir)

    print("\n" + "=" * 60)
    print(f"STAGE 3 → STAGE 4 DECISION ({variant}, {horizon} horizon)")
    print("=" * 60)

    if verdict == "GREEN":
        full = "GREEN — proceed to Stage 4 (portfolio optimization)"
        reason = f"IC={ic:+.4f} >= {IC_GREEN}, ICIR={icir:+.3f} >= {ICIR_GREEN}"
    elif verdict == "YELLOW":
        full = "YELLOW — marginal; consider adding features before proceeding"
        reason = f"IC={ic:+.4f}, ICIR={icir:+.3f} (below green but above red)"
    elif verdict == "N/A":
        full = "N/A — verdict could not be computed"
        reason = f"IC={ic}, ICIR={icir} (NaN in one or both metrics)"
    else:
        full = "RED — signal too weak; add features or redesign approach"
        reason = f"IC={ic:+.4f}, ICIR={icir:+.3f} (below yellow thresholds)"

    print(f"\n  Verdict: {full}")
    print(f"  Reason:  {reason}")


# ---- Single-variant evaluation ----

def evaluate_variant(variant):
    """Run full evaluation on a single variant."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = get_variant_config(variant)

    print(f"\n{'='*60}")
    print(f"EVALUATING VARIANT: {variant} ({config['horizon']} horizon)")
    print(f"  Actual column:  {config['actual_return_col']}")
    print(f"{'='*60}\n")

    df = load_predictions(config)

    # Regression (variant-specific horizon)
    reg_summary, reg_per_fold = regression_metrics(df, config["actual_return_col"])
    print_regression_report(reg_summary, reg_per_fold, config["horizon"])

    # Classification (always 21-day drawdown)
    clf_summary, clf_per_fold = classification_metrics(df)
    print_classification_report(clf_summary, clf_per_fold)

    # Charts
    print("\n" + "=" * 60)
    print("GENERATING CHARTS")
    print("=" * 60 + "\n")
    plot_calibration(df, config["calibration_plot"])

    print("\nComputing feature importance...")
    fi_reg = compute_feature_importance(config["models_dir"], config["feature_cols"], "regression")
    fi_clf = compute_feature_importance(config["models_dir"], config["feature_cols"], "classification")
    plot_feature_importance(fi_reg, "regression", variant, config["fi_reg_plot"])
    plot_feature_importance(fi_clf, "classification", variant, config["fi_clf_plot"])

    # Fold alignment check before merging
    reg_folds = set(reg_per_fold["fold"].astype(int))
    clf_folds = set(clf_per_fold["fold"].astype(int))
    if reg_folds != clf_folds:
        raise ValueError(
            f"Fold mismatch: regression folds {sorted(reg_folds)}, "
            f"classification folds {sorted(clf_folds)}"
        )

    # Per-fold summary CSV
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
        }).drop(columns=["n_obs"]).assign(fold=lambda d: d["fold"].astype(int)),
        on="fold",
    )
    summary_df.to_csv(config["summary_csv"], index=False)
    print(f"\n  Per-fold summary saved to {config['summary_csv']}")

    print(f"\nTop 5 features (regression, by mean gain):")
    for feat, gain in fi_reg["mean"].head(5).items():
        print(f"    {feat:<35} {gain:>10.4f}")
    print(f"\nTop 5 features (classification, by mean gain):")
    for feat, gain in fi_clf["mean"].head(5).items():
        print(f"    {feat:<35} {gain:>10.4f}")

    go_no_go(reg_summary, variant, config["horizon"])


# ---- Compare mode ----

def detect_available_variants():
    """Return a list of variants whose predictions file exists on disk."""
    return [
        v for v in ALL_VARIANTS
        if (PREDICTIONS_DIR / f"cv_predictions_{v}.parquet").exists()
    ]


def compute_variant_summary(variant):
    """Compute pooled metrics for one variant (used in compare mode)."""
    config = get_variant_config(variant)
    df = pd.read_parquet(config["predictions_file"])

    if config["actual_return_col"] not in df.columns:
        raise ValueError(
            f"Predictions file for {variant} is missing column "
            f"'{config['actual_return_col']}'. Re-run train.py --variant {variant}."
        )

    reg_summary, reg_per_fold = regression_metrics(df, config["actual_return_col"])
    clf_summary, _ = classification_metrics(df)

    verdict = classify_verdict(
        reg_summary["mean_ic_pooled"], reg_summary["icir_pooled"]
    )

    row = {
        "variant": variant,
        "horizon": config["horizon"],
        "mean_ic": reg_summary["mean_ic_pooled"],
        "icir": reg_summary["icir_pooled"],
        "worst_fold_ic": reg_summary["worst_fold_ic"],
        "best_fold_ic": reg_summary["best_fold_ic"],
        "auc": clf_summary["auc_pooled"],
        "brier": clf_summary["brier_pooled"],
        "verdict": verdict,
        "stage4_ready": config["stage4_ready"],
        "notes": config["notes"],
    }
    return row, reg_per_fold[["fold", "mean_ic"]]


def plot_variant_comparison(per_fold_by_variant, output_path):
    """Line chart: per-fold mean IC by variant."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(IC_GREEN, color="green", linestyle=":", linewidth=1, alpha=0.6,
               label=f"IC={IC_GREEN} (green threshold)")
    ax.axhline(IC_YELLOW, color="goldenrod", linestyle=":", linewidth=1, alpha=0.6,
               label=f"IC={IC_YELLOW} (yellow threshold)")

    colors = {
        "v1": "#888888", "v2a": "#3b7dd8", "v2b": "#e07a3b",
        "v2c": "#6b3bd8", "v2d": "#2ca02c", "v3a": "#d62728", "v3d": "#e377c2",
    }
    for variant, per_fold in per_fold_by_variant.items():
        ax.plot(
            per_fold["fold"], per_fold["mean_ic"],
            marker="o", linewidth=2,
            label=variant, color=colors.get(variant, "black"),
        )

    ax.set_xlabel("Fold (validation year 2015..2022)")
    ax.set_ylabel("Mean IC on fold (each variant on its own horizon)")
    ax.set_title("Per-fold regression IC across variants")
    ax.set_xticks(range(1, 9))
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart saved to {output_path}")


def compare_variants():
    """Compare all available variants side-by-side."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    available = detect_available_variants()
    if len(available) < 2:
        raise ValueError(
            f"Need at least 2 variants to compare; found {len(available)}: {available}"
        )

    print(f"\n{'='*60}")
    print(f"VARIANT COMPARISON")
    print(f"Detected: {', '.join(available)}")
    print(f"{'='*60}\n")

    rows = []
    per_fold_by_variant = {}
    for variant in available:
        row, per_fold = compute_variant_summary(variant)
        rows.append(row)
        per_fold_by_variant[variant] = per_fold

    comp_df = pd.DataFrame(rows)

    # Crown a best variant per horizon — cross-horizon comparisons of IC
    # aren't apples-to-apples since each variant's IC is measured against
    # its own forward-return horizon.
    best_by_horizon = {}  # horizon -> variant name
    for horizon, group in comp_df.groupby("horizon"):
        best_idx = group["mean_ic"].idxmax()
        best_by_horizon[horizon] = comp_df.loc[best_idx, "variant"]

    # Print the table — mark the best variant WITHIN each horizon group
    print(f"{'Variant':<7} {'Horiz':>6} {'Mean IC':>9} {'ICIR':>7} {'Worst IC':>9} "
          f"{'AUC':>7} {'Brier':>7} {'Verdict':>8} {'Stage4':>8}  Notes")
    print("-" * 105)
    for _, row in comp_df.iterrows():
        is_best_in_horizon = row["variant"] == best_by_horizon.get(row["horizon"])
        marker = " ←" if is_best_in_horizon else ""
        print(
            f"{row['variant']:<7} "
            f"{row['horizon']:>6} "
            f"{row['mean_ic']:>+9.4f} "
            f"{row['icir']:>+7.3f} "
            f"{row['worst_fold_ic']:>+9.4f} "
            f"{row['auc']:>7.4f} "
            f"{row['brier']:>7.4f} "
            f"{row['verdict']:>8} "
            f"{row['stage4_ready']:>8}  "
            f"{row['notes']}{marker}"
        )

    print(f"\nBest variant per horizon (cross-horizon IC is NOT directly comparable):")
    for horizon in sorted(best_by_horizon.keys()):
        print(f"  {horizon}: {best_by_horizon[horizon]}")

    comp_df.to_csv(RESULTS_DIR / "variant_comparison.csv", index=False)
    print(f"\nComparison CSV saved to {RESULTS_DIR / 'variant_comparison.csv'}")

    plot_variant_comparison(
        per_fold_by_variant,
        RESULTS_DIR / "variant_comparison_ic.png",
    )

    print(f"\nNote: 'best' is chosen by pooled mean IC on {len(available)} variants. "
          f"Differences below ~0.005 IC may not be statistically meaningful; "
          f"bootstrap confidence intervals are a candidate future extension.")


# ---- Entry point ----

def main():
    args = parse_args()
    if args.compare:
        compare_variants()
    else:
        evaluate_variant(args.variant)


if __name__ == "__main__":
    main()