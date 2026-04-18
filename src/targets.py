import pandas as pd
import numpy as np

FORWARD_WINDOW = 21
DRAWDOWN_THRESHOLD = 0.05


def add_target_ret_21d(panel):
    """Add target_ret_21d: forward 21-trading-day return.

    Value at date T is the return earned from T to T+21. This is the
    regression target — what the ML model will try to predict.

    NOTE: uses shift(-21), which looks FORWARD. Only appropriate for
    targets, never for features.
    """
    panel["target_ret_21d"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(lambda x: x.shift(-FORWARD_WINDOW) / x - 1)
    )
    return panel

def add_target_dd5_21d(panel):
    """Add target_dd5_21d: binary flag for >5% drawdown in next 21 days.

    Value at date T is 1 if the adj_close at any point in [T+1, T+21]
    drops more than 5% below adj_close at T, else 0.

    NOTE: looks FORWARD. Only appropriate for targets, never for features.
    """
    def compute_dd_flag(prices):
        # For each T, get the min of prices[T+1 : T+21+1]
        # rolling(21).min() normally looks backward, so shift(-21) to pull
        # the forward-looking min into row T
        forward_min = (
            prices.shift(-1)
            .rolling(FORWARD_WINDOW)
            .min()
            .shift(-(FORWARD_WINDOW - 1))
        )
        # Drawdown from entry price
        forward_dd = forward_min / prices - 1
        # Binary flag, preserving NaN where forward window is incomplete
        flag = (forward_dd < -DRAWDOWN_THRESHOLD).astype(float)
        flag[forward_dd.isna()] = np.nan
        return flag

    panel["target_dd5_21d"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(compute_dd_flag)
    )
    return panel