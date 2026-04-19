import pandas as pd


def add_return_features(panel):
    """Add trailing return features: ret_21d, ret_63d, ret_252d.

    Each value at date T is the return earned over the N trading days
    ending at T, computed per-ticker from adj_close.
    """
    for window in [21, 63, 252]:
        col_name = f"ret_{window}d"
        panel[col_name] = (
            panel.groupby("ticker")["adj_close"]
            .transform(lambda x: x.pct_change(window))
        )
    return panel

def add_trend_features(panel):
    """Add price-to-SMA trend feature: price_to_sma50.

    Value at date T is adj_close[T] divided by the 50-day simple moving
    average of adj_close ending at T. Values > 1 mean price is above trend
    (bullish); values < 1 mean below trend (bearish).
    """
    panel["price_to_sma50"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(lambda x: x / x.rolling(50).mean())
    )
    return panel

def add_volatility_features(panel):
    """Add volatility and risk features: vol_20d, range_pct_20d, max_dd_90d.

    vol_20d: annualized 20-day realized volatility from close-to-close returns
    range_pct_20d: average 20-day (high - low) / close, captures intraday range
    max_dd_90d: largest peak-to-trough drawdown over the trailing 90 days
    """
    # Daily returns (intermediate, not a feature itself)
    daily_ret = (
        panel.groupby("ticker")["adj_close"]
        .transform(lambda x: x.pct_change())
    )

    # vol_20d: 20-day rolling std of daily returns, annualized by sqrt(252)
    panel["vol_20d"] = (
        daily_ret.groupby(panel["ticker"])
        .transform(lambda x: x.rolling(20).std() * (252 ** 0.5))
    )

    # range_pct_20d: 20-day average of (high - low) / close
    daily_range = (panel["high"] - panel["low"]) / panel["close"]
    panel["range_pct_20d"] = (
        daily_range.groupby(panel["ticker"])
        .transform(lambda x: x.rolling(20).mean())
    )

    # max_dd_90d: rolling 90-day maximum drawdown from peak
    panel["max_dd_90d"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(lambda x: (x / x.rolling(90).max()) - 1)
    )

    return panel

def add_market_features(panel):
    """Add market-relative features: excess_ret_21d, beta_60d.

    Requires the panel to already contain a 'spy_ret_1d' column with SPY's
    daily returns aligned to each row's date. The orchestrator attaches this
    before calling this function.

    excess_ret_21d: ticker's 21-day return minus SPY's 21-day return
    beta_60d: slope from 60-day rolling regression of stock daily returns
              on SPY daily returns (= rolling cov / rolling var)
    """
    if "spy_ret_1d" not in panel.columns:
        raise ValueError(
            "add_market_features requires 'spy_ret_1d' column. "
            "Attach SPY returns in the orchestrator before calling this."
        )

    # SPY's 21-day compounded return, aligned to each row's date
    spy_ret_21d = (
        (1 + panel["spy_ret_1d"]).groupby(panel["ticker"])
        .transform(lambda x: x.rolling(21).apply(lambda w: w.prod(), raw=True) - 1)
    )

    # excess_ret_21d requires ret_21d to already exist
    if "ret_21d" not in panel.columns:
        raise ValueError(
            "add_market_features requires 'ret_21d'. "
            "Call add_return_features before add_market_features."
        )
    panel["excess_ret_21d"] = panel["ret_21d"] - spy_ret_21d

    # Ticker daily returns (intermediate)
    daily_ret = (
        panel.groupby("ticker")["adj_close"]
        .transform(lambda x: x.pct_change())
    )

    # beta_60d = rolling cov(stock, spy) / rolling var(spy), per ticker
    def rolling_beta(group):
        stock = daily_ret.loc[group.index]
        spy = group
        cov = stock.rolling(60).cov(spy)
        var = spy.rolling(60).var()
        return cov / var

    panel["beta_60d"] = (
        panel["spy_ret_1d"].groupby(panel["ticker"]).transform(rolling_beta)
    )

    return panel

def add_rsi_14(panel, window=14):
    """Add rsi_14: 14-day Relative Strength Index, per ticker.

    RSI is an oscillator between 0 and 100 measuring the ratio of average
    gains to average losses over the lookback window. Conventional reads:
    RSI > 70 = overbought, RSI < 30 = oversold.
    """
    def compute_rsi(prices):
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(window).mean()
        avg_loss = losses.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    panel["rsi_14"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(compute_rsi)
    )
    return panel

def add_bb_position_20(panel, window=20):
    """Add bb_position_20: normalized position within 20-day Bollinger Bands.

    0 = at lower band (price - 2 std below mean)
    0.5 = at the middle band (20-day mean)
    1 = at upper band (price + 2 std above mean)
    Values can go outside [0, 1] on extreme moves.
    """
    def compute_bb_position(prices):
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        lower = mean - 2 * std
        upper = mean + 2 * std
        return (prices - lower) / (upper - lower)

    panel["bb_position_20"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(compute_bb_position)
    )
    return panel

def add_macd_hist(panel, fast=12, slow=26, signal=9):
    """Add macd_hist: MACD histogram (MACD line - signal line).

    MACD line = EMA(fast) - EMA(slow). Signal line = EMA of MACD line.
    Histogram measures momentum of the momentum — positive & rising means
    the trend is accelerating, negative & falling means it's weakening.
    """
    def compute_macd_hist(prices):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    panel["macd_hist"] = (
        panel.groupby("ticker")["adj_close"]
        .transform(compute_macd_hist)
    )
    return panel

def add_atr_14_pct(panel, window=14):
    """Add atr_14_pct: 14-day Average True Range as a percent of close.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR averages TR over the window. Dividing by close makes it comparable
    across stocks at different price levels.
    """
    def compute_atr_pct(group):
        high = group["high"]
        low = group["low"]
        close = group["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window).mean()
        return atr / close

    panel["atr_14_pct"] = (
        panel.groupby("ticker", group_keys=False)
        .apply(compute_atr_pct, include_groups=False)
    )
    return panel

def add_volume_ratio_20d(panel, window=20):
    """Add volume_ratio_20d: today's volume divided by 20-day average volume.

    Values > 1 indicate above-average trading activity. Useful signal for
    detecting when news/events are driving participation.
    """
    panel["volume_ratio_20d"] = (
        panel.groupby("ticker")["volume"]
        .transform(lambda x: x / x.rolling(window).mean())
    )
    return panel

def add_rank_features(panel, feature_cols):
    """Add per-date cross-sectional rank-transformed versions of features.

    For each date, each feature value is replaced by its rank across all
    tickers on that date, normalized to (0, 1] via pandas' pct=True option (equivalent to rank/N).
    The lowest-ranked ticker on each date gets 1/N, the highest gets 1.0.
    This makes features regime-invariant: a rank of 0.9 means "top 10% of today's
    universe" regardless of whether today is a calm market or a crash.

    Ties are handled with method='average' (standard quant convention).
    NaN values stay NaN — ranking only happens among non-NaN values on
    each date. Rows with missing features will be filtered by the
    downstream notna() check in build_features.py.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'date' and all columns named in feature_cols.
    feature_cols : list of str
        Feature column names to rank-transform. Does NOT include sector
        one-hots (those are binary and shouldn't be ranked) or target
        columns (those get a separate rank transform in targets.py).

    Returns
    -------
    panel : pd.DataFrame
        Same DataFrame with new columns '{feature}_rank' added, one per
        input column.
    """
    for feat in feature_cols:
        if feat not in panel.columns:
            raise ValueError(f"Feature '{feat}' not in panel; can't rank-transform.")

        panel[f"{feat}_rank"] = (
            panel.groupby("date")[feat]
            .transform(lambda x: x.rank(method="average", pct=True))
        )

    return panel

def add_sector_excess_ret_21d(panel):
    """Add sector_excess_ret_21d: ticker's 21-day return minus that day's
    sector-average 21-day return.

    This captures "is this stock outperforming its sector peers" rather
    than just "is this stock going up." Distinct from excess_ret_21d
    (which uses SPY as the benchmark) — this uses same-sector peers,
    letting the model learn within-sector stock selection signal.

    The sector average on date T is computed cross-sectionally using
    only same-date values of ret_21d — no future leakage.

    Requires ret_21d and sector columns to already exist on the panel.
    Call AFTER add_return_features and AFTER attach_sector.
    """
    if "ret_21d" not in panel.columns:
        raise ValueError(
            "add_sector_excess_ret_21d requires 'ret_21d'. "
            "Call add_return_features first."
        )
    if "sector" not in panel.columns:
        raise ValueError(
            "add_sector_excess_ret_21d requires 'sector' column. "
            "Call attach_sector first."
        )

    # Compute the sector-average ret_21d for each (date, sector) pair,
    # then broadcast back to every row that shares that (date, sector).
    sector_avg = (
        panel.groupby(["date", "sector"])["ret_21d"]
        .transform("mean")
    )

    panel["sector_excess_ret_21d"] = panel["ret_21d"] - sector_avg

    return panel

def add_target_ret_21d_rank(panel):
    """Add target_ret_21d_rank: per-date cross-sectional rank of target_ret_21d.

    This is the same metric our evaluation uses (Spearman rank IC), directly
    baked into a training target. Training against this aligns the loss
    function (MSE) with the evaluation metric (rank correlation) — addresses
    the loss/metric mismatch ChatGPT flagged in v1.

    Ranks are normalized to [0, 1] via method='average' with pct=True,
    matching the rank transform used for features. NaN inputs stay NaN
    (last 21 rows per ticker, where forward returns can't be computed).

    Requires target_ret_21d to already exist. Call AFTER add_target_ret_21d.
    """
    if "target_ret_21d" not in panel.columns:
        raise ValueError(
            "add_target_ret_21d_rank requires 'target_ret_21d'. "
            "Call add_target_ret_21d first."
        )

    panel["target_ret_21d_rank"] = (
        panel.groupby("date")["target_ret_21d"]
        .transform(lambda x: x.rank(method="average", pct=True))
    )

    return panel