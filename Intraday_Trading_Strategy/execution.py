import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from config import H, THR, TC_RATE

if TYPE_CHECKING:
    from features import FeatureSanitizer, FeatureEngineer
    from model import ReturnRegressor
    
"""
Execution engine for intraday trading strategy.

This module converts model predictions into trading signals
and simulates causal, bar-by-bar execution with full PnL
accounting and transaction costs.
"""

def prediction_to_signal_persistent(
    pred,
    pred_std,
    prev_position,
    entry_z=1.0,
    exit_z=0.3
):
    
    """
    Convert a regression prediction into a persistent directional signal.

    Parameters
    ----------
    pred : float
        Current model prediction (e.g. forward log-return).
    pred_std : float or None
        Rolling standard deviation of past predictions.
    prev_position : int
        Previous directional position (-1, 0, +1).
    entry_z : float
        Z-score threshold for entering a position.
    exit_z : float
        Z-score threshold for exiting an existing position.

    Returns
    -------
    int
        Desired directional position: -1, 0, or +1.
    """
    # No trading if std undefined
    if pred_std is None or pred_std <= 0 or np.isnan(pred_std):
        return 0

    z = pred / pred_std

    # Flat -> enter only if strong signla
    if prev_position == 0:
        if z > entry_z:
            return 1
        elif z < -entry_z:
            return -1
        else:
            return 0

    # Long position logic
    if prev_position == 1:
        if z < -entry_z:
            return -1   # strong flip
        elif z < exit_z:
            return 0    # exit
        else:
            return 1    # hold

    # Short position logic
    if prev_position == -1:
        if z > entry_z:
            return 1
        elif z > -exit_z:
            return 0
        else:
            return -1
    
    # Final fallback for safety 
    return 0



def run_execution_for_day(
    df_day_raw,
    sanitizer,
    feature_engineer,  # can be None
    model,
    price_col="P3",
    threshold=THR,
    tc_rate=TC_RATE,
    # --- churn controls ---
    use_min_hold=True,
    min_hold_bars=30,
    use_flip_confirm=True,
    confirm_bars=2,
    force_flat_before_flip=True,
    use_horizon_decisions=True,
    decision_step=30,  
    ROLLING_PRED_WINDOW=200,
    entry_z=1.4,
    exit_z=0.3,
    use_risk_targeting=True,
    vol_window=200,
    target_risk=1.5,
    max_pos=1.1,
    min_vol=1e-12
):

    """
    Run a causal, intraday execution simulation for a single trading day.

    Parameters
    ----------
    df_day_raw : pd.DataFrame
        Single-day intraday data sorted by timestamp.
    sanitizer : FeatureSanitizer
        Fitted feature sanitizer from offline training.
    feature_engineer : FeatureEngineer or None
        Optional fitted feature engineer.
    model : fitted regression model
        Predicts forward returns.
    price_col : str
        Name of the tradable price column.
    tc_rate : float
        Transaction cost rate (fraction of notional).

    Returns
    -------
    pd.DataFrame
        Per-bar trade log with PnL accounting.
    """

    df = df_day_raw.copy()
    df = df.sort_values("ts_ns").reset_index(drop=True)

    # Drop target columns if present
    drop_cols = [c for c in ["target_reg", "target_cls"] if c in df.columns]
    feat_df = df.drop(columns=drop_cols)

    # Apply feature engineering
    if feature_engineer is not None:
        feat_df = feature_engineer.transform(feat_df)

    X_day = sanitizer.transform(feat_df)

    prices = df[price_col].values
    rets = np.diff(prices, prepend=prices[0])
    vol = pd.Series(rets).rolling(vol_window, min_periods=vol_window).std().values

    n = len(df)

    # Execution State
    direction = 0        # discrete directional state {-1,0,1}
    position = 0.0       # sized position only
    prev_price = prices[0]
    entry_price = None

    cum_pnl = 0.0
    cum_tc = 0.0
    realized_pnl = 0.0

    records = []

    bars_in_pos = 0
    pending_flip_dir = 0
    pending_flip_count = 0

    pred_history = []

    for t in range(n):
        price_t = prices[t]

        x_t = X_day.iloc[t:t+1]

        # generate signal
        if use_horizon_decisions and (t % decision_step != 0):
            signal_t = direction   
        else:
            pred_t = model.predict(x_t)[0]
            pred_history.append(pred_t)

            if len(pred_history) >= 10:
                pred_std_t = np.std(pred_history[-ROLLING_PRED_WINDOW:])
            else:
                pred_std_t = None

            signal_t = prediction_to_signal_persistent(
                pred=pred_t,
                pred_std=pred_std_t,
                prev_position=direction,  
                entry_z=entry_z,
                exit_z=exit_z
            )

        desired = signal_t

        # track holding time
        if direction == 0:
            bars_in_pos = 0
        else:
            bars_in_pos += 1

        # min hold
        if use_min_hold and direction != 0 and bars_in_pos < min_hold_bars:
            desired = direction  

        # confirm flip 
        if use_flip_confirm and direction != 0:
            if desired == -direction:  
                if pending_flip_dir != desired:
                    pending_flip_dir = desired
                    pending_flip_count = 1
                else:
                    pending_flip_count += 1

                if pending_flip_count < confirm_bars:
                    desired = direction
            else:
                pending_flip_dir = 0
                pending_flip_count = 0

        # flat before flip
        if force_flat_before_flip and direction != 0 and desired == -direction:
            desired = 0

        # update direction BEFORE sizing
        direction = int(desired)
        assert direction in (-1, 0, 1)

        # Positin sizing
        if use_risk_targeting:
            vol_t = vol[t]
            if np.isnan(vol_t) or vol_t <= min_vol:
                new_position = 0.0
            else:
                scale = min(max_pos, target_risk / vol_t)
                new_position = direction * scale
        else:
            new_position = float(direction)

        # trade & costs
        trade = new_position - position
        traded_notional = abs(trade) * price_t
        tx_cost = traded_notional * tc_rate

        pnl_realized_t = 0.0
        if trade != 0 and position != 0:
            pnl_realized_t = (price_t - entry_price) * position
            realized_pnl += pnl_realized_t

        entry_price = price_t if new_position != 0 else None
        cum_tc += tx_cost

        mtm_pnl_t = (price_t - prev_price) * position
        cum_pnl += pnl_realized_t + mtm_pnl_t - tx_cost

        
        records.append({
            "ts_ns": df.loc[t, "ts_ns"],
            price_col: price_t,
            "entry_price": entry_price,
            # "pred": pred_t if 'pred_t' in locals() else np.nan,
            # "signal": signal_t,
            "position": new_position,
            "trade": trade,
            # "trade_price": price_t if trade != 0 else np.nan,
            "pnl_realized_t": pnl_realized_t,
            "pnl_mtm_t": mtm_pnl_t,
            "tx_cost_t": tx_cost,
            "cum_pnl": cum_pnl,
            "cum_realized_pnl": realized_pnl,
            "cum_tx_cost": cum_tc,
        })

        position = new_position
        prev_price = price_t

    trades_df = pd.DataFrame(records)
    trades_df["trade_count_cum"] = (trades_df["trade"] != 0).cumsum()

    return trades_df
