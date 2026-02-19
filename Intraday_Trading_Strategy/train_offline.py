"""
Offline training pipeline for intraday prediction model.

Trains the model, feature engineer, and sanitizer
using historical data only. This module is NOT
used at runtime during execution.
"""

import joblib
import numpy as np
import pandas as pd
import os

from model import ReturnRegressor
from features import FeatureSanitizer, FeatureEngineer
from config import H, THR

def add_forward_return_targets(
    df,
    price_col="P3",
    horizon=H,
    classification=True,
    threshold=THR
):
    """
    Add forward-looking targets on P3 with horizon >= 30 bars.
    
    Parameters
    ----------
    df : pd.DataFrame
        Single-day intraday data, already sorted by ts_ns and indexed 0..n-1.
    price_col : str
        Name of the tradable price column, here 'P3'.
    horizon : int
        Lookahead in bars. Must be >= 30.
    classification : bool
        If True, also create a sign-based classification target.
    threshold : float
        Dead-zone threshold on log-return for classification, e.g. 0.0 or 1e-4.
        
    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with new columns:
          - 'target_reg': future log-return at horizon H
          - 'target_cls' (optional): sign-based label in {-1, 0, +1}
        Last `horizon` rows are dropped (no future price).
    """
    assert horizon >= 30, "Horizon must be at least 30 bars to satisfy spec."
    df = df.copy()
    
    # Ensure no zero or negative prices (should not happen, but be safe)
    if (df[price_col] <= 0).any():
        raise ValueError(f"Non-positive values found in {price_col}.")
    
    # Future price at t+H (shifted backward so that row t sees t+H)
    future_price = df[price_col].shift(-horizon)
    
    # Log-return from t to t+H
    target_reg = np.log(future_price) - np.log(df[price_col])
    df["target_reg"] = target_reg
    
    if classification:
        # Sign target with optional dead-zone
        # y > threshold -> +1, y < -threshold -> -1, else 0
        y = target_reg.values
        labels = np.zeros_like(y, dtype=int)
        labels[y > threshold] = 1
        labels[y < -threshold] = -1
        df["target_cls"] = labels
    
    # Drop rows where future price is NaN (last `horizon` rows)
    df_out = df.iloc[:-horizon].reset_index(drop=True)
    return df_out
    

def build_per_day_data(
    day_ids,
    data_dir,
    horizon=H,
    threshold=THR,
    feature_engineer=None  # optional FeatureEngineer instance
):
    """
    For each day, build:
      - raw DataFrame with targets
      - feature DataFrame (without targets)
      - target Series (regression)
    
    Returns
    -------
    day_to_raw : dict[day_id -> pd.DataFrame]
        Each DataFrame includes target_reg and target_cls.
    day_to_features : dict[day_id -> pd.DataFrame]
        Features only (no targets), unsanitized.
    day_to_target_reg : dict[day_id -> pd.Series]
        Regression target per day.
    """
    day_to_raw = {}
    day_to_features = {}
    day_to_target_reg = {}
    
    for d in day_ids:
        path = os.path.join(data_dir, f"{d}.csv")
        df_day = pd.read_csv(path)
        df_day = df_day.sort_values("ts_ns").reset_index(drop=True)
        df_day_t = add_forward_return_targets(
            df_day,
            price_col="P3",
            horizon=horizon,
            classification=True,
            threshold=threshold
        )

        
        
        # Separate features / target
        y_reg = df_day_t["target_reg"].copy()
        feat_df = df_day_t.drop(columns=["target_reg", "target_cls"])

        # Apply feature engineering if provided
        if feature_engineer is not None:
            feat_df = feature_engineer.transform(feat_df)
        
        day_to_raw[d] = df_day_t
        day_to_features[d] = feat_df
        day_to_target_reg[d] = y_reg
    
    return day_to_raw, day_to_features, day_to_target_reg




def train_offline_pipeline(
    day_ids,
    day_to_features,
    day_to_target_reg,
    excluded_cols,
    training_mode="rolling",
    rolling_window_size=None,
    add_family_agg=True,
    base_model=None
):
    """
    Train model and preprocessing pipeline offline.

    Parameters
    ----------
    day_ids : list
        Ordered list of available training day IDs.
    day_to_features : dict
        day_id -> feature DataFrame
    day_to_target_reg : dict
        day_id -> target Series
    training_mode : str
        "rolling" or "expanding"
    rolling_window_size : int or None
        Window size if using rolling training
    """

    # Select training days

    if training_mode == "expanding":
        train_days = list(day_ids)

    elif training_mode == "rolling":
        if rolling_window_size is None:
            raise ValueError("rolling_window_size must be set for rolling mode")
        train_days = list(day_ids[-rolling_window_size:])

    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")

    if len(train_days) == 0:
        raise ValueError("No training days selected")


    # Build training matrices

    X_list, y_list = [], []

    for d in train_days:
        X_list.append(day_to_features[d])
        y_list.append(day_to_target_reg[d])

    X_train = pd.concat(X_list, axis=0)
    y_train = pd.concat(y_list, axis=0)

    # Feature engineering

    if add_family_agg:
        feature_engineer = FeatureEngineer(add_family_agg=True, keep_raw=True)
        X_train = feature_engineer.fit_transform(X_train)
    else:
        feature_engineer = None

    # Feature sanitization

    sanitizer = FeatureSanitizer(excluded_cols=excluded_cols)
    X_train_sanitized = sanitizer.fit_transform(X_train)

    # Model training
    
    if base_model is None:
        model = ReturnRegressor()
    else:
        model = ReturnRegressor(base_model=base_model)

    model.fit(X_train_sanitized, y_train)

    return model, sanitizer, feature_engineer


#def train_offline_from_config(
    #day_ids,
    #day_to_features,
    #day_to_target_reg,
    #excluded_cols
#):
    #return train_offline_pipeline(
        #day_ids=day_ids,
        #day_to_features=day_to_features,
        #day_to_target_reg=day_to_target_reg,
        #excluded_cols=excluded_cols,
        #training_mode=TRAINING_MODE,
        #rolling_window_size=ROLLING_WINDOW_SIZE,
        #add_family_agg=ADD_FAMILY_AGG,
        #base_model=None
    #)


## OFFLINE TRAINING

#model, sanitizer, feature_engineer = train_offline_from_config(
    #day_ids=day_ids_used,
    #day_to_features=day_to_features,
    #day_to_target_reg=day_to_target_reg,
    #excluded_cols=EXCLUDED_COLUMNS,
#)

## SINGLE-DAY EXECUTION TEST


#for day in test_day:
    #trades_df = run_execution_for_day(
        #df_day_raw=day_to_raw[day],#day_to_raw[test_day],
        #sanitizer=sanitizer,
        #feature_engineer=feature_engineer,
        #model=model,
        #price_col="P3",
        #threshold=THR,
        #tc_rate=TC_RATE,
        #use_min_hold=True,
        #use_flip_confirm=True,
        #force_flat_before_flip=True
    #)
    
    #print("Final PnL:", trades_df["cum_pnl"].iloc[-1], "for day",day)


def save_trained_pipeline(path, model, sanitizer, feature_engineer):
    """
    Save trained pipeline components to disk.
    """
    joblib.dump(
        {
            "model": model,
            "sanitizer": sanitizer,
            "feature_engineer": feature_engineer,
        },
        path,
    )

def load_trained_pipeline(path):
    obj = joblib.load(path)
    return obj["model"], obj["sanitizer"], obj["feature_engineer"]
