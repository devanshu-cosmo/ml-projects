import numpy as np
import pandas as pd

"""
Feature engineering and sanitization utilities.

Includes:
- FeatureSanitizer: selects stable numeric features
- FeatureEngineer: optional family-wise feature aggregation
"""

class FeatureSanitizer:
    """
    Selects a stable subset of numeric features.

    Drops:
    - explicitly excluded columns
    - constant features
    - near-zero features
    """
    def __init__(self, excluded_cols=None):
        self.excluded_cols = set(excluded_cols) if excluded_cols is not None else set()
        self.feature_names_ = None
    
    def fit(self, df):
        # Start from all numeric columns
        cols = [c for c in df.columns if c not in self.excluded_cols]
        numeric_cols = df[cols].select_dtypes(include=["float64", "int64"]).columns.tolist()

        non_constant = []
        for c in numeric_cols:
            series = df[c]
            if series.nunique(dropna=False) <= 1:
                continue
            # Drop columns that are almost always zero
            zero_frac = (series == 0).mean()
            if zero_frac > 0.98:
                continue
            non_constant.append(c)

        self.feature_names_ = non_constant
        return self
    
    def transform(self, df):
        assert self.feature_names_ is not None, "Call fit() first."
        # Any missing columns (e.g. new day missing a feature) will raise KeyError,
        return df[self.feature_names_].copy()
    
    def fit_transform(self, df):
        return self.fit(df).transform(df)


class FeatureEngineer:
    """
    Simple feature engineering using prefix-based families.
    
    Family = prefix before first '_', e.g.:
      'm0_T', 'm0_AI', ... -> family 'm0'
      'C_AU', 'C_BU', ...  -> family 'C'
      'F_H_B', 'F_H_S', .. -> family 'F'
    
    Parameters
    ----------
    add_family_agg : bool
        If True, add per-family aggregates (mean, std, min, max).
    keep_raw : bool
        If True, keep original raw features and add aggregates.
        If False, drop original features for families and keep only aggregates.
    excluded_cols : set or None
        Columns to exclude
    """
    def __init__(self, add_family_agg=True, keep_raw=True, excluded_cols=None):
        self.add_family_agg = add_family_agg
        self.keep_raw = keep_raw
        if excluded_cols is None:
            excluded_cols = {"ts_ns", "P1", "P2", "P3", "P4", "target_reg", "target_cls"}
        self.excluded_cols = excluded_cols
        self.family_groups_ = None
    
    def _get_prefix_family(self, col):
        if "_" in col:
            return col.split("_")[0]
        else:
            return col  # single token family (e.g. 'IC')
    
    def fit(self, X):
        if not self.add_family_agg:
            return self
        
        family_groups = {}
        for col in X.columns:
            if col in self.excluded_cols:
                continue
            fam = self._get_prefix_family(col)
            if fam not in family_groups:
                family_groups[fam] = []
            family_groups[fam].append(col)
        
        # Keep families with at least 2 members
        self.family_groups_ = {
            fam: cols for fam, cols in family_groups.items() if len(cols) >= 2
        }
        # print(f"FeatureEngineerSimple: Found {len(self.family_groups_)} families with 2+ members:")
        # for fam, cols in sorted(self.family_groups_.items()):
        #     print(f"  {fam}: {len(cols)} cols")
        return self
    
    def transform(self, X):
        if not self.add_family_agg or self.family_groups_ is None:
            return X.copy()
        
        X_out = X.copy()
        
        # Optionally drop raw features belonging to families
        if not self.keep_raw:
            cols_to_drop = []
            for fam, cols in self.family_groups_.items():
                cols_to_drop.extend(cols)
            cols_to_drop = [c for c in cols_to_drop if c in X_out.columns]
            X_out = X_out.drop(columns=cols_to_drop)
        
        # Add aggregates
        for fam, cols in self.family_groups_.items():
            fam_cols = [c for c in cols if c in X.columns]
            if len(fam_cols) == 0:
                continue
            X_out[f"{fam}_mean"] = X[fam_cols].mean(axis=1)
            # X_out[f"{fam}_range"] = X[fam_cols].max(axis=1) - X[fam_cols].min(axis=1)
            X_out[f"{fam}_std"] = X[fam_cols].std(axis=1)
            X_out[f"{fam}_min"] = X[fam_cols].min(axis=1)
            X_out[f"{fam}_max"] = X[fam_cols].max(axis=1)
        
        return X_out
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
