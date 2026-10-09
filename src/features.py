"""Modeling frame and preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def feature_columns(cfg: dict) -> list:
    m = cfg["modeling"]
    return m["categorical_features"] + m["numeric_features"]


def build_modeling_frame(wide: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only rows where the target is genuinely observed (no target imputation)."""
    m = cfg["modeling"]
    df = wide.dropna(subset=[m["target_raw"]]).copy()
    df[m["target"]] = np.log(df[m["target_raw"]])
    return df.dropna(subset=feature_columns(cfg) + [m["target"]]).reset_index(drop=True)


def make_preprocessor(cfg: dict) -> ColumnTransformer:
    m = cfg["modeling"]
    return ColumnTransformer(
        [("country_ohe", OneHotEncoder(handle_unknown="ignore"), m["categorical_features"])],
        remainder="passthrough",   # numeric features pass through unchanged
    )
