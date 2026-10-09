"""Model factory and leave-one-out evaluation."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline

from .features import make_preprocessor

DISPLAY = {"linear_regression": "Linear Regression (baseline)",
           "random_forest": "Random Forest",
           "lightgbm": "LightGBM"}


def build_pipeline(name: str, cfg: dict) -> Pipeline:
    seed, params = cfg["project"]["seed"], cfg["modeling"]["models"][name] or {}
    if name == "linear_regression":
        est = LinearRegression(**params)
    elif name == "random_forest":
        est = RandomForestRegressor(random_state=seed, **params)
    elif name == "lightgbm":
        est = lgb.LGBMRegressor(random_state=seed, verbosity=-1, **params)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("pre", make_preprocessor(cfg)), ("model", est)])


def oof_predictions(pipe: Pipeline, X, y) -> np.ndarray:
    return cross_val_predict(clone(pipe), X, y, cv=LeaveOneOut())


def evaluate(name: str, pipe: Pipeline, X, y) -> dict:
    """LOOCV out-of-fold metrics plus in-sample R2 (to expose the overfit gap)."""
    oof = oof_predictions(pipe, X, y)
    train_r2 = r2_score(y, clone(pipe).fit(X, y).predict(X))
    loo_r2 = r2_score(y, oof)
    return {
        "model": DISPLAY[name], "key": name,
        "train_R2": train_r2, "LOOCV_R2": loo_r2, "gap": train_r2 - loo_r2,
        "LOOCV_MAE_log": mean_absolute_error(y, oof),
        "LOOCV_MAPE_count": mean_absolute_percentage_error(np.exp(y), np.exp(oof)),
    }


def leaderboard(cfg: dict, X, y) -> pd.DataFrame:
    rows = [evaluate(n, build_pipeline(n, cfg), X, y) for n in cfg["modeling"]["models"]]
    return pd.DataFrame(rows).sort_values("LOOCV_R2", ascending=False).reset_index(drop=True)
