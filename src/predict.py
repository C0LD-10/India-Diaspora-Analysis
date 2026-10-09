"""Load the saved pipeline and estimate diaspora size."""
import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve


def load_model(cfg: dict = None):
    cfg = cfg or load_config()
    return joblib.load(resolve(cfg, "best_model"))


def estimate(model, country: str, year: int, unemployment: float) -> float:
    """Estimated Indian-born population (count) for a country-year.

    A pooled-trend descriptive estimate, NOT a forecast with calibrated
    uncertainty. Extrapolating beyond the observed 2000-2024 range is unreliable.
    """
    X = pd.DataFrame({"Destination_Country": [country], "Year": [year],
                      "Unemployment_India": [unemployment]})
    return float(np.exp(model.predict(X)[0]))
