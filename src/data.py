"""Raw CSV (long format, mixed units) -> analysis-ready wide panel."""
import numpy as np
import pandas as pd


def parse_value(v, missing_token: str = "NOT AVAILABLE") -> float:
    """Parse the mixed-format Value column.

    Handles thousands-separated ints, percentages, and money strings in
    billions ($x.xB) or trillions ($x.xT). Money is normalised to USD billions.
    """
    if pd.isna(v) or v == missing_token:
        return np.nan
    v = str(v).strip().replace(",", "")
    if v.startswith("$") and v.endswith("B"):
        return float(v[1:-1])
    if v.startswith("$") and v.endswith("T"):
        return float(v[1:-1]) * 1000.0
    return float(v)


def load_raw(path, missing_token: str = "NOT AVAILABLE") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["Value_num"] = raw["Value"].apply(parse_value, missing_token=missing_token)
    return raw


def to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide via explicit merges.

    pivot_table silently dropped (year, country) rows on this data (its dropna
    handling discards all-NaN rows); explicit merges are unambiguous. The
    assertion guards against row loss.
    """
    base = raw[["Year", "Destination_Country", "Region"]].drop_duplicates().reset_index(drop=True)
    for col in raw["Column_Name"].unique():
        sub = (raw.loc[raw["Column_Name"] == col, ["Year", "Destination_Country", "Value_num"]]
                  .rename(columns={"Value_num": col}))
        base = base.merge(sub, on=["Year", "Destination_Country"], how="left")
    wide = base.sort_values(["Destination_Country", "Year"]).reset_index(drop=True)
    expected = raw[["Year", "Destination_Country"]].drop_duplicates().shape[0]
    assert len(wide) == expected, "Reshape lost or duplicated (year, country) rows"
    return wide


def quality_scorecard(wide: pd.DataFrame, national_cols: list) -> pd.DataFrame:
    rows = []
    for c in wide.columns:
        if c in ("Year", "Destination_Country", "Region"):
            continue
        rows.append({
            "column": c,
            "granularity": "national (India-level)" if c in national_cols else "destination-specific",
            "completeness": wide[c].notna().mean(),
            "n_unique_values": wide[c].nunique(dropna=True),
        })
    return pd.DataFrame(rows).set_index("column")
