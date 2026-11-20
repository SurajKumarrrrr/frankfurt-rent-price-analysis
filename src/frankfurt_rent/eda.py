"""Descriptive analysis tables used in the report."""

from __future__ import annotations

import pandas as pd

SIZE_BINS = [0, 40, 60, 80, 100, 130, 1000]
SIZE_LABELS = ["<40", "40-60", "60-80", "80-100", "100-130", "130+"]


def district_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median EUR/m2 and median cold rent per district, most expensive first."""
    g = df.groupby("district")
    out = pd.DataFrame(
        {
            "listings": g.size(),
            "median_eur_m2": g["eur_per_m2"].median(),
            "median_rent_eur": g["cold_rent_eur"].median(),
            "median_size_m2": g["size_m2"].median(),
        }
    )
    return out.sort_values("median_eur_m2", ascending=False).round(2)


def size_table(df: pd.DataFrame) -> pd.DataFrame:
    """Median EUR/m2 by flat-size bucket (shows the small-flat premium)."""
    bucket = pd.cut(df["size_m2"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    g = df.groupby(bucket, observed=True)
    return pd.DataFrame({"listings": g.size(), "median_eur_m2": g["eur_per_m2"].median()}).round(2)


def amenity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (unadjusted) median EUR/m2 with vs. without each amenity."""
    rows = []
    for col in ("furnished", "fitted_kitchen", "balcony", "elevator"):
        with_, without = df.loc[df[col], "eur_per_m2"].median(), df.loc[~df[col], "eur_per_m2"].median()
        rows.append(
            {
                "feature": col,
                "share_pct": round(100 * df[col].mean(), 1),
                "median_with": round(with_, 2),
                "median_without": round(without, 2),
                "raw_premium_pct": round(100 * (with_ / without - 1), 1),
            }
        )
    return pd.DataFrame(rows)


def overview(df: pd.DataFrame) -> dict[str, float]:
    return {
        "listings": int(len(df)),
        "median_rent_eur": float(df["cold_rent_eur"].median()),
        "median_size_m2": float(df["size_m2"].median()),
        "median_eur_m2": float(df["eur_per_m2"].median()),
        "p10_eur_m2": float(df["eur_per_m2"].quantile(0.1)),
        "p90_eur_m2": float(df["eur_per_m2"].quantile(0.9)),
    }
