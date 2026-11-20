"""Stateless feature engineering: clean listings -> numeric model matrix.

Kept stateless (fixed category lists, fixed imputation values) so it can sit inside a
scikit-learn Pipeline without leaking information across CV folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frankfurt_rent.generate import DISTRICTS, ENERGY_CLASSES

RAW_FEATURES = [
    "district", "km_to_center", "size_m2", "rooms", "floor", "year_built", "condition",
    "energy_class", "balcony", "elevator", "fitted_kitchen", "furnished",
]
DISTRICT_NAMES = [d.name for d in DISTRICTS]
# "well_kept" is the reference level, so the other coefficients read as premiums vs it.
CONDITION_DUMMIES = ["needs_renovation", "renovated", "first_occupancy"]
ENERGY_RANK = {c: i for i, c in enumerate(ENERGY_CLASSES)}
ENERGY_FALLBACK_RANK = 4  # class "D", used for unknown labels


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric feature frame with explicit, human-readable column names."""
    x = pd.DataFrame(index=df.index)
    x["log_size"] = np.log(df["size_m2"].astype(float))
    x["rooms"] = df["rooms"].astype(float)
    x["floor"] = df["floor"].astype(float)
    x["km_to_center"] = df["km_to_center"].astype(float)
    year = df["year_built"].astype(float)
    x["built_2015_plus"] = (year >= 2015).astype(float)
    x["built_1950_1979"] = ((year >= 1950) & (year <= 1979)).astype(float)
    for b in ("balcony", "elevator", "fitted_kitchen", "furnished"):
        x[b] = df[b].astype(float)
    for c in CONDITION_DUMMIES:
        x[f"cond_{c}"] = (df["condition"] == c).astype(float)
    rank = df["energy_class"].map(ENERGY_RANK)
    x["energy_unknown"] = rank.isna().astype(float)
    x["energy_rank"] = rank.fillna(ENERGY_FALLBACK_RANK).astype(float)
    for name in DISTRICT_NAMES:
        x[f"district_{name}"] = (df["district"] == name).astype(float)
    return x
