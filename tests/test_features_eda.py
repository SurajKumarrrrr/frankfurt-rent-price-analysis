import numpy as np

from frankfurt_rent import eda
from frankfurt_rent.features import DISTRICT_NAMES, build_features


def test_features_numeric_and_complete(cleaned):
    x = build_features(cleaned)
    assert len(x) == len(cleaned)
    assert not x.isna().any().any()
    assert all(np.issubdtype(t, np.number) for t in x.dtypes)
    one_hot = x[[f"district_{d}" for d in DISTRICT_NAMES]]
    assert (one_hot.sum(axis=1) == 1).all()


def test_features_are_stateless(cleaned):
    """Same columns regardless of which rows are passed (no fold leakage / mismatch)."""
    assert list(build_features(cleaned.head(3)).columns) == list(build_features(cleaned).columns)


def test_unknown_energy_gets_flag(cleaned):
    df = cleaned.head(5).copy()
    df["energy_class"] = "unknown"
    x = build_features(df)
    assert (x["energy_unknown"] == 1).all()
    assert (x["energy_rank"] == 4).all()


def test_eda_tables(cleaned):
    d = eda.district_table(cleaned)
    assert d["median_eur_m2"].is_monotonic_decreasing
    assert d["listings"].sum() == len(cleaned)
    s = eda.size_table(cleaned)
    assert s["median_eur_m2"].iloc[0] > s["median_eur_m2"].iloc[-1]  # small-flat premium
    a = eda.amenity_table(cleaned).set_index("feature")
    assert a.loc["furnished", "raw_premium_pct"] > 0
    assert eda.overview(cleaned)["listings"] == len(cleaned)
