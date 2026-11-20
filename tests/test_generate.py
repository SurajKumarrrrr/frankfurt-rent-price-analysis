import pandas as pd

from frankfurt_rent.generate import DISTRICTS, generate_clean, generate_raw


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_raw(500, seed=1), generate_raw(500, seed=1))


def test_different_seeds_differ():
    assert not generate_clean(200, seed=1).equals(generate_clean(200, seed=2))


def test_clean_data_is_plausible():
    df = generate_clean(2000)
    eur_m2 = df["cold_rent_eur"] / df["size_m2"]
    assert eur_m2.between(5, 60).all()
    assert df["size_m2"].between(18, 220).all()
    assert set(df["district"]) <= {d.name for d in DISTRICTS}
    assert df["listing_id"].is_unique


def test_district_ordering_follows_base_price():
    df = generate_clean(6000)
    med = (df["cold_rent_eur"] / df["size_m2"]).groupby(df["district"]).median()
    assert med["Westend-Süd"] > med["Gallus"] > med["Höchst"]


def test_raw_contains_quality_issues():
    raw = generate_raw(2000)
    assert len(raw) > 2000  # duplicates appended
    assert raw["year_built"].isna().any()
    assert raw["energy_class"].isna().any()
    assert (raw["cold_rent_eur"] == 1.0).any()
