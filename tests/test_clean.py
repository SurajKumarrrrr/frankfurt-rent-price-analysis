import pandas as pd

from frankfurt_rent.clean import EUR_M2_RANGE, SIZE_RANGE, clean


def test_clean_removes_issues(raw, cleaned):
    assert len(cleaned) < len(raw)
    assert cleaned["year_built"].notna().all()
    assert cleaned["energy_class"].notna().all()
    assert cleaned["size_m2"].between(*SIZE_RANGE).all()
    assert cleaned["eur_per_m2"].between(*EUR_M2_RANGE).all()


def test_report_adds_up(raw):
    df, rep = clean(raw)
    removed = sum(v for k, v in rep.steps.items() if k.startswith("Removed"))
    assert rep.rows_in - removed == rep.rows_out == len(df)
    assert "| Clean rows |" in rep.as_markdown()


def test_reposted_duplicate_keeps_original():
    base = {
        "district": "Gallus", "km_to_center": 2.5, "size_m2": 60.0, "rooms": 2.0, "floor": 1,
        "year_built": 1990.0, "condition": "well_kept", "energy_class": "C", "balcony": True,
        "elevator": False, "fitted_kitchen": True, "furnished": False, "cold_rent_eur": 1000.0,
    }
    raw = pd.DataFrame([{**base, "listing_id": "FFM-R0001"}, {**base, "listing_id": "FFM-00001"}])
    df, rep = clean(raw)
    assert df["listing_id"].tolist() == ["FFM-00001"]
    assert rep.steps["Removed re-posted duplicates"] == 1


def test_missing_year_imputed_with_district_median():
    rows = [
        {"listing_id": f"FFM-{i}", "district": "Ostend", "km_to_center": 2.0, "size_m2": 50.0 + i,
         "rooms": 2.0, "floor": i, "year_built": y, "condition": "well_kept", "energy_class": None,
         "balcony": False, "elevator": False, "fitted_kitchen": False, "furnished": False,
         "cold_rent_eur": 900.0}
        for i, y in enumerate([1960.0, 2000.0, None])
    ]
    df, _ = clean(pd.DataFrame(rows).astype({"year_built": "Float64"}))
    assert df.loc[df["listing_id"] == "FFM-2", "year_built"].item() == 1980
    assert (df["energy_class"] == "unknown").all()
