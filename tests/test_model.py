import numpy as np
import pandas as pd

from frankfurt_rent import model
from frankfurt_rent.features import RAW_FEATURES


def test_baseline_predicts_district_median():
    x = pd.DataFrame({"district": ["A", "A", "B"], "size_m2": [50.0, 100.0, 40.0]})
    y = np.log(np.array([1000.0, 2000.0, 400.0]))  # A: 20 EUR/m2, B: 10 EUR/m2
    b = model.DistrictMedianBaseline().fit(x, y)
    test = pd.DataFrame({"district": ["A", "B", "C"], "size_m2": [10.0, 10.0, 10.0]})
    np.testing.assert_allclose(np.exp(b.predict(test)), [200.0, 100.0, 200.0])


def test_euro_metrics_perfect():
    y = np.log(np.array([500.0, 1000.0, 1500.0]))
    m = model.euro_metrics(y, y)
    assert m["MAE_eur"] == 0 and m["MAPE_pct"] == 0 and m["R2"] == 1


def test_models_beat_baseline(cleaned):
    cv = model.cross_validate(cleaned, k=3)
    base = cv.loc["Baseline (district median EUR/m2)", "MAE_eur"]
    assert cv.loc["Ridge (log-linear)", "MAE_eur"] < base
    assert cv.loc["HistGradientBoosting", "MAE_eur"] < base
    assert (cv["R2"] > 0.8).all()


def test_ridge_recovers_true_effects(cleaned):
    eff = model.ridge_effects(cleaned)
    assert eff["estimated"].notna().all()
    assert (eff["abs_error"] < 0.03).all()
    assert np.sign(eff["estimated"]).equals(np.sign(eff["true"]))


def test_holdout_and_overpriced(cleaned):
    metrics, imp, preds = model.holdout_importance(cleaned, repeats=2)
    assert metrics["R2"] > 0.85
    assert set(imp["feature"]) == set(RAW_FEATURES)
    assert imp.iloc[0]["feature"] == "size_m2"
    flagged = model.flag_overpriced(preds, threshold=0.15)
    assert (flagged["price_gap"] > 0.15).all()
    assert len(flagged) < 0.15 * len(preds)
