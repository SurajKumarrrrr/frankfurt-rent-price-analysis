"""Models, cross-validation and interpretation for cold-rent prediction.

All models predict log(cold rent); metrics are reported back on the euro scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from frankfurt_rent.features import RAW_FEATURES, build_features
from frankfurt_rent.generate import SEED, TRUE_EFFECTS


class DistrictMedianBaseline(BaseEstimator, RegressorMixin):
    """What a human would do first: district median EUR/m2 times flat size."""

    def fit(self, x: pd.DataFrame, y_log: np.ndarray) -> DistrictMedianBaseline:
        eur_m2 = np.exp(y_log) / x["size_m2"].to_numpy()
        s = pd.Series(eur_m2, index=x.index)
        self.district_median_ = s.groupby(x["district"]).median().to_dict()
        self.global_median_ = float(s.median())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        m2 = x["district"].map(self.district_median_).fillna(self.global_median_)
        return np.log(m2.to_numpy() * x["size_m2"].to_numpy())


def make_models(seed: int = SEED) -> dict[str, BaseEstimator]:
    feats = FunctionTransformer(build_features)
    return {
        "Baseline (district median EUR/m2)": DistrictMedianBaseline(),
        "Ridge (log-linear)": Pipeline([("features", feats), ("model", Ridge(alpha=0.1))]),
        "HistGradientBoosting": Pipeline(
            [
                ("features", feats),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                        l2_regularization=1.0, random_state=seed,
                    ),
                ),
            ]
        ),
    }


def euro_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y, p = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "MAE_eur": float(mean_absolute_error(y, p)),
        "MAPE_pct": float(100 * mean_absolute_percentage_error(y, p)),
        "R2": float(r2_score(y, p)),
    }


def target(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["cold_rent_eur"].to_numpy(dtype=float))


def cross_validate(df: pd.DataFrame, k: int = 5, seed: int = SEED) -> pd.DataFrame:
    """K-fold CV for every model; returns mean and std of each euro-scale metric."""
    x, y = df[RAW_FEATURES], target(df)
    rows = []
    for name, model in make_models(seed).items():
        for fold, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(x)):
            model.fit(x.iloc[tr], y[tr])
            rows.append({"model": name, "fold": fold, **euro_metrics(y[te], model.predict(x.iloc[te]))})
    res = pd.DataFrame(rows).drop(columns="fold").groupby("model", sort=False)
    out = res.mean().join(res.std(), rsuffix="_std")
    return out.round(3)


def ridge_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Ridge on all data and compare coefficients with the known true effects.

    Target is log(rent) and log_size is a feature, so the size elasticity of EUR/m2 is
    ``coef(log_size) - 1``. Other coefficients are log-premiums; ``exp(b) - 1`` is the %.
    """
    model = make_models()["Ridge (log-linear)"]
    model.fit(df[RAW_FEATURES], target(df))
    coefs = pd.Series(model.named_steps["model"].coef_, index=build_features(df.head(1)).columns)
    est = {k: float(coefs[k]) for k in TRUE_EFFECTS if k in coefs.index}
    est["log_size_elasticity"] = float(coefs["log_size"] - 1)
    est["energy_per_class"] = float(coefs["energy_rank"])
    out = pd.DataFrame({"true": pd.Series(TRUE_EFFECTS), "estimated": pd.Series(est)})
    out["abs_error"] = (out["estimated"] - out["true"]).abs()
    return out.round(4)


def holdout_importance(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED, repeats: int = 10
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Train HGB on a random split, return test metrics, permutation importance, predictions.

    Permutation importance is computed on the *raw* columns (the pipeline engineers
    features internally), so "district" is one importance, not 20 one-hot columns.
    """
    rng = np.random.default_rng(seed)
    test_mask = rng.random(len(df)) < test_size
    train, test = df[~test_mask], df[test_mask]
    model = make_models(seed)["HistGradientBoosting"]
    model.fit(train[RAW_FEATURES], target(train))
    pred_log = model.predict(test[RAW_FEATURES])
    metrics = euro_metrics(target(test), pred_log)
    pi = permutation_importance(
        model, test[RAW_FEATURES], target(test), n_repeats=repeats, random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    imp = (
        pd.DataFrame({"feature": RAW_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    preds = test.assign(predicted_rent_eur=np.exp(pred_log).round(0))
    return metrics, imp, preds


def flag_overpriced(preds: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
    """Listings whose asking rent exceeds the model's fair-rent estimate by > threshold."""
    gap = preds["cold_rent_eur"] / preds["predicted_rent_eur"] - 1
    return preds.assign(price_gap=gap.round(3))[gap > threshold].sort_values("price_gap", ascending=False)
