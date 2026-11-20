"""Deterministic synthetic generator for Frankfurt-style rental listings.

The data is SYNTHETIC. District price levels are rough, plausible orders of magnitude
for Frankfurt am Main cold rents, not measured market values. Because we write the
data-generating process ourselves, we know the true effect of every feature and can
check whether the models recover it (see ``TRUE_EFFECTS``).

The generator also injects realistic data-quality problems (re-posted duplicates,
missing values, unit typos) so the cleaning step has real work to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 42


@dataclass(frozen=True)
class District:
    name: str
    base_eur_m2: float  # cold rent per m2 for a 65 m2 "reference" flat
    km_to_center: float  # distance to Hauptwache
    share: float  # relative share of listings


DISTRICTS: tuple[District, ...] = (
    District("Westend-Süd", 21.5, 1.5, 0.05),
    District("Westend-Nord", 20.5, 2.3, 0.03),
    District("Innenstadt", 20.0, 0.3, 0.04),
    District("Nordend-West", 19.0, 1.8, 0.07),
    District("Nordend-Ost", 18.5, 2.0, 0.06),
    District("Sachsenhausen-Nord", 18.5, 1.6, 0.08),
    District("Ostend", 18.0, 2.2, 0.06),
    District("Bahnhofsviertel", 17.5, 1.0, 0.03),
    District("Bornheim", 17.5, 2.8, 0.07),
    District("Bockenheim", 17.5, 2.6, 0.08),
    District("Gallus", 17.0, 2.5, 0.08),
    District("Riedberg", 16.0, 8.5, 0.04),
    District("Niederrad", 15.5, 4.5, 0.05),
    District("Rödelheim", 14.5, 5.0, 0.04),
    District("Preungesheim", 14.0, 6.0, 0.03),
    District("Bergen-Enkheim", 14.0, 9.0, 0.03),
    District("Griesheim", 13.5, 6.5, 0.04),
    District("Fechenheim", 13.0, 6.8, 0.04),
    District("Sossenheim", 13.0, 8.0, 0.03),
    District("Höchst", 13.0, 10.0, 0.05),
)

ENERGY_CLASSES = ("A+", "A", "B", "C", "D", "E", "F", "G", "H")
CONDITIONS = ("needs_renovation", "well_kept", "renovated", "first_occupancy")

# Ground-truth effects on log(EUR per m2). Used for generation and for validating models.
TRUE_EFFECTS: dict[str, float] = {
    "log_size_elasticity": -0.12,  # bigger flats are cheaper per m2
    "furnished": 0.15,
    "fitted_kitchen": 0.04,
    "balcony": 0.03,
    "elevator": 0.02,
    "built_2015_plus": 0.10,
    "built_1950_1979": -0.05,
    "cond_needs_renovation": -0.10,
    "cond_renovated": 0.05,
    "cond_first_occupancy": 0.08,
    "energy_per_class": -0.01,  # per step from A+ to H
}
NOISE_SD = 0.08


def _year_built(rng: np.random.Generator, n: int) -> np.ndarray:
    era = rng.choice(5, size=n, p=[0.22, 0.30, 0.18, 0.15, 0.15])
    lows = np.array([1890, 1950, 1980, 2000, 2015])
    highs = np.array([1949, 1979, 1999, 2014, 2025])
    return rng.integers(lows[era], highs[era] + 1)


def generate_clean(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Generate ``n`` clean listings following the known data-generating process."""
    rng = np.random.default_rng(seed)
    shares = np.array([d.share for d in DISTRICTS])
    d_idx = rng.choice(len(DISTRICTS), size=n, p=shares / shares.sum())
    district = np.array([DISTRICTS[i].name for i in d_idx])
    base = np.array([DISTRICTS[i].base_eur_m2 for i in d_idx])
    km = np.array([DISTRICTS[i].km_to_center for i in d_idx]) + rng.normal(0, 0.4, n).clip(-0.3, 1)

    size = np.exp(rng.normal(np.log(68), 0.42, n)).clip(18, 220).round(1)
    rooms = np.clip(np.round(size / 28 + rng.normal(0.3, 0.5, n) * 2) / 2, 1, 7)
    year = _year_built(rng, n)
    floor = rng.integers(0, 9, n)
    new = year >= 2015
    elevator = rng.random(n) < np.where(new | (floor >= 4), 0.85, 0.30)
    balcony = rng.random(n) < np.where(year >= 1950, 0.70, 0.45)
    kitchen = rng.random(n) < 0.60
    furnished = rng.random(n) < 0.12
    cond_p = np.where(new[:, None], [0.0, 0.25, 0.05, 0.70], [0.12, 0.55, 0.30, 0.03])
    cond_idx = (rng.random(n)[:, None] > cond_p.cumsum(axis=1)).sum(axis=1).clip(0, 3)
    condition = np.array(CONDITIONS)[cond_idx]
    e_shift = np.select([year >= 2015, year >= 2000, year >= 1980], [0, 2, 3], 4)
    energy_idx = np.clip(e_shift + rng.integers(-1, 3, n), 0, len(ENERGY_CLASSES) - 1)
    energy = np.array(ENERGY_CLASSES)[energy_idx]

    t = TRUE_EFFECTS
    log_m2 = (
        np.log(base)
        + t["log_size_elasticity"] * (np.log(size) - np.log(65))
        + t["furnished"] * furnished
        + t["fitted_kitchen"] * kitchen
        + t["balcony"] * balcony
        + t["elevator"] * elevator
        + t["built_2015_plus"] * new
        + t["built_1950_1979"] * ((year >= 1950) & (year <= 1979))
        + t["cond_needs_renovation"] * (condition == "needs_renovation")
        + t["cond_renovated"] * (condition == "renovated")
        + t["cond_first_occupancy"] * (condition == "first_occupancy")
        + t["energy_per_class"] * energy_idx
        + rng.normal(0, NOISE_SD, n)
    )
    rent = np.round(np.exp(log_m2) * size, 0)

    return pd.DataFrame(
        {
            "listing_id": [f"FFM-{i:05d}" for i in range(n)],
            "district": district,
            "km_to_center": km.round(2),
            "size_m2": size,
            "rooms": rooms,
            "floor": floor,
            "year_built": year,
            "condition": condition,
            "energy_class": energy,
            "balcony": balcony,
            "elevator": elevator,
            "fitted_kitchen": kitchen,
            "furnished": furnished,
            "cold_rent_eur": rent,
        }
    )


def add_quality_issues(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Inject typical portal-scraping problems into a clean frame."""
    rng = np.random.default_rng(seed + 1)
    out = df.copy()
    out["year_built"] = out["year_built"].astype("Float64")
    n = len(out)

    def pick(frac: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(frac * n)), replace=False)

    out.loc[pick(0.04), "year_built"] = pd.NA
    out.loc[pick(0.03), "energy_class"] = None
    out.loc[pick(0.005), "size_m2"] *= 10  # "65" typed as "650"
    out.loc[pick(0.005), "cold_rent_eur"] = 1.0  # "price on request" placeholder
    out.loc[pick(0.003), "cold_rent_eur"] *= 100  # cents instead of euros

    dup_src = out.iloc[pick(0.02)].copy()
    dup_src["listing_id"] = [f"FFM-R{i:04d}" for i in range(len(dup_src))]  # re-posted ad
    out = pd.concat([out, dup_src], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_raw(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Clean listings plus injected quality issues, as a scraper would deliver them."""
    return add_quality_issues(generate_clean(n, seed), seed)
