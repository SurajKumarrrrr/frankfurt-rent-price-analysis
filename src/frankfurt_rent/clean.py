"""Cleaning rules for raw listings, with an audit log of what was removed and why."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

FEATURE_KEY = [
    "district", "size_m2", "rooms", "floor", "year_built", "condition",
    "balcony", "elevator", "fitted_kitchen", "furnished", "cold_rent_eur",
]
SIZE_RANGE = (12.0, 300.0)
EUR_M2_RANGE = (5.0, 60.0)


@dataclass
class CleaningReport:
    rows_in: int = 0
    steps: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0

    def as_markdown(self) -> str:
        lines = ["| Step | Rows affected |", "|---|---:|", f"| Raw rows | {self.rows_in} |"]
        lines += [f"| {k} | {v} |" for k, v in self.steps.items()]
        lines.append(f"| Clean rows | {self.rows_out} |")
        return "\n".join(lines)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Deduplicate, drop implausible values and impute missing fields.

    Rules are deliberately simple and explainable:
    * re-posted ads: identical on all listing attributes except ``listing_id``
    * implausible size or EUR/m2 (unit typos, "price on request" placeholders)
    * missing ``year_built``: median of the district; missing energy class: "unknown"
    """
    rep = CleaningReport(rows_in=len(raw))
    df = raw.copy()

    before = len(df)
    df = df.sort_values("listing_id").drop_duplicates(subset=FEATURE_KEY, keep="first")
    rep.steps["Removed re-posted duplicates"] = before - len(df)

    bad_size = ~df["size_m2"].between(*SIZE_RANGE)
    rep.steps["Removed implausible size"] = int(bad_size.sum())
    df = df[~bad_size]

    eur_m2 = df["cold_rent_eur"] / df["size_m2"]
    bad_price = ~eur_m2.between(*EUR_M2_RANGE)
    rep.steps["Removed implausible EUR/m2"] = int(bad_price.sum())
    df = df[~bad_price].copy()

    missing_year = df["year_built"].isna()
    rep.steps["Imputed year_built (district median)"] = int(missing_year.sum())
    district_median = df.groupby("district")["year_built"].transform("median")
    df["year_built"] = df["year_built"].fillna(district_median).round().astype(int)

    missing_energy = df["energy_class"].isna()
    rep.steps["Filled energy_class = 'unknown'"] = int(missing_energy.sum())
    df["energy_class"] = df["energy_class"].fillna("unknown")

    df["eur_per_m2"] = (df["cold_rent_eur"] / df["size_m2"]).round(2)
    df = df.reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep
