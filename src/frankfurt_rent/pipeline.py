"""End-to-end run: generate -> clean -> EDA -> model -> report.

Usage: ``python -m frankfurt_rent.pipeline [--n 3000] [--out reports]``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from frankfurt_rent import eda, model, plots
from frankfurt_rent.clean import clean
from frankfurt_rent.generate import SEED, generate_raw


def _md(df: pd.DataFrame, index: bool = True, digits: int = 2) -> str:
    """Markdown table; integer columns without decimals, floats with ``digits``."""
    fmts = [".0f" if pd.api.types.is_integer_dtype(t) else f".{digits}f" for t in df.dtypes]
    if index:
        fmts = ["", *fmts]
    return df.to_markdown(index=index, floatfmt=fmts)


def run(n: int = 3000, out: Path = Path("reports"), figures: Path = Path("docs/figures")) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    raw = generate_raw(n, SEED)
    raw.to_csv(data_dir / "raw_listings.csv", index=False)
    df, cleaning = clean(raw)
    df.to_csv(data_dir / "clean_listings.csv", index=False)

    overview = eda.overview(df)
    districts = eda.district_table(df)
    sizes = eda.size_table(df)
    amenities = eda.amenity_table(df)
    cv = model.cross_validate(df)
    effects = model.ridge_effects(df)
    holdout, importance, preds = model.holdout_importance(df)
    overpriced = model.flag_overpriced(preds)

    plots.plot_districts(districts, figures / "district_eur_m2.png")
    plots.plot_importance(importance, figures / "permutation_importance.png")

    report = "\n\n".join(
        [
            "# Frankfurt rent analysis report (synthetic data)",
            "## Cleaning\n" + cleaning.as_markdown(),
            "## Overview\n" + _md(pd.Series(overview, name="value").to_frame(), digits=2),
            "## Districts\n" + _md(districts),
            "## Size buckets\n" + _md(sizes),
            "## Amenities (raw, unadjusted)\n" + _md(amenities, index=False),
            "## 5-fold cross-validation\n" + _md(cv),
            "## Ridge coefficients vs. true effects\n" + _md(effects, digits=3),
            "## Hold-out (HistGradientBoosting)\n" + _md(pd.Series(holdout, name="value").to_frame()),
            "## Permutation importance\n" + _md(importance, index=False, digits=4),
            f"## Over-priced listings (>15% above fair-rent estimate): {len(overpriced)} of {len(preds)}\n"
            + _md(overpriced[["listing_id", "district", "size_m2", "cold_rent_eur",
                              "predicted_rent_eur", "price_gap"]].head(10), index=False),
        ]
    )
    (out / "report.md").write_text(report + "\n", encoding="utf-8")

    metrics = {
        "cleaning": {"rows_in": cleaning.rows_in, **cleaning.steps, "rows_out": cleaning.rows_out},
        "overview": overview,
        "cv": cv.reset_index().to_dict(orient="records"),
        "holdout": holdout,
        "overpriced": {"flagged": int(len(overpriced)), "test_rows": int(len(preds))},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=3000, help="number of synthetic listings")
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    metrics = run(args.n, args.out)
    print(json.dumps({"cv": metrics["cv"], "holdout": metrics["holdout"]}, indent=2))
    print(f"Report written to {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
