"""Static matplotlib figures for the README (single-series bar charts, light theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"


def _style(ax: plt.Axes, title: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def _hbar(labels: list[str], values: list[float], title: str, xlabel: str, fmt: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(labels) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, height=0.62, color=SERIES)
    ax.set_yticks(list(y), labels)
    for yi, v in zip(y, values, strict=True):
        ax.text(v, yi, " " + format(v, fmt), va="center", ha="left", color=INK_2, fontsize=8)
    _style(ax, title, xlabel)
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_districts(districts: pd.DataFrame, path: Path) -> Path:
    return _hbar(
        districts.index.tolist(), districts["median_eur_m2"].tolist(),
        "Median cold rent per m² by district (synthetic data)", "EUR per m²", ".1f", path,
    )


def plot_importance(imp: pd.DataFrame, path: Path) -> Path:
    return _hbar(
        imp["feature"].tolist(), imp["importance"].tolist(),
        "Permutation importance, gradient boosting (hold-out set)",
        "Increase in MAE of log(rent) when the feature is shuffled", ".3f", path,
    )
