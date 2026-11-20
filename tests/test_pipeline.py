import json

from frankfurt_rent.pipeline import run


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = run(n=600, out=tmp_path / "reports", figures=tmp_path / "figs")
    assert (tmp_path / "reports" / "report.md").read_text().startswith("# Frankfurt")
    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["cleaning"]["rows_out"] == metrics["cleaning"]["rows_out"]
    assert (tmp_path / "figs" / "district_eur_m2.png").stat().st_size > 10_000
    assert (tmp_path / "data" / "clean_listings.csv").exists()
