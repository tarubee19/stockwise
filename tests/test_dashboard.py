from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def test_dashboard_without_results(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    app = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
    at = streamlit_testing.AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    assert any("No results" in e.value for e in at.error)
