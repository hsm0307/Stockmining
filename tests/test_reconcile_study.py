"""The comparison CLI must distinguish missing events from unknown signals."""
import json
import sys
import warnings

import pandas as pd
import pytest

from krxlimit import analysis_signal
from scripts import compare_local_study as comparison


def _event(code="000010"):
    return {"event_type": "close_up", "date": "2024-01-02", "market": "KOSPI",
            "code": code, "prior_volume": {"has_signal": None}}


def _source(tmp_path, events):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _setup_cli(monkeypatch, tmp_path, events, start="2024-01-01", end="2024-01-31"):
    panel = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=3), "market": "KOSPI", "code": "000010",
        "volume": 100, "shares": 1000, "limit_valid": True,
        "up_close": [True, False, False], "down_close": False,
    })
    source = _source(tmp_path, events)
    output = tmp_path / "comparison"
    monkeypatch.setattr(comparison, "cache_signature", lambda path: "stable-source")
    monkeypatch.setattr(comparison, "load_comparison_panel", lambda path: panel)
    # Keep the actual signal computation and reconciliation, isolating unrelated return reports.
    monkeypatch.setattr(analysis_signal, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "compare_local_study", "--normalized-db", str(tmp_path / "source.sqlite"),
        "--study-events", str(source), "--output-dir", str(output),
        "--event-start", start, "--event-end", end,
    ])
    return output


def test_study_events_preserve_merge_schema_for_empty_range(tmp_path):
    result = comparison.study_events(_source(tmp_path, [_event()]), "2025-01-01", "2025-12-31")
    assert result.empty
    assert list(result) == ["date", "market", "code", "direction", "study_signal"]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_empty_event_range_finishes_with_zero_counts_and_unknown_rate(monkeypatch, tmp_path):
    output = _setup_cli(monkeypatch, tmp_path, [_event()], start="2025-01-01", end="2025-01-31")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        comparison.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["event_period_rows"] == 0
    assert manifest["volume_signal_event_checks"] == manifest["volume_signal_mismatches"] == 0
    rates = pd.read_csv(output / "local_study_signal_reproduction.csv")
    assert rates["events"].eq(0).all() and rates["evaluable"].eq(0).all()
    assert rates["signal_rate_pct"].isna().all()


def test_observed_event_with_unknown_history_is_retained_without_zero_denominator_warning(monkeypatch, tmp_path):
    output = _setup_cli(monkeypatch, tmp_path, [_event()])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        comparison.main()
    rates = pd.read_csv(output / "local_study_signal_reproduction.csv").set_index("direction")
    assert rates.loc["up", "events"] == rates.loc["up", "unknown"] == 1
    assert rates.loc["up", "evaluable"] == 0
    assert rates["signal_rate_pct"].isna().all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["volume_signal_event_checks"] == 1
    assert manifest["volume_signal_mismatches"] == 0


def test_absent_event_fails_even_when_its_original_signal_was_unknown(monkeypatch, tmp_path):
    output = _setup_cli(monkeypatch, tmp_path, [_event("999990")])
    with pytest.raises(ValueError, match="1 local-study events are missing from the comparison panel"):
        comparison.main()
    assert not (output / "local_study_signal_reproduction.csv").exists()
