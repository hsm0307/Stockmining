import numpy as np
import pandas as pd
import pytest

from krxlimit.volume_signal import compute_volume_signals, summarize_volume_signals


def panel(n=150, start="2020-01-01", code="111110", market="KOSPI"):
    return pd.DataFrame({
        "date": pd.bdate_range(start, periods=n), "market": market, "code": code,
        "volume": 100, "shares": 1000, "limit_valid": True,
        "up_close": False, "down_close": False,
    })


def test_exact_three_times_counts_and_event_day_is_excluded():
    for lead, expected in ((21, False), (20, True), (1, True), (0, False)):
        p = panel()
        p.loc[120 - lead, "volume"] = 300
        out = compute_volume_signals(p)
        assert out.loc[120, "pre20_status"] == "ok"
        assert bool(out.loc[120, "pre20_has_spike"]) == expected
    p = panel()
    p.loc[119, "volume"] = 299
    assert not compute_volume_signals(p).loc[120, "pre20_has_spike"]


def test_each_candidate_day_uses_own_shifted40_day_mean():
    p = panel()
    p.loc[100, "volume"] = 500
    p.loc[101, "volume"] = 310
    out = compute_volume_signals(p)
    assert out.loc[100, "rvol40"] == 5
    assert out.loc[101, "rvol40"] == pytest.approx(310 / 110)
    assert out.loc[110, "pre20_spike_count"] == 1


def test_regression_fixed_older_median_is_not_prior40_mean():
    p = panel()
    p.loc[80:139, "volume"] = 1000
    p.loc[139, "volume"] = 500
    out = compute_volume_signals(p)
    # analysis_pre's old D-120..D-61 fixed median would count all 20 days.
    old_baseline = np.expm1(np.median(np.log1p(p.loc[20:79, "volume"])))
    assert (p.loc[120:139, "volume"] > 3 * old_baseline).sum() == 20
    assert out.loc[140, "pre20_spike_count"] == 0
    assert not out.loc[140, "pre20_has_spike"]


def test_regression_strict_greater_drops_exact_three_boundary():
    p = panel()
    p.loc[119, "volume"] = 300
    assert not (300 > 3 * 100)
    assert compute_volume_signals(p).loc[120, "pre20_has_spike"]


def test_full60_is_required_and_warmup_used_before2016():
    p = panel(start="2015-10-01")
    p.loc[65, "up_close"] = True
    p.loc[60, "volume"] = 300
    out = compute_volume_signals(p)
    assert pd.isna(out.loc[59, "pre20_has_spike"])
    assert out.loc[59, "pre20_status"] == "insufficient_60_day_calendar"
    assert out.loc[60, "pre20_status"] == "ok"
    summary = summarize_volume_signals(out, event_start="2016-01-01")
    stock_days = summary.query("population=='all_observed_stock_days' and market=='ALL'").iloc[0]
    assert stock_days.observation_count == (p.date >= "2016-01-01").sum()
    assert stock_days.unknown_count == 0


def test_missing_security_day_preserves_global_market_calendar_and_is_unknown():
    p = panel()
    anchor = panel(code="222220")
    p = p.drop(index=70)
    source = pd.concat([p, anchor], ignore_index=True)
    out = compute_volume_signals(source)
    target = out[(out.code == "111110") & (out.date == anchor.loc[120, "date"])].iloc[0]
    assert target.pre20_status == "missing_history_observation"
    assert pd.isna(target.pre20_has_spike)
    assert pd.isna(target.pre20_spike_count)
    assert pd.isna(target.pre20_max_rvol)


def test_zero_volume_is_observation_and_stays_in_mean():
    p = panel()
    p.loc[70, "volume"] = 0
    p.loc[100, "volume"] = 293
    out = compute_volume_signals(p)
    assert out.loc[100, "rvol40"] == pytest.approx(293 / 97.5)
    assert out.loc[110, "pre20_has_spike"]
    assert out.loc[110, "zero_volume_days_pre60"] == 1
    assert out.loc[110, "pre20_status"] == "ok"


def test_zero_mean_and_missing_observation_do_not_become_no_signal():
    p = panel()
    p.loc[:59, "volume"] = 0
    out = compute_volume_signals(p)
    assert out.loc[60, "rvol40_status"] == "zero_40_day_baseline"
    assert out.loc[70, "pre20_status"] == "zero_baseline_in_pre20"
    assert pd.isna(out.loc[70, "pre20_has_spike"])
    p["volume"] = p["volume"].astype(float)
    p.loc[50, "volume"] = np.nan
    out = compute_volume_signals(p)
    assert out.loc[70, "pre20_status"] == "missing_history_observation"
    assert out.attrs["invalid_or_missing_observed_volume_rows"] == 1


@pytest.mark.parametrize("bad", [-1, 0.5, np.inf])
def test_invalid_volume_is_unknown_not_zero(bad):
    p = panel()
    p["volume"] = p["volume"].astype(float)
    p.loc[70, "volume"] = bad
    out = compute_volume_signals(p)
    assert out.loc[120, "pre20_status"] == "missing_history_observation"
    assert pd.isna(out.loc[120, "pre20_has_spike"])


def test_future_and_event_day_changes_do_not_change_prior_signal():
    p = panel()
    before = compute_volume_signals(p).loc[120]
    p.loc[120:, "volume"] = 100000
    p.loc[120:, "shares"] = 2000
    after = compute_volume_signals(p).loc[120]
    for col in ("pre20_has_spike", "pre20_spike_count", "pre20_max_rvol", "share_count_changed_pre60"):
        assert before[col] == after[col]


def test_share_missing_and_changes_are_separate_history_flags():
    p = panel()
    p["shares"] = p["shares"].astype(float)
    p.loc[70, "shares"] = np.nan
    p.loc[90:, "shares"] = 2000
    out = compute_volume_signals(p)
    assert out.loc[120, "share_count_changed_pre60"]
    assert out.loc[120, "share_count_missing_pre60"]
    assert out.loc[120, "pre20_status"] == "ok"


def test_volume_definition_boundary_excludes_event_day_itself():
    p = panel(start="2026-05-01")
    out = compute_volume_signals(p).set_index("date")
    assert not out.loc["2026-09-14", "volume_definition_break_pre60"]
    assert out.loc["2026-09-15", "volume_definition_break_pre60"]


def test_market_transfer_does_not_join_histories():
    p = panel()
    p.loc[100:, "market"] = "KOSDAQ"
    anchors = pd.concat([panel(code="222220"), panel(code="333330", market="KOSDAQ")], ignore_index=True)
    source = pd.concat([p, anchors], ignore_index=True)
    out = compute_volume_signals(source)
    assert out.loc[120, "market"] == "KOSDAQ"
    assert out.loc[120, "pre20_status"] == "missing_history_observation"
    assert pd.isna(out.loc[120, "pre20_has_spike"])


def test_calendar_mismatch_rejected_but_explicit_partial_mode_preserves_missing():
    a, b = panel(), panel(code="222220", market="KOSDAQ")
    source = pd.concat([a, b.drop(index=70)], ignore_index=True)
    with pytest.raises(ValueError, match="calendars differ"):
        compute_volume_signals(source)
    out = compute_volume_signals(source, calendar=a.date, require_matching_market_calendars=False)
    target = out[(out.market == "KOSDAQ") & (out.date == a.loc[120, "date"])].iloc[0]
    assert pd.isna(target.pre20_has_spike)


def test_row_order_index_and_input_values_are_preserved():
    p = panel().sample(frac=1, random_state=4)
    source = p.copy(deep=True)
    out = compute_volume_signals(p)
    pd.testing.assert_frame_equal(p, source)
    assert out.index.equals(p.index)
    assert out.date.equals(p.date)
    assert out.loc[120, "pre20_status"] == "ok"
    with pytest.raises(ValueError, match="Duplicate"):
        compute_volume_signals(pd.concat([p, p.iloc[:1]]))


def test_observed_and_label_eligible_population_denominators_are_distinct():
    p = panel()
    p.loc[100, "volume"] = 300
    p.loc[[110, 121], "up_close"] = True
    p.loc[120, "limit_valid"] = False
    p.loc[120, "down_close"] = True  # independently supplied alternative label
    out = compute_volume_signals(p)
    summary = summarize_volume_signals(out, event_start=p.loc[110, "date"], event_end=p.loc[121, "date"])
    cells = summary.query("market=='ALL'").set_index("population")
    assert cells.loc["all_observed_stock_days", "observation_count"] == 12
    assert cells.loc["label_eligible_stock_days", "observation_count"] == 11
    assert cells.loc["close_up", "evaluable_count"] == 2
    assert cells.loc["close_up", "signal_count"] == 1
    assert cells.loc["close_up", "signal_rate_pct"] == 50
    assert cells.loc["close_down", "signal_count"] == 1
    out["strict_eligible"] = False
    alt = summarize_volume_signals(out, label_eligible_col="strict_eligible")
    assert alt.query("population=='label_eligible_stock_days'").observation_count.sum() == 0


def test_missing_histories_remain_in_original_denominator_not_signal_denominator():
    p = panel()
    p.loc[[20, 80], "up_close"] = True
    out = compute_volume_signals(p)
    s = summarize_volume_signals(out)
    row = s.query("market=='ALL' and population=='close_up'").iloc[0]
    assert row.observation_count == 2 and row.evaluable_count == 1
    assert row.unknown_count == 1 and row.signal_count == 0
    assert row.unknown_status_counts == {"insufficient_60_day_calendar": 1}


def test_fractional_threshold_uses_exact_rational_integer_comparison():
    p = panel()
    p.loc[119, "volume"] = 250
    assert compute_volume_signals(p, threshold=2.5).loc[120, "pre20_has_spike"]
    assert not compute_volume_signals(p, threshold=3).loc[120, "pre20_has_spike"]
