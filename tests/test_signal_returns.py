"""신호별 수익률 집계의 분모와 진입 시점에 대한 회귀검증."""
import numpy as np
import pandas as pd
import pytest

from krxlimit.analysis_signal import compare_signal_returns


class ControlledContext:
    """분모 집계를 수익률 경로 계산과 독립적으로 검증한다."""
    T = 130

    def locate(self, events):
        positions = events["tidx"].to_numpy(dtype=int)
        return positions, np.zeros(len(events), dtype=int), np.zeros(len(events), dtype=int)

    def fwd_return(self, ti, ci, h, entry="close"):
        values = np.where(ti == 20, 0.0, np.where(ti == 70, 0.10 if entry == "close" else 0.05, np.nan))
        values = np.where(ti + h < self.T, values, np.nan)
        return values, ti, np.zeros(len(ti), dtype=bool)


def grouped_inputs():
    days = pd.bdate_range("2020-01-01", periods=130)
    source = pd.DataFrame({
        "date": days[[20, 70, 80, 129]], "market": "KOSPI", "code": "111110",
        "tidx": [20, 70, 80, 129], "up_close": True, "down_close": False,
    })
    signals = source[["date", "market", "code"]].copy()
    signals["pre20_has_spike"] = pd.array([pd.NA, True, True, False], dtype="boolean")
    return source, signals


def get_cell(result, signal, h=1, entry="close", direction="up"):
    return result[(result.direction == direction) & (result.market == "ALL") &
                  (result.signal == signal) & (result.entry == entry) &
                  (result.horizon_sessions_from_event == h)].iloc[0]


def test_yes_no_unknown_and_three_denominators_are_separate():
    source, signals = grouped_inputs()
    result = compare_signal_returns(source, signals, ctx=ControlledContext())
    yes = get_cell(result, "yes")
    assert (yes.event_count, yes.mature_count, yes.observed_return_count) == (2, 2, 1)
    assert yes.unavailable_count == 1
    assert yes.mean_pct == pytest.approx(10)
    no = get_cell(result, "no")
    assert (no.event_count, no.mature_count, no.observed_return_count) == (1, 0, 0)
    assert np.isnan(no.mean_pct)
    unknown = get_cell(result, "unknown")
    assert (unknown.event_count, unknown.mature_count, unknown.observed_return_count) == (1, 1, 1)
    # 실제 0수익률은 결측과 달리 관측분모에 남는다.
    assert unknown.mean_pct == 0 and unknown.positive_rate_pct == 0


def test_original_signal_denominators_are_unchanged_across_horizons_and_entries():
    source, signals = grouped_inputs()
    result = compare_signal_returns(source, signals, ctx=ControlledContext())
    for signal, n in (("yes", 2), ("no", 1), ("unknown", 1)):
        for entry in ("close", "next_open"):
            for horizon in (1, 21, 126):
                row = get_cell(result, signal, horizon, entry)
                assert row.event_count == n
                assert row.observed_return_count <= row.mature_count <= row.event_count
    assert get_cell(result, "yes", 126).mature_count == 0
    assert get_cell(result, "yes", 1, "next_open").mean_pct == pytest.approx(5)
    assert get_cell(result, "yes", direction="down").event_count == 0


def test_event_range_filters_events_without_reclassifying_unknown_signals():
    source, signals = grouped_inputs()
    result = compare_signal_returns(source, signals, ctx=ControlledContext(),
                                    event_start=source.loc[1, "date"], event_end=source.loc[2, "date"])
    assert get_cell(result, "yes").event_count == 2
    assert get_cell(result, "unknown").event_count == 0
    assert get_cell(result, "no").event_count == 0


def test_identifier_and_index_misalignment_is_rejected():
    source, signals = grouped_inputs()
    with pytest.raises(ValueError, match="index"):
        compare_signal_returns(source, signals.iloc[::-1], ctx=ControlledContext())
    signals.loc[0, "code"] = "999990"
    with pytest.raises(ValueError, match="identifiers"):
        compare_signal_returns(source, signals, ctx=ControlledContext())


def test_one_day_close_entry_and_next_open_entry_have_same_exit_different_entry():
    days = pd.bdate_range("2024-01-02", periods=3)
    source = pd.DataFrame({
        "date": days, "market": "KOSPI", "code": "111110",
        "close": [1000, 1100, 1100], "open": [1000, 1200, 1100],
        "high": [1010, 1210, 1110], "low": [990, 1090, 1090],
        "base": [1000, 1000, 1100], "ret": [0, 0.1, 0], "ri": [1, 1.1, 1.1],
        "volume": 100, "traded": True,
        "up_close": [True, False, False], "down_close": False,
        "up_jeom": False, "down_jeom": False, "up_fail": False, "down_fail": False,
    })
    signals = source[["date", "market", "code"]].copy()
    signals["pre20_has_spike"] = pd.array([True, False, False], dtype="boolean")
    result = compare_signal_returns(source, signals)
    at_close, next_open = get_cell(result, "yes"), get_cell(result, "yes", entry="next_open")
    assert at_close.observed_return_count == next_open.observed_return_count == 1
    assert at_close.mean_pct == pytest.approx((1100 / 1000 - 1) * 100, abs=1e-4)
    # 1d는 D+1 시가에 진입해 같은 D+1 종가로 끝난다. D+2 종가로 밀리지 않는다.
    assert next_open.mean_pct == pytest.approx((1100 / 1200 - 1) * 100, abs=1e-4)
