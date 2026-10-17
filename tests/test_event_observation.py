"""Regression cases for fixed market-date returns and unavailable observations."""
import numpy as np
import pandas as pd
import pytest

from krxlimit.events import EventContext


def _panel(a_rows, n_days=5):
    """Quotes are (day index, close, base, open, volume); B keeps the market calendar."""
    days = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for code, quotes in [("A", a_rows), ("B", [(t, 100, 100, 100, 10) for t in range(n_days)])]:
        for t, close, base, open_, volume in quotes:
            row = {"date": days[t], "code": code, "market": "KOSPI", "close": close,
                   "base": base, "open": open_, "high": max(open_, close) if open_ else 0,
                   "low": min(open_, close) if open_ else 0, "volume": volume,
                   "ret": close / base - 1, "traded": volume > 0 and open_ > 0,
                   "upper": close, "lower": close, "mktcap": 100_000, "limit_valid": True,
                   "up_streak": 0, "down_streak": 0, "is_spac": False, "is_pref": False}
            row.update({f"{side}_{kind}": False for side in ("up", "down")
                        for kind in ("close", "touch", "jeom", "fail")})
            rows.append(row)
    p = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
    p["ri"] = (1 + p["ret"].fillna(0)).groupby(p["code"]).cumprod()
    return p


def _event(panel, day=0):
    date = np.sort(panel["date"].unique())[day]
    return panel[(panel["code"] == "A") & (panel["date"] == date)]


def _return(ctx, panel, h, day=0, **kwargs):
    ti, ci, _ = ctx.locate(_event(panel, day))
    return ctx.fwd_return(ti, ci, h, **kwargs)


def test_missing_endpoint_is_not_stale_zero_or_verified_delisting():
    p = _panel([(0, 100, 100, 100, 10), (1, 130, 100, 130, 10)])
    ctx = EventContext(p)
    r, start, observation_ended = _return(ctx, p, 1, day=1)
    assert np.isnan(r[0])
    assert start.tolist() == [1]
    assert observation_ended.tolist() == [True]
    ti, ci, _ = ctx.locate(_event(p, 1))
    assert np.isnan(ctx.ri[ti[0] + 1, ci[0]])
    carried, _, _ = _return(ctx, p, 1, day=1, observation_policy="last_observation")
    assert carried[0] == 0  # A named valuation assumption, not an observed exit.


@pytest.mark.parametrize("policy", ["observed_path", "last_observation"])
def test_end_of_dataset_is_right_censored_under_every_policy(policy):
    p = _panel([(0, 100, 100, 100, 10)])
    r, _, ended = _return(EventContext(p), p, 5, observation_policy=policy)
    assert np.isnan(r[0])
    assert not ended[0]


def test_interior_missing_row_breaks_reference_link_even_after_reappearance():
    p = _panel([(0, 100, 100, 100, 10), (2, 132, 120, 120, 10)])
    ctx = EventContext(p)
    r, _, ended = _return(ctx, p, 2)
    assert np.isnan(r[0]) and not ended[0]
    assumed, _, _ = _return(ctx, p, 2, observation_policy="last_observation")
    assert assumed[0] == pytest.approx(0.10, abs=1e-6)
    # The missing 20% daily move cannot be recovered from the observed RI path.
    assert p.loc[p.code == "A", "close"].iloc[-1] / 100 - 1 == pytest.approx(0.32)


def test_known_halt_cannot_be_an_exit_but_can_be_inside_a_holding_path():
    p = _panel([(0, 100, 100, 100, 10), (1, 100, 100, 0, 0), (2, 110, 100, 105, 10)])
    ctx = EventContext(p)
    halted, _, ended = _return(ctx, p, 1)
    assert np.isnan(halted[0]) and not ended[0]
    resumed, _, _ = _return(ctx, p, 2)
    assert resumed[0] == pytest.approx(0.10, abs=1e-6)


def test_present_row_with_unknown_daily_return_does_not_silently_link():
    p = _panel([(0, 100, 100, 100, 10), (1, 110, 100, 110, 10), (2, 121, 110, 120, 10)])
    p.loc[(p.code == "A") & (p.date == p.date.unique()[1]), "ret"] = np.nan
    ctx = EventContext(p)
    assert np.isnan(_return(ctx, p, 2)[0][0])
    # An unknown return before a new anchor does not invalidate later observed holding intervals.
    assert _return(ctx, p, 1, day=1)[0][0] == pytest.approx(0.10, abs=1e-6)


def test_unverified_market_change_breaks_a_holding_path_but_allows_new_anchor():
    p = _panel([(0, 100, 100, 100, 10), (1, 110, 100, 110, 10), (2, 121, 110, 120, 10)])
    p.loc[(p.code == "A") & (p.date >= p.date.unique()[1]), "market"] = "KOSDAQ"
    ctx = EventContext(p)
    assert np.isnan(_return(ctx, p, 2)[0][0])
    assert _return(ctx, p, 1, day=1)[0][0] == pytest.approx(0.10, abs=1e-6)


def test_reference_linked_proxy_does_not_equal_unadjusted_close_ratio():
    p = _panel([(0, 100, 100, 100, 10), (1, 55, 50, 50, 10)])
    proxy, _, _ = _return(EventContext(p), p, 1)
    assert proxy[0] == pytest.approx(0.10, abs=1e-6)
    assert p.loc[p.code == "A", "close"].iloc[-1] / 100 - 1 == pytest.approx(-0.45)


def test_next_open_horizon_one_is_same_day_intraday_return():
    p = _panel([(0, 100, 100, 100, 10), (1, 154, 100, 140, 10), (2, 168, 154, 160, 10)])
    ctx = EventContext(p)
    assert _return(ctx, p, 1, entry="next_open")[0][0] == pytest.approx(0.10, abs=1e-6)
    assert _return(ctx, p, 2, entry="next_open")[0][0] == pytest.approx(0.20, abs=1e-6)
    with pytest.raises(ValueError, match="h >= 1"):
        _return(ctx, p, 0, entry="next_open")


@pytest.mark.parametrize("policy", ["observed_path", "last_observation"])
def test_next_open_never_invents_entry_on_a_missing_or_halted_day(policy):
    for quotes in [[(0, 100, 100, 100, 10), (2, 110, 100, 110, 10)],
                   [(0, 100, 100, 100, 10), (1, 100, 100, 0, 0), (2, 110, 100, 110, 10)]]:
        p = _panel(quotes)
        assert np.isnan(_return(EventContext(p), p, 2, entry="next_open", observation_policy=policy)[0][0])


def test_mask_works_in_both_directions_without_filling_the_path():
    p = _panel([(0, 100, 100, 100, 10), (2, 110, 100, 110, 10), (3, 121, 110, 120, 10)])
    ctx = EventContext(p)
    ti, ci, _ = ctx.locate(_event(p, 2))
    offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
    assert ctx.observed_path_mask(ti, ci, offsets).tolist() == [[False, False, False, True, True, False, False]]
    assert np.isnan(ctx.path(ti, ci, offsets, ctx.ri)[0, 2])
    assert ctx.observed_path_mask(np.array([], int), np.array([], int), offsets).shape == (0, 7)


def test_stats_keep_original_denominator_and_full_empty_schema():
    from krxlimit.analysis_post import _stats
    st = _stats(np.array([np.nan, np.nan, np.nan]), np.full(3, np.nan),
                np.array([True, False, False]), np.array([True, True, False]))
    assert st["사건수"] == 3 and st["관측기간충족수"] == 2 and st["n"] == 0
    assert st["관측소실건수"] == 1 and st["관측소실비율"] == 0.5
    assert st["수익률미산출수"] == 3
    assert np.isnan(st["평균"]) and np.isnan(st["시장초과평균"])
    assert "상장폐지비율" not in st


def test_two_observations_can_have_empty_quantile_trim_without_warning():
    from krxlimit.analysis_post import _stats
    with np.errstate(all="raise"):
        st = _stats(np.array([0.1, 0.2]), np.array([np.nan, np.nan]), np.array([False, False]))
    assert st["n"] == 2 and st["평균"] == pytest.approx(0.15)
    assert np.isnan(st["절사평균(1%)"])


def test_next_open_market_excess_is_unavailable_without_open_benchmark():
    from krxlimit.analysis_post import event_returns, _stats
    p = _panel([(0, 100, 100, 100, 10), (1, 110, 100, 110, 10)])
    ctx = EventContext(p)
    out = event_returns(ctx, _event(p), "next_open", "up")
    assert out["r_1d"].iloc[0] == pytest.approx(0)
    assert np.isnan(out["m_1d"].iloc[0])
    st = _stats(out["r_1d"].to_numpy(), out["m_1d"].to_numpy(),
                out["observation_ended_1d"].to_numpy())
    assert st["n"] == 1 and np.isnan(st["시장초과평균"])


def test_empty_event_analysis_writes_defined_empty_tables(monkeypatch):
    from krxlimit import analysis_post
    p = _panel([(0, 100, 100, 100, 10)])
    saved = {}
    monkeypatch.setattr(analysis_post, "save_table", lambda df, name: saved.setdefault(name, df))
    monkeypatch.setattr(analysis_post, "save_fig", lambda *args: pytest.fail("No observed returns to plot"))
    result = analysis_post.run(p)
    assert not result["full"]["n"].any()
    assert result["paths"].empty
    assert "g3_paths" in saved and "평균" in result["full"]


def test_post_path_does_not_resume_after_an_unobserved_gap(monkeypatch):
    from krxlimit import analysis_post
    p = _panel([(0, 100, 100, 100, 10), (1, 130, 100, 130, 10), (3, 156, 143, 150, 10)])
    ix = _event(p, 1).index
    p.loc[ix, ["up_close", "up_touch"]] = True
    p.loc[ix, "up_streak"] = 1
    monkeypatch.setattr(analysis_post, "save_table", lambda *args: None)
    monkeypatch.setattr(analysis_post, "save_fig", lambda *args: pytest.fail("No observed horizon return"))
    result = analysis_post.run(p)
    path = result["paths"].query("이벤트 == '상따(상한가 터치 전체)'").set_index("경과일")
    assert path.loc[0, "표본수"] == 1
    assert (path.loc[1:, "표본수"] == 0).all()
    assert path.loc[1:, "평균"].isna().all()
