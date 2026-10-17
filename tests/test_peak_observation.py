import numpy as np
import pandas as pd
import pytest

from krxlimit.analysis_peak import streak_extremes
from krxlimit.events import EventContext
from krxlimit.panel import build_panel
from test_panel import _row


def panel_with_gap(missing=()):
    days = pd.bdate_range('2020-01-02', periods=300)
    rows = []
    for i, day in enumerate(days):
        ds = day.strftime('%Y%m%d')
        rows.append(_row(ds, '222220', 1000, 0, 1000, 1000, 1000, 100))
        if i in missing:
            continue
        price = 1000 if i < 10 else 1300 if i == 10 else 1690
        chg = 300 if i == 10 else 390 if i == 11 else 0
        rows.append(_row(ds, '111110', price, chg, price, price, price, 100))
    return build_panel(pd.DataFrame(rows), save=False)


def extreme(missing=()):
    panel = panel_with_gap(missing)
    return streak_extremes(panel, EventContext(panel), 'up')


def test_missing_price_inside_extreme_selection_window_is_not_ignored():
    assert len(extreme()) == 1
    assert extreme([20]).empty


def test_internal_future_gap_is_unknown_not_flat_zero_return():
    row = extreme([60]).iloc[0]
    assert row['1년기간도달']
    assert not row['1년수익률관측']
    assert np.isnan(row['극점후1년'])
    assert np.isnan(row['1년내최대낙폭'])
    assert row['목표일전관측소실'] == 0  # The code reappears; not a delisting classification.


def test_terminal_disappearance_is_separate_from_return_availability():
    row = extreme(range(200, 300)).iloc[0]
    assert row['1년기간도달'] and not row['1년수익률관측']
    assert np.isnan(row['극점후1년'])
    assert row['목표일전관측소실'] == 1
    assert '상장폐지' not in row.index


@pytest.mark.parametrize('missing', [(), (20,)])
def test_peak_run_can_plot_an_empty_direction(monkeypatch, missing):
    from krxlimit import analysis_peak
    figures = []
    monkeypatch.setattr(analysis_peak, 'save_table', lambda *args: None)
    monkeypatch.setattr(analysis_peak, 'save_fig', lambda fig, name: figures.append(fig))
    try:
        result = analysis_peak.run(panel_with_gap(missing))
        assert len(result['up'][0]) == (0 if missing else 1)
        assert result['down'][0].empty
        assert result['down'][1].loc['전체', 'n'] == 0
        assert len(figures) == 1
    finally:
        for fig in figures:
            analysis_peak.plt.close(fig)


def test_plot_does_not_resume_after_a_gap_when_the_table_excludes_it(monkeypatch):
    from krxlimit import analysis_peak
    base = panel_with_gap([60])
    stock = base[base['code'].astype(str) == '111110']
    panel = pd.concat([stock.assign(code=f'{100000 + i}') for i in range(10)] +
                      [base[base['code'].astype(str) == '222220']], ignore_index=True)
    panel = panel.sort_values(['code', 'date']).reset_index(drop=True)
    ctx = EventContext(panel)
    res = {}
    for side in ('up', 'down'):
        details = streak_extremes(panel, ctx, side)
        res[side] = (details, analysis_peak.summarize(details, side))
    assert len(res['up'][0]) == 10
    assert res['up'][0]['극점후1년'].isna().all()
    figures = []
    monkeypatch.setattr(analysis_peak, 'save_fig', lambda fig, name: figures.append(fig))
    try:
        analysis_peak._plot(panel, ctx, res)
        lines = [line for line in figures[0].axes[0].lines if line.get_label().startswith('2일 연속')]
        assert len(lines) == 1
        x, y = lines[0].get_data()
        # The peak is market day 11; missing day 60 disconnects every later target.
        assert np.isfinite(y[x == 48]).all()
        assert np.isnan(y[x >= 49]).all()
        assert np.isnan(y[x == 250]).all()
    finally:
        for fig in figures:
            analysis_peak.plt.close(fig)
