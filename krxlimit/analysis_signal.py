"""Exact pre-event volume signal and subsequent reference-linked price changes.

This is an event study, not a fitted model or a claim that limit-price orders fill.
The 1/21/126-session horizons are measured from event day D for both entries.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .events import EventContext
from .volume_signal import compute_volume_signals, summarize_volume_signals

HORIZONS = (1, 21, 126)


def compare_signal_returns(panel, signals, *, event_start='2016-01-01', event_end=None, ctx=None):
    """Keep unknown signals separate and expose maturity/observation denominators."""
    if not panel.index.equals(signals.index):
        raise ValueError('Signal rows must align with the source panel index.')
    for col in ('date', 'market', 'code'):
        if not np.array_equal(panel[col].astype(str).to_numpy(), signals[col].astype(str).to_numpy()):
            raise ValueError('Signal identifiers must align with the source panel.')
    ctx = ctx or EventContext(panel)
    date_ok = panel['date'].ge(pd.Timestamp(event_start))
    if event_end:
        date_ok &= panel['date'].le(pd.Timestamp(event_end))
    rows = []
    for side in ('up', 'down'):
        mask = date_ok & panel[f'{side}_close']
        ev = panel.loc[mask]
        signal = signals.loc[mask, 'pre20_has_spike']
        groups = np.where(signal.isna(), 'unknown', np.where(signal.fillna(False), 'yes', 'no'))
        ti, ci, _ = ctx.locate(ev)
        markets = ev['market'].astype(str).to_numpy()
        for entry in ('close', 'next_open'):
            for horizon in HORIZONS:
                returns, _, _ = ctx.fwd_return(ti, ci, horizon, entry=entry)
                mature = ti + horizon < ctx.T
                for market in ('ALL', *sorted(set(markets))):
                    market_mask = np.ones(len(ev), dtype=bool) if market == 'ALL' else markets == market
                    for group in ('yes', 'no', 'unknown'):
                        selected = market_mask & (groups == group)
                        values = returns[selected]
                        finite = values[np.isfinite(values)]
                        rows.append({
                            'direction': side, 'market': market, 'signal': group,
                            'entry': entry, 'horizon_sessions_from_event': horizon,
                            'event_count': int(selected.sum()),
                            'mature_count': int((selected & mature).sum()),
                            'observed_return_count': len(finite),
                            'unavailable_count': int(selected.sum()) - len(finite),
                            'mean_pct': float(finite.mean()*100) if len(finite) else np.nan,
                            'median_pct': float(np.median(finite)*100) if len(finite) else np.nan,
                            'positive_rate_pct': float((finite > 0).mean()*100) if len(finite) else np.nan,
                            'return_definition': 'reference_linked_proxy_observed_path',
                        })
    return pd.DataFrame(rows)


def run(panel, *, event_start='2016-01-01', event_end=None, output_dir=None, signals=None):
    output_dir = Path(output_dir) if output_dir else config.TABLE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    signals = compute_volume_signals(panel) if signals is None else signals
    rates = summarize_volume_signals(signals, event_start=event_start, event_end=event_end)
    returns = compare_signal_returns(panel, signals, event_start=event_start, event_end=event_end)
    rates.to_csv(output_dir / 'g6_volume_signal_rates.csv', index=False, encoding='utf-8-sig')
    returns.to_csv(output_dir / 'g6_signal_returns.csv', index=False, encoding='utf-8-sig')
    manifest = dict(rates.attrs)
    manifest.update({
        'horizons_sessions_from_event': list(HORIZONS),
        'entries': {'close': 'D close', 'next_open': 'D+1 open; horizon 1 is the same-day close'},
        'return_definition': 'Product of close/(close-change), a reference-linked price proxy; not verified total return',
        'observation_policy': 'Observed contiguous path and tradable endpoint required; missing is not zero return or confirmed delisting',
        'source_sha256': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                          for name in ('analysis_signal.py', 'volume_signal.py', 'events.py', 'panel.py', 'ticks.py')},
        'limitations': ['No execution guarantee, cash dividends, rights cash flows or delisting recovery',
                        'Future path selection changes the sample; not a return lower bound',
                        'Signal frequency conditional on events is not future-event precision',
                        'Repeated events and overlapping horizons are not independent observations'],
    })
    (output_dir / 'g6_methodology.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(rates.loc[rates['market'].eq('ALL')].to_string(index=False))
    return {'rates': rates, 'returns': returns, 'signals': signals}
