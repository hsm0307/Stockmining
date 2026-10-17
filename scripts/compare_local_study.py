"""Reconcile event definitions on the same read-only local-study quotes.

This adapter only compares event labels and the exact volume signal. The local
SQLite schema lacks traded value and listing section; those fields stay missing
and this adapter must not be used to train the prediction model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krxlimit.panel import build_panel


def cache_signature(database):
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in [database, Path(str(database)+'-wal'), root/'krxlimit/panel.py', root/'krxlimit/ticks.py']:
        digest.update(path.name.encode())
        if path.exists():
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    digest.update(chunk)
    from krxlimit.panel import LIMIT_RULE
    digest.update(LIMIT_RULE.encode())
    return digest.hexdigest()


def load_comparison_panel(database: Path) -> pd.DataFrame:
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        q = pd.read_sql_query('SELECT * FROM quotes', conn)
    raw = q.rename(columns={
        'day': 'BAS_DD', 'code': 'ISU_CD', 'name': 'ISU_NM', 'market': 'MARKET',
        'close': 'TDD_CLSPRC', 'open': 'TDD_OPNPRC', 'high': 'TDD_HGPRC',
        'low': 'TDD_LWPRC', 'change': 'CMPPREVDD_PRC', 'change_pct': 'FLUC_RT',
        'volume': 'ACC_TRDVOL', 'shares': 'LIST_SHRS',
    })
    raw['BAS_DD'] = raw['BAS_DD'].str.replace('-', '', regex=False)
    raw['MKT_NM'] = raw['MARKET']
    raw['SECT_TP_NM'] = ''
    raw['ACC_TRDVAL'] = np.nan
    raw['MKTCAP'] = np.nan
    return build_panel(raw, save=False)


def study_events(path: Path, start: str, end: str) -> pd.DataFrame:
    rows = []
    for line in path.open(encoding='utf-8'):
        e = json.loads(line)
        if e['event_type'] not in ('close_up', 'close_down') or not start <= e['date'] <= end:
            continue
        rows.append({
            'date': pd.Timestamp(e['date']), 'market': e['market'], 'code': e['code'],
            'direction': e['event_type'].removeprefix('close_'),
            'study_signal': e['prior_volume']['has_signal'],
        })
    result = pd.DataFrame(rows, columns=['date', 'market', 'code', 'direction', 'study_signal'])
    result['date'] = pd.to_datetime(result['date'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--normalized-db', type=Path, required=True)
    ap.add_argument('--study-events', type=Path, required=True)
    ap.add_argument('--study-exclusions', type=Path)
    ap.add_argument('--event-start', default='2016-01-01')
    ap.add_argument('--event-end', default='2026-10-14')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--panel-cache', type=Path)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(a.normalized_db)
    stamp = a.panel_cache.with_suffix('.signature') if a.panel_cache else None
    if a.panel_cache and a.panel_cache.exists() and stamp.exists() and stamp.read_text() == signature:
        p = pd.read_parquet(a.panel_cache)
    else:
        print('Building comparison-only panel from local read-only SQLite...', flush=True)
        p = load_comparison_panel(a.normalized_db)
        if a.panel_cache:
            a.panel_cache.parent.mkdir(parents=True, exist_ok=True)
            p.to_parquet(a.panel_cache, index=False)
            if cache_signature(a.normalized_db) != signature:
                raise RuntimeError('The source SQLite changed during comparison; retry on a stable snapshot.')
            stamp.write_text(signature)
    sample = p[p['date'].between(a.event_start, a.event_end)]
    previous = study_events(a.study_events, a.event_start, a.event_end)
    keys = ['date', 'market', 'code', 'direction']
    labels = []
    for side in ['up', 'down']:
        ev = sample.loc[sample[f'{side}_close'], ['date', 'market', 'code']].copy()
        ev['direction'] = side
        labels.append(ev)
    joined = pd.concat(labels).merge(previous, on=keys, how='outer', indicator=True)
    joined['membership'] = joined['_merge'].map({
        'left_only': 'repository_only', 'right_only': 'local_study_only', 'both': 'both'})
    counts = joined.groupby(['direction', 'membership'], observed=True).size().unstack(fill_value=0)
    counts.to_csv(a.output_dir / 'event_label_overlap.csv', encoding='utf-8-sig')
    joined[joined['_merge'] != 'both'].drop(columns='_merge').to_csv(
        a.output_dir / 'event_label_disagreements.csv', index=False, encoding='utf-8-sig')
    metadata = {
        'comparison_base_commit': 'ce87af8d91c3eebd3410d3f33028f2028da48d87',
        'event_start': a.event_start, 'event_end': a.event_end,
        'data_start': str(p['date'].min().date()), 'data_end': str(p['date'].max().date()),
        'panel_rows': len(p), 'event_period_rows': len(sample),
        'security_codes': int(p['code'].nunique()),
        'market_security_pairs': int(p[['market', 'code']].drop_duplicates().shape[0]),
        'missing_adapter_fields': ['value', 'mktcap', 'sect'],
        'note': 'Same local source quotes; not a re-run on the unavailable GitHub raw data.'}
    if a.study_exclusions:
        excluded = {}
        for line in a.study_exclusions.open(encoding='utf-8'):
            e = json.loads(line)
            excluded[(e['date'], e['market'], e['code'])] = e
        reasons = Counter()
        for row in joined[joined['_merge'] == 'left_only'].itertuples():
            e = excluded.get((str(row.date.date()), row.market, row.code), {})
            reasons.update((row.direction, r) for r in e.get('exclusion_reasons', ['not_in_exclusion_log']))
        pd.DataFrame([{'direction': s, 'reason': r, 'count': n} for (s, r), n in reasons.items()]).to_csv(
            a.output_dir / 'event_label_difference_reasons.csv', index=False, encoding='utf-8-sig')
    (a.output_dir / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(counts.to_string(), flush=True)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    from krxlimit.volume_signal import compute_volume_signals, summarize_volume_signals
    from krxlimit.analysis_signal import run
    print('Computing exact 20/40 volume signal and subsequent returns...', flush=True)
    signals = compute_volume_signals(p)
    signal_keys = ['date', 'market', 'code']
    checked = previous.merge(signals[signal_keys + ['pre20_has_spike', 'pre20_status']],
                             on=signal_keys, how='left', validate='many_to_one', indicator=True)
    missing_events = checked['_merge'].ne('both')
    if missing_events.any():
        raise ValueError(f'{int(missing_events.sum())} local-study events are missing from the comparison panel.')
    checked = checked.drop(columns='_merge')
    def value(v):
        return 'unknown' if pd.isna(v) else str(bool(v))
    mismatch = checked['study_signal'].map(value) != checked['pre20_has_spike'].map(value)
    checked.loc[mismatch].to_csv(a.output_dir/'volume_signal_disagreements.csv', index=False, encoding='utf-8-sig')
    metadata['volume_signal_event_checks'] = len(checked)
    metadata['volume_signal_mismatches'] = int(mismatch.sum())
    rates = []
    for side in ('up', 'down'):
        g = checked[checked['direction'].eq(side)]
        known = g['pre20_has_spike'].notna()
        known_count = int(known.sum())
        yes = int(g.loc[known, 'pre20_has_spike'].sum())
        rates.append({'direction': side, 'events': len(g), 'evaluable': known_count,
                      'signal_yes': yes, 'signal_no': known_count-yes,
                      'unknown': int((~known).sum()),
                      'signal_rate_pct': 100*yes/known_count if known_count else np.nan})
    pd.DataFrame(rates).to_csv(a.output_dir/'local_study_signal_reproduction.csv', index=False, encoding='utf-8-sig')
    (a.output_dir / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    run(p, signals=signals, event_start=a.event_start, event_end=a.event_end, output_dir=a.output_dir)
    print(f'Signal cross-check: {len(checked)} events, {int(mismatch.sum())} differences', flush=True)


if __name__ == '__main__':
    main()
