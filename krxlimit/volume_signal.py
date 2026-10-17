"""사용자 정의 RVOL 신호: D-20..D-1 중 직전 40시장일 평균의 3배 이상.

기존 analysis_pre의 고정 과거 중앙값이나 features의 60일 중앙값과 다른
정의이다. 사건일 D는 신호 및 평균에서 제외한다. 계산에는 준비기간을 모두
넣고 summarize_volume_signals의 event_start만 2016년으로 설정한다.

0거래량은 관측치이며 평균에 포함한다. 없는 종목일/결측/음수/비정수 거래량은
0으로 채우지 않는다. 사전 60시장일이 완전하고 20개 평균이 모두 양수여야
pre20_has_spike가 True/False가 된다. 그렇지 않으면 pd.NA다.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

import numpy as np
import pandas as pd

VOLUME_DEFINITION_BREAK = pd.Timestamp("2026-09-14")
RVOL_STATUSES = (
    "ok", "insufficient_40_day_calendar", "missing_40_day_observation",
    "missing_current_volume", "zero_40_day_baseline",
)
PRE_STATUSES = ("ok", "insufficient_60_day_calendar", "missing_history_observation", "zero_baseline_in_pre20")
DEFAULT_LABEL_COLUMNS = ("limit_valid", "up_close", "down_close")


def _dates(values) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if result.tz is not None:
        result = result.tz_localize(None)
    if result.isna().any() or not result.equals(result.normalize()):
        raise ValueError("Dates must be nonmissing market-session dates at midnight.")
    return result


def _preceding_sum(values: np.ndarray, window: int) -> np.ndarray:
    """각 행 t의 [max(0,t-window),t) 합. 현재 행은 제외한다."""
    prefix = np.r_[np.int64(0), np.cumsum(values, dtype=np.int64)]
    t = np.arange(len(values))
    return prefix[t] - prefix[np.maximum(t - window, 0)]


def _preceding_extreme(values: np.ndarray, window: int, fn: str) -> np.ndarray:
    rolling = pd.Series(values).rolling(window, min_periods=1)
    return getattr(rolling, fn)().shift(1).to_numpy()


def compute_volume_signals(
    panel: pd.DataFrame,
    *,
    calendar: Sequence | None = None,
    threshold: float = 3,
    label_columns: Sequence[str] = DEFAULT_LABEL_COLUMNS,
    require_matching_market_calendars: bool = True,
) -> pd.DataFrame:
    """원본 panel 행 순서/index에 맞춘 일별 RVOL과 사전20일 신호를 반환한다.

    필수 열: date, market, code, volume. 선택 열: shares 및 label_columns.
    시장과 종목을 함께 식별하므로 시장 이전 전후 이력을 자동 연결하지 않는다.
    calendar 생략 시 모든 관측일의 합집합을 사용한다. 두 시장이 모두 있으면
    기본적으로 시장별 관측일이 같은지 검증한다. 부분시장 자료를 명시적으로
    처리할 때만 require_matching_market_calendars=False와 외부 달력을 사용한다.

    신호 비교는 정수 거래량 합으로 40*V*분모 >= 40일합*분자를 계산한다.
    rvol40/pre20_max_rvol은 표시용 실수다. label_columns는 복사만 하며 사건
    정의를 변경하지 않는다. 결과 attrs에 달력과 계산 정의를 기록한다.
    """
    required = {"date", "market", "code", "volume"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError("Missing panel columns: " + ", ".join(sorted(missing_columns)))
    fraction = Fraction(str(threshold))
    if fraction <= 0:
        raise ValueError("threshold must be positive and finite")
    numerator, denominator = fraction.numerator, fraction.denominator
    if max(numerator, denominator) > 1_000_000:
        raise ValueError("threshold precision is too large for safe integer comparison")
    date_values = _dates(panel["date"])
    for column in ("market", "code"):
        if panel[column].isna().any() or panel[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must be nonmissing")
    keys = panel[["market", "code"]].copy()
    keys["date"] = date_values
    if keys.duplicated(["market", "code", "date"]).any():
        raise ValueError("Duplicate market/code/date observations must be resolved before analysis.")
    market_days = {
        str(market): np.sort(date_values[np.asarray(positions)].unique().to_numpy())
        for market, positions in panel.groupby("market", observed=True, sort=False).indices.items()
    }
    relevant = [market_days[m] for m in ("KOSPI", "KOSDAQ") if m in market_days]
    calendars_match = len(relevant) < 2 or np.array_equal(relevant[0], relevant[1])
    if require_matching_market_calendars and not calendars_match:
        raise ValueError("KOSPI/KOSDAQ observed calendars differ; inspect coverage before analysis.")
    days = (_dates(calendar).sort_values().unique() if calendar is not None else
            date_values.sort_values().unique())
    if len(days) == 0:
        raise ValueError("At least one market session is required.")
    positions = days.get_indexer(date_values)
    if (positions < 0).any():
        raise ValueError("Provided calendar omits panel observation dates.")
    n, T = len(panel), len(days)
    t = np.arange(T)
    volumes = pd.to_numeric(panel["volume"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    shares = (pd.to_numeric(panel["shares"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
              if "shares" in panel else np.full(n, np.nan))
    ratio_out = np.full(n, np.nan)
    ratio_status = np.full(n, 3, dtype=np.int8)
    pre_status = np.full(n, 2, dtype=np.int8)
    signal_values, signal_missing = np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    signal_count, observed_spike_count, defined_ratio_count = (np.zeros(n, dtype=np.int16) for _ in range(3))
    max_rvol = np.full(n, np.nan)
    share_change, share_missing = np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    zero_days = np.zeros(n, dtype=np.int16)
    # D itself is excluded: a break on event day does not contaminate its prior window.
    boundary = np.zeros(T, dtype=bool)
    if T > 1:
        boundary[1:] = ((days[np.maximum(t[1:] - 60, 0)] < VOLUME_DEFINITION_BREAK) &
                        (days[t[1:] - 1] >= VOLUME_DEFINITION_BREAK))
    safe_cap = min(2**53 - 1, np.iinfo(np.int64).max // max(T, 40 * numerator, 40 * denominator))
    invalid_observed_volumes = 0
    for _, rows in panel.groupby(["market", "code"], observed=True, sort=False).indices.items():
        rows = np.asarray(rows)
        ix = positions[rows]
        vol = np.full(T, np.nan)
        vol[ix] = volumes[rows]
        valid = np.isfinite(vol) & (vol >= 0) & (vol == np.floor(vol))
        invalid_observed_volumes += int((~valid[ix]).sum())
        if np.any(vol[valid] > safe_cap):
            raise ValueError("Trading volume exceeds the safe exact-integer comparison range.")
        integer_vol = np.where(valid, vol, 0).astype(np.int64)
        previous_sum = _preceding_sum(integer_vol, 40)
        previous_count = _preceding_sum(valid.astype(np.int64), 40)
        rs = np.zeros(T, dtype=np.int8)
        rs[previous_sum == 0] = 4
        rs[~valid] = 3
        rs[previous_count != 40] = 2
        rs[t < 40] = 1
        valid_ratio = rs == 0
        ratio = np.full(T, np.nan)
        ratio[valid_ratio] = integer_vol[valid_ratio] * 40 / previous_sum[valid_ratio]
        spike = valid_ratio & (integer_vol * (40 * denominator) >= previous_sum * numerator)
        complete_history = (t >= 60) & (_preceding_sum(valid.astype(np.int64), 60) == 60)
        defined = _preceding_sum(valid_ratio.astype(np.int64), 20)
        counts = _preceding_sum(spike.astype(np.int64), 20)
        ps = np.zeros(T, dtype=np.int8)
        ps[defined != 20] = 3
        ps[~complete_history] = 2
        ps[t < 60] = 1
        evaluable = ps == 0
        ratio_out[rows], ratio_status[rows], pre_status[rows] = ratio[ix], rs[ix], ps[ix]
        signal_values[rows] = counts[ix] > 0
        signal_missing[rows] = ~evaluable[ix]
        signal_count[rows] = counts[ix]
        observed_spike_count[rows] = counts[ix]
        defined_ratio_count[rows] = defined[ix]
        maxima = _preceding_extreme(ratio, 20, "max")
        max_rvol[rows] = np.where(evaluable[ix], maxima[ix], np.nan)
        sh = np.full(T, np.nan)
        sh[ix] = shares[rows]
        known_shares = np.isfinite(sh) & (sh > 0)
        sh = np.where(known_shares, sh, np.nan)
        share_change[rows] = (_preceding_extreme(sh, 60, "max") > _preceding_extreme(sh, 60, "min"))[ix]
        share_missing[rows] = ((_preceding_sum(known_shares.astype(np.int64), 60) != 60) | (t < 60))[ix]
        zero_days[rows] = _preceding_sum((valid & (integer_vol == 0)).astype(np.int64), 60)[ix]
    copied = ["date", "market", "code"] + [c for c in label_columns if c in panel and c not in required]
    out = panel[copied].copy()
    out["date"] = date_values
    out["rvol40"] = ratio_out
    out["rvol40_status"] = pd.Categorical.from_codes(ratio_status, categories=RVOL_STATUSES)
    out["pre20_status"] = pd.Categorical.from_codes(pre_status, categories=PRE_STATUSES)
    out["pre20_has_spike"] = pd.arrays.BooleanArray(signal_values, signal_missing)
    out["pre20_spike_count"] = pd.arrays.IntegerArray(signal_count, signal_missing.copy())
    out["pre20_observed_spike_count"] = observed_spike_count
    out["pre20_defined_ratio_days"] = defined_ratio_count
    out["pre20_max_rvol"] = max_rvol
    out["share_count_changed_pre60"] = share_change
    out["share_count_missing_pre60"] = share_missing
    out["zero_volume_days_pre60"] = zero_days
    out["volume_definition_break_pre60"] = boundary[positions]
    out.attrs.update({
        "threshold_multiple": float(fraction), "calendar_start": days[0].date().isoformat(),
        "calendar_end": days[-1].date().isoformat(), "calendar_sessions": T,
        "market_calendar_sessions": {m: len(d) for m, d in market_days.items()},
        "kospi_kosdaq_calendars_match": calendars_match,
        "invalid_or_missing_observed_volume_rows": invalid_observed_volumes,
        "definition": "Any V[d] >= threshold * mean(V[d-40:d]) for d in D-20..D-1; 60 complete preceding market-session observations and positive means required",
        "missing": "Missing rows and invalid volumes remain unknown; observed zero volume is included",
        "history_flags": "share and zero-volume flags use D-60..D-1; they do not change the main signal; missing share counts are flagged separately",
        "event_labels": "Copied from input without relabeling; results depend on supplied event and eligibility definitions",
    })
    return out


def summarize_volume_signals(
    signals: pd.DataFrame,
    *,
    event_start: str = "2016-01-01",
    event_end: str | None = None,
    label_eligible_col: str | None = "limit_valid",
    up_col: str = "up_close",
    down_col: str = "down_close",
) -> pd.DataFrame:
    """관측 종목일/라벨 적격 종목일/종가 사건을 각각 집계한다.

    원분모 observation_count, 완전이력 분모 evaluable_count, 신호수 signal_count,
    signal_rate_pct를 분리한다. eligible 열의 정의는 호출자가 명시해야 한다.
    label_eligible_col=None이면 해당 모집단을 만들지 않는다. 사건 열은 이미
    판정된 라벨을 그대로 사용한다. 다른 라벨을 비교할 때 계산된 signals에
    열을 붙인 뒤 해당 *_col 인자로 지정하면 동일 신호를 재사용할 수 있다.
    """
    needed = {"date", "market", "pre20_status", "pre20_has_spike"}
    if label_eligible_col is not None:
        needed.add(label_eligible_col)
    needed.update((up_col, down_col))
    absent = needed - set(signals.columns)
    if absent:
        raise ValueError("Missing summary columns: " + ", ".join(sorted(absent)))
    dates = _dates(signals["date"])
    start = pd.Timestamp(event_start)
    end = pd.Timestamp(event_end) if event_end else (dates.max() if len(dates) else start)
    if start > end:
        raise ValueError("event_start must not exceed event_end")
    selected = signals.loc[(dates >= start) & (dates <= end)]
    populations = {"all_observed_stock_days": np.ones(len(selected), dtype=bool)}
    if label_eligible_col is not None:
        populations["label_eligible_stock_days"] = selected[label_eligible_col].eq(True).fillna(False).to_numpy(dtype=bool)
    populations["close_up"] = selected[up_col].eq(True).fillna(False).to_numpy(dtype=bool)
    populations["close_down"] = selected[down_col].eq(True).fillna(False).to_numpy(dtype=bool)
    records = []
    for population, population_mask in populations.items():
        for market in ("ALL", *sorted(selected["market"].astype(str).unique())):
            market_mask = np.ones(len(selected), dtype=bool) if market == "ALL" else selected["market"].astype(str).eq(market).to_numpy()
            group = selected.loc[population_mask & market_mask]
            valid = group["pre20_status"].eq("ok") & group["pre20_has_spike"].notna()
            n = int(valid.sum())
            yes = int(group.loc[valid, "pre20_has_spike"].sum())
            bad = group.loc[~valid, "pre20_status"].astype(str).value_counts().to_dict()
            records.append({
                "population": population, "market": market,
                "observation_count": len(group), "evaluable_count": n,
                "signal_count": yes, "no_signal_count": n - yes,
                "unknown_count": len(group) - n, "signal_rate_pct": 100 * yes / n if n else np.nan,
                "unknown_status_counts": {str(k): int(v) for k, v in bad.items()},
            })
    result = pd.DataFrame(records)
    result.attrs.update(signals.attrs)
    result.attrs.update({"event_start": start.date().isoformat(), "event_end": end.date().isoformat(),
                         "label_eligible_column": label_eligible_col,
                         "event_columns": {"close_up": up_col, "close_down": down_col},
                         "rate_denominator": "evaluable_count; unknown histories are not treated as no signal"})
    return result
