"""호가가격단위와 상·하한가 계산 (KRX 규정, 벡터화).

상·하한가 = 기준가격 ± 가격제한폭
  가격제한폭 = 기준가격 × 30% 에서 '기준가격의 호가단위' 미만 절사
  산출된 상한가는 그 가격대 호가단위로 절사, 하한가는 절상
  (2015-06-15 이전은 15%)

호가가격단위
  ~2023-01-24  유가: <1천 1 / <5천 5 / <1만 10 / <5만 50 / <10만 100 / <50만 500 / 이상 1000
               코스닥: <1천 1 / <5천 5 / <1만 10 / <5만 50 / 이상 100
  2023-01-25~  공통: <2천 1 / <5천 5 / <2만 10 / <5만 50 / <20만 100 / <50만 500 / 이상 1000
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_OLD_KOSPI = ([1_000, 5_000, 10_000, 50_000, 100_000, 500_000], [1, 5, 10, 50, 100, 500, 1000])
_OLD_KOSDAQ = ([1_000, 5_000, 10_000, 50_000], [1, 5, 10, 50, 100])
_NEW = ([2_000, 5_000, 20_000, 50_000, 200_000, 500_000], [1, 5, 10, 50, 100, 500, 1000])


def _apply_table(price: np.ndarray, table) -> np.ndarray:
    bounds, ticks = table
    idx = np.searchsorted(np.asarray(bounds), price, side="right")
    return np.asarray(ticks, dtype=np.int64)[idx]


def tick_size(price, dates, is_kosdaq) -> np.ndarray:
    """가격·날짜·시장별 호가단위 (int64 배열)."""
    price = np.asarray(price, dtype=np.int64)
    dates = pd.to_datetime(np.asarray(dates))
    is_kosdaq = np.asarray(is_kosdaq, dtype=bool)
    new = dates >= pd.Timestamp(config.TICK_UNIFIED_FROM)
    return np.where(
        new,
        _apply_table(price, _NEW),
        np.where(is_kosdaq, _apply_table(price, _OLD_KOSDAQ), _apply_table(price, _OLD_KOSPI)),
    )


def limit_prices(base, dates, is_kosdaq) -> tuple[np.ndarray, np.ndarray]:
    """기준가격으로부터 (상한가, 하한가). 정수 연산으로 부동소수 오차를 피한다."""
    base = np.asarray(base, dtype=np.int64)
    dates = pd.to_datetime(np.asarray(dates))
    pct = np.where(dates >= pd.Timestamp(config.LIMIT_30_FROM), 30, 15).astype(np.int64)
    t_base = tick_size(base, dates, is_kosdaq)
    width = (base * pct // 100) // t_base * t_base
    upper = base + width
    lower = base - width
    t_up = tick_size(upper, dates, is_kosdaq)
    upper = upper // t_up * t_up
    t_lo = tick_size(np.maximum(lower, 1), dates, is_kosdaq)
    lower = -((-lower) // t_lo) * t_lo  # 절상
    return upper, lower
