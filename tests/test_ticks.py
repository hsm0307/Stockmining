import numpy as np
import pandas as pd

from krxlimit.ticks import limit_prices, tick_size

OLD = pd.Timestamp("2022-06-02")
NEW = pd.Timestamp("2024-06-03")


def _lim(base, d, kosdaq=False):
    u, l = limit_prices([base], np.array([d]), [kosdaq])
    return int(u[0]), int(l[0])


def test_tick_tables():
    assert tick_size([999, 1_000, 4_995, 5_000, 49_950, 50_000, 120_000], [OLD] * 7, [False] * 7).tolist() == \
        [1, 5, 5, 10, 50, 100, 500]
    # 코스닥 구 규정: 5만원 이상 100원
    assert tick_size([120_000, 600_000], [OLD] * 2, [True] * 2).tolist() == [100, 100]
    # 2023-01-25 통일 규정
    assert tick_size([1_999, 2_000, 19_990, 20_000, 199_900, 200_000, 500_000], [NEW] * 7, [True] * 7).tolist() == \
        [1, 5, 10, 50, 100, 500, 1000]


def test_limit_prices_basic():
    assert _lim(10_000, OLD) == (13_000, 7_000)
    assert _lim(10_000, NEW) == (13_000, 7_000)
    # 기준가 호가단위 미만 절사: 1,005 × 0.3 = 301.5 → 300 (호가 5)
    assert _lim(1_005, OLD) == (1_305, 705)


def test_limit_upper_truncated_to_its_tick():
    # 9,990 (호가 10) → 폭 2,990 → 12,980 은 1만~5만 구간(호가 50)이므로 12,950 으로 절사
    up, lo = _lim(9_990, OLD)
    assert up == 12_950
    assert lo == 7_000


def test_limit_lower_rounded_up():
    # 신 규정 2,100 (호가 5) → 폭 630 → 하한 1,470 (호가 1) / 상한 2,730
    assert _lim(2_100, NEW) == (2_730, 1_470)
    # 구 규정 5,010 (호가 10) → 폭 1,500 → 하한 3,510 (호가 5 → 그대로)
    assert _lim(5_010, OLD) == (6_510, 3_510)


def test_fifteen_percent_before_2015():
    assert _lim(10_000, pd.Timestamp("2014-03-03")) == (11_500, 8_500)
