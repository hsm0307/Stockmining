"""테스트 전용 합성 데이터 생성기.

KRX Open API 응답과 같은 형식(문자열 컬럼)의 가짜 시세를 만들어 파이프라인이
끝까지 돌아가는지만 검증한다. 여기서 나온 수치는 분석 결과가 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from krxlimit import ticks


def _tick(price, d, is_kosdaq):
    new = d >= pd.Timestamp("2023-01-25")
    table = ticks._NEW if new else (ticks._OLD_KOSDAQ if is_kosdaq else ticks._OLD_KOSPI)
    return int(table[1][int(np.searchsorted(table[0], price, side="right"))])


def _limits(base, d, is_kosdaq):
    t = _tick(base, d, is_kosdaq)
    w = (base * 30 // 100) // t * t
    up, lo = base + w, base - w
    up = up // _tick(up, d, is_kosdaq) * _tick(up, d, is_kosdaq)
    tl = _tick(lo, d, is_kosdaq)
    return up, -((-lo) // tl) * tl


def make_raw(n_stocks: int = 60, start: str = "2021-06-01", end: str = "2023-06-30",
             seed: int = 0, delist: int = 3, new_list: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    rows = []
    for j in range(n_stocks):
        market = "KOSDAQ" if j % 2 else "KOSPI"
        is_kosdaq = market == "KOSDAQ"
        code = f"{100000 + j * 10:06d}" if j % 13 else f"{100000 + j * 10 + 5:06d}"  # 일부 우선주(끝자리 5)
        name = f"테스트{j}" + ("우" if code.endswith("5") else "")
        shares = int(rng.integers(5, 50) * 1_000_000)
        close = float(rng.choice([800, 3_000, 9_000, 25_000, 120_000]))
        start_k = int(rng.integers(30, 200)) if j < new_list else 0
        end_k = int(rng.integers(len(days) - 200, len(days) - 30)) if j >= n_stocks - delist else len(days)
        streak = 0
        for k in range(start_k, end_k):
            d = days[k]
            t = _tick(close, d, is_kosdaq)
            base = int(round(close / t) * t)
            # 가끔 액면분할(기준가 조정)
            if k > start_k + 5 and rng.random() < 0.0005 and base > 5_000:
                tb = _tick(base // 5, d, is_kosdaq)
                base = (base // 5) // tb * tb
            up, lo = _limits(base, d, is_kosdaq)
            halted = rng.random() < 0.003
            u = rng.random()
            if halted:
                o = h = l = 0
                c = base
                vol = 0
            else:
                if streak > 0 and rng.random() < 0.35 or u < 0.006:
                    c = up
                    streak += 1
                elif u < 0.009:
                    c = lo
                    streak = 0
                else:
                    streak = 0
                    r = rng.normal(0, 0.03)
                    tc = _tick(max(base * (1 + r), 1), d, is_kosdaq)
                    c = int(np.clip(round(base * (1 + r) / tc) * tc, lo, up))
                if c == up and rng.random() < 0.3:
                    o = h = l = c  # 점상
                else:
                    o = int(np.clip(base * (1 + rng.normal(0, 0.01)), lo, up))
                    o = _tick(o, d, is_kosdaq) * (o // _tick(o, d, is_kosdaq))
                    h = max(o, c, int(min(up, max(o, c) * (1 + abs(rng.normal(0, 0.01))))))
                    l = min(o, c, int(max(lo, min(o, c) * (1 - abs(rng.normal(0, 0.01))))))
                    if rng.random() < 0.01:
                        h = up  # 상한가 터치 후 이탈
                vol = int(rng.lognormal(11, 1) * (5 if c == up else 1))
            chg = c - base if k > start_k or start_k == 0 else 0
            rows.append({
                "BAS_DD": d.strftime("%Y%m%d"), "ISU_CD": code, "ISU_NM": name, "MKT_NM": market,
                "SECT_TP_NM": ("벤처기업부" if is_kosdaq else ""), "TDD_CLSPRC": str(c),
                "CMPPREVDD_PRC": str(chg), "FLUC_RT": f"{(chg / base * 100) if base else 0:.2f}",
                "TDD_OPNPRC": str(o), "TDD_HGPRC": str(h), "TDD_LWPRC": str(l),
                "ACC_TRDVOL": str(vol), "ACC_TRDVAL": str(vol * c), "MKTCAP": str(c * shares),
                "LIST_SHRS": str(shares), "MARKET": market,
            })
            close = c
    return pd.DataFrame(rows)
