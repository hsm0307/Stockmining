"""이벤트(상·하한가 발생일) 전후 수익률 계산 공통 모듈.

- 수익률은 수익률지수(ri)로 계산하므로 분할·증자 등 기업행위의 영향을 받지 않는다(현금배당 제외 가격수익률).
- 상장폐지 종목: 마지막 거래(정리매매 포함) 가격으로 청산된 것으로 본다 → 생존편향 제거.
- 데이터 끝을 넘어서는 구간은 NaN (아직 관측되지 않음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import Wide, market_daily


class EventContext:
    def __init__(self, panel: pd.DataFrame):
        self.W = W = Wide(panel)
        self.T, self.N = W.shape
        self.days = W.days
        self.ri = pd.DataFrame(W.wide(panel["ri"].to_numpy())).ffill().to_numpy()
        self.close = W.wide(panel["close"].to_numpy())
        self.open = W.wide(panel["open"].to_numpy())
        self.base = W.wide(panel["base"].to_numpy())
        self.high = W.wide(panel["high"].to_numpy())
        self.low = W.wide(panel["low"].to_numpy())
        self.volume = W.wide(panel["volume"].to_numpy())
        self.ret = W.wide(panel["ret"].to_numpy())
        self.traded = W.wide(panel["traded"].to_numpy(), fill=False, dtype=bool)
        exists = W.wide(np.ones(len(panel)), fill=0, dtype=np.int8).astype(bool)
        self.exists = exists
        rows = np.arange(self.T)[:, None]
        self.last_t = np.where(exists, rows, -1).max(axis=0)
        # 시장별 동일가중 지수 (0: KOSPI, 1: KOSDAQ)
        md = market_daily(panel)
        self.mkt_ret = np.zeros((self.T, 2))
        pos = np.searchsorted(self.days, md["date"].to_numpy())
        for k, m in enumerate(["KOSPI", "KOSDAQ"]):
            sel = (md["market"] == m).to_numpy()
            self.mkt_ret[pos[sel], k] = md.loc[sel, "ew_ret"].to_numpy()
        self.mi = np.cumprod(1 + self.mkt_ret, axis=0)

    def locate(self, ev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """이벤트 행(date/code/market 포함) → (시점 인덱스, 종목 열 인덱스, 시장 인덱스)."""
        ti = np.searchsorted(self.days, ev["date"].to_numpy())
        ci = np.searchsorted(self.W.codes, ev["code"].astype(str).to_numpy())
        mi = (ev["market"].astype(str) == "KOSDAQ").to_numpy().astype(int)
        return ti, ci, mi

    def fwd_return(self, ti, ci, h: int, entry: str = "close", entry_price=None):
        """h 거래일 뒤 종가에 판다고 할 때 수익률.

        entry='close'     : 이벤트일 종가 매수
        entry='price'     : 이벤트일 entry_price(예: 상한가 가격)에 매수
        entry='next_open' : 다음 거래일 시가 매수
        반환: (수익률, 시장 동일가중 수익률, 상장폐지로 조기청산 여부)
        """
        T = self.T
        tgt = ti + h
        ok = tgt < T
        t2 = np.minimum(tgt, T - 1)
        end = self.ri[t2, ci]
        if entry == "close":
            r = end / self.ri[ti, ci] - 1
            t_start = ti
        elif entry == "price":
            r = end / self.ri[ti, ci] * self.close[ti, ci] / np.asarray(entry_price, dtype=float) - 1
            t_start = ti
        elif entry == "next_open":
            t1 = np.minimum(ti + 1, T - 1)
            o = self.open[t1, ci]
            good = (ti + 1 < T) & self.traded[t1, ci] & (o > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = end / self.ri[t1, ci] * self.close[t1, ci] / o - 1
            r = np.where(good, r, np.nan)
            t_start = ti
        else:
            raise ValueError(entry)
        r = np.where(ok, r, np.nan)
        delisted = ok & (self.last_t[ci] < tgt) & (self.last_t[ci] < T - 1)
        return r, t_start, delisted

    def mkt_fwd(self, ti, mi, h: int):
        tgt = ti + h
        t2 = np.minimum(tgt, self.T - 1)
        r = self.mi[t2, mi] / self.mi[ti, mi] - 1
        return np.where(tgt < self.T, r, np.nan)

    def path(self, ti, ci, offsets: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """(이벤트 수 × 오프셋) 행렬로 arr 값을 모은다. 범위 밖은 NaN."""
        idx = ti[:, None] + offsets[None, :]
        ok = (idx >= 0) & (idx < self.T)
        out = arr[np.clip(idx, 0, self.T - 1), ci[:, None]].astype(float)
        out[~ok] = np.nan
        return out
