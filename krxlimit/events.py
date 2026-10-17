"""이벤트(상·하한가 발생일) 전후 수익률 계산 공통 모듈.

- ri는 공시된 등락액으로 역산한 기준가격을 연결한 가격수익률 대용치다. 검증된 수정주가나
  배당·분배금 포함 총수익률이 아니며, 모든 기업행위가 정확히 처리됐음을 보장하지 않는다.
- 기본값은 진입일부터 목표일까지 관측이 이어지고 양 끝에 정규장 거래 흔적이 있는 경로만 계산한다.
  끝점 무거래 제외는 체결 조건에 따른 표본 선택이며, 손실의 보수적 추정이나 생존편향 제거가 아니다.
- 관측소실은 상장폐지 확인이나 청산을 뜻하지 않는다. 마지막 관측값 유지는 명시적 민감도 옵션이다.
- 시장 거래일 기준 목표일이 데이터 끝을 넘으면 모든 정책에서 NaN이다.
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
        self.ri = W.wide(panel["ri"].to_numpy())
        self.close = W.wide(panel["close"].to_numpy())
        self.open = W.wide(panel["open"].to_numpy())
        self.base = W.wide(panel["base"].to_numpy())
        self.high = W.wide(panel["high"].to_numpy())
        self.low = W.wide(panel["low"].to_numpy())
        self.volume = W.wide(panel["volume"].to_numpy())
        self.ret = W.wide(panel["ret"].to_numpy())
        self.market = W.wide(pd.factorize(panel["market"], sort=True)[0], fill=-1, dtype=np.int16)
        self.traded = W.wide(panel["traded"].to_numpy(), fill=False, dtype=bool)
        exists = W.wide(np.ones(len(panel)), fill=0, dtype=np.int8).astype(bool)
        self.exists = exists
        rows = np.arange(self.T)[:, None]
        self.last_t = np.where(exists, rows, -1).max(axis=0)
        self._observation_bad_prefix = None
        self._link_bad_prefix = None
        self._last_observation_ri = None
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

    def observed_path_mask(self, ti, ci, offsets, entry_offset: int = 0) -> np.ndarray:
        """각 앵커(ti+entry_offset)와 목표일(ti+offset) 사이의 관측 가능 여부.

        두 끝점은 양수 종가/ri와 정규장 거래 흔적(traded, 양수 시가·거래량)이 필요하다.
        중간의 알려진 거래정지는 보유할 수 있지만, 종목 행 누락이나 기준수익률 미상은
        연결하지 않는다. 같은 코드의 시장 변경도 연결 이력이 검증되지 않아 단절로 취급한다.
        음수 오프셋도 같은 기준을 적용한다. 체결 보장이나 기업행위 검증은 아니다.
        """
        ti, ci = np.asarray(ti, dtype=int), np.asarray(ci, dtype=int)
        offsets = np.asarray(offsets, dtype=int)
        if self._observation_bad_prefix is None:
            observed = self.exists & np.isfinite(self.ri) & (self.ri > 0)
            observed &= np.isfinite(self.close) & (self.close > 0)
            linked = np.isfinite(self.ret) & (self.ret > -1)
            linked &= np.isfinite(self.base) & (self.base > 0)
            linked[1:] &= (self.market[1:] == self.market[:-1]) & (self.market[1:] >= 0)
            # Separate prefixes let the interval's first row have an unknown prior-day return.
            self._observation_bad_prefix = np.zeros((self.T + 1, self.N), dtype=np.int32)
            self._link_bad_prefix = np.zeros((self.T + 1, self.N), dtype=np.int32)
            np.cumsum(~observed, axis=0, dtype=np.int32, out=self._observation_bad_prefix[1:])
            np.cumsum(~linked, axis=0, dtype=np.int32, out=self._link_bad_prefix[1:])
        start = ti[:, None] + entry_offset
        target = ti[:, None] + offsets[None, :]
        within = (start >= 0) & (start < self.T) & (target >= 0) & (target < self.T)
        start, target = np.clip(start, 0, self.T - 1), np.clip(target, 0, self.T - 1)
        lo, hi = np.minimum(start, target), np.maximum(start, target)
        cc = ci[:, None]
        bad_rows = self._observation_bad_prefix[hi + 1, cc] - self._observation_bad_prefix[lo, cc]
        bad_links = self._link_bad_prefix[hi + 1, cc] - self._link_bad_prefix[lo + 1, cc]
        endpoints_traded = self.traded[start, cc] & self.traded[target, cc]
        endpoints_traded &= (self.open[start, cc] > 0) & (self.open[target, cc] > 0)
        endpoints_traded &= (self.volume[start, cc] > 0) & (self.volume[target, cc] > 0)
        return within & (bad_rows == 0) & (bad_links == 0) & endpoints_traded

    def fwd_return(self, ti, ci, h: int, entry: str = "close", entry_price=None,
                   *, observation_policy: str = "observed_path"):
        """이벤트일 D로부터 h 시장 거래일 뒤(D+h) 종가 기준 가격수익률 대용치.

        entry='close'     : 이벤트일 종가 매수
        entry='price'     : 이벤트일 entry_price(예: 상한가 가격)에 매수
        entry='next_open' : D+1 시가 매수. h=1은 D+1 시가→같은 날 종가다.
        observed_path는 미관측 끝점/중간 누락/끝점 무거래를 NaN으로 둔다.
        last_observation은 누락 뒤 ri를 마지막 관측 수준으로 유지하는 민감도 가정이다.
        이 값은 실제 청산수익률이 아니며 거래정지·상장폐지 손익을 식별하지 못한다.
        반환: (수익률, 이벤트 시점 인덱스, 목표일 전에 관측이 끝났는지 여부).
        세 번째 값은 관측소실 flag일 뿐, 상장폐지 확인값이 아니다.
        """
        if not isinstance(h, (int, np.integer)) or h < 0:
            raise ValueError("h must be a nonnegative integer")
        if entry == "next_open" and h < 1:
            raise ValueError("next_open requires h >= 1 (event-day horizon)")
        if observation_policy not in {"observed_path", "last_observation"}:
            raise ValueError(f"Unknown observation_policy: {observation_policy}")
        ti, ci = np.asarray(ti, dtype=int), np.asarray(ci, dtype=int)
        T = self.T
        tgt = ti + h
        ok = (tgt >= 0) & (tgt < T)
        if entry not in {"close", "price", "next_open"}:
            raise ValueError(entry)
        entry_offset = 1 if entry == "next_open" else 0
        start = np.clip(ti + entry_offset, 0, T - 1)
        t2 = np.clip(tgt, 0, T - 1)
        if observation_policy == "observed_path":
            ri = self.ri
            ok &= self.observed_path_mask(ti, ci, [h], entry_offset=entry_offset)[:, 0]
        else:
            if self._last_observation_ri is None:
                self._last_observation_ri = pd.DataFrame(self.ri).ffill().to_numpy()
            ri = self._last_observation_ri
            # The sensitivity assumption fills valuations, never the entry transaction.
            ok &= self.observed_path_mask(ti, ci, [entry_offset], entry_offset=entry_offset)[:, 0]
        price = (self.open[start, ci] if entry == "next_open" else
                 np.asarray(entry_price, dtype=float) if entry == "price" else self.close[start, ci])
        ok &= np.isfinite(price) & (price > 0)
        ok &= np.isfinite(ri[t2, ci]) & (ri[t2, ci] > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ri[t2, ci] / ri[start, ci] * self.close[start, ci] / price - 1
        r = np.where(ok & np.isfinite(r), r, np.nan)
        observation_ended = (tgt < T) & (self.last_t[ci] < tgt) & (self.last_t[ci] < T - 1)
        return r, ti, observation_ended

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

