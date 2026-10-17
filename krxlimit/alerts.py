"""KRX 시장경보(투자주의·경고·위험)와 단기과열종목 지정을 일별 시세로 재구성한 특징.

KIND 공시 규정 문구를 따라 '오늘 장 마감 기준으로 내일 어떤 조치가 붙는가'를 상태기계로 추정한다.
- 투자경고: 예고(10거래일 관찰) → 지정 → (2일 +40% & 지정 전 종가 상회 시 1일 매매정지) → 위험예고 → 투자위험(지정일 정지)
  → 10거래일 뒤 해제 요건 → 해제 후 10일 재지정 감시
- 단기과열: 가격(40일 평균 대비 130%)·회전율(6배)·변동성(1.5배) 동시 충족 → 예고 → 3일 단일가매매(연장 1회),
  우선주는 보통주 대비 괴리율 50% 초과 시 바로 지정
- 날짜는 '해당 종목의 매매거래일' 기준. 행 t 의 값은 t일 종가까지의 정보만 사용.
시가총액 상위 100 종목 예외(2026-05-27~), 우선주 괴리율 기준(2020-09-28~) 같은 규정 변경일도 반영.
※ 실제 지정 이력 데이터가 아니라 규정을 가격으로 재현한 추정치다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _div, _roll, _shift  # noqa: F401 (features 모듈의 공통 함수 재사용)

ALERT_FEATURES = ["alert_lvl_next", "alert_days", "alert_event", "halt_pred", "warn_need", "halt_need",
                  "rel_need", "soft_trig", "n_desig250"]
OVERHEAT_FEATURES = ["oh_price", "oh_turn", "oh_vol", "oh_lvl_next", "oh_left", "pref_prem"]
TOP100_FROM = "2026-05-27"
PREF_PREM_FROM = "2020-09-28"


def _own_shift(a: np.ndarray, traded: np.ndarray, k: int) -> np.ndarray:
    """종목 자신의 매매거래일 기준 k일 전 값."""
    out = np.full(a.shape, np.nan)
    for j in range(a.shape[1]):
        idx = np.flatnonzero(traded[:, j])
        if len(idx) > k:
            out[idx[k:], j] = a[idx[:-k], j]
    return out


def _own_rollmax(a: np.ndarray, traded: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    for j in range(a.shape[1]):
        idx = np.flatnonzero(traded[:, j])
        if len(idx):
            out[idx, j] = pd.Series(a[idx, j]).rolling(k, min_periods=1).max().to_numpy()
    return out


def market_alert_features(days, codes, ri, ret, mcap, close, high, low, vol, shares, traded, exists,
                          is_kosdaq, is_pref, is_spac) -> dict[str, np.ndarray]:
    """(T 일자 × N 종목) 행렬을 받아 시장경보·단기과열 특징 행렬들을 돌려준다."""
    T, N = ri.shape
    days = pd.DatetimeIndex(days)
    ri = np.asarray(ri, dtype=np.float64)

    # ---- 지수 대용: 시장별 시가총액 가중 수익률 누적 (우선주·스팩 제외)
    mprev = _shift(np.asarray(mcap, dtype=np.float64), 1)
    okw = exists & ~is_pref & ~is_spac & np.isfinite(ret) & (mprev > 0)
    lvl = {}
    for name, m in (("KOSPI", ~is_kosdaq), ("KOSDAQ", is_kosdaq)):
        sel = okw & m
        num = np.where(sel, mprev * np.nan_to_num(ret), 0).sum(1)
        den = np.where(sel, mprev, 0).sum(1)
        lvl[name] = np.cumprod(1 + np.where(den > 0, num / np.maximum(den, 1), 0.0))
    IL = np.where(is_kosdaq, lvl["KOSDAQ"][:, None], lvl["KOSPI"][:, None])

    # ---- 종목 자신의 매매거래일 기준 수익률
    R = {k: _own_shift(ri, traded, k) for k in (1, 2, 3, 4, 5, 14, 15)}
    I = {k: _own_shift(IL, traded, k) for k in (3, 5, 15)}  # noqa: E741
    r = {k: _div(ri, R[k]) - 1 for k in (2, 3, 5, 15)}
    ir = {k: _div(IL, I[k]) - 1 for k in (3, 5, 15)}
    mx15 = _own_rollmax(ri, traded, 15)
    hi15 = traded & (ri >= mx15 * (1 - 1e-9))

    rank_prev = _shift(pd.DataFrame(np.where(exists, mcap, np.nan)).rank(axis=1, ascending=False).to_numpy(), 1)
    top100 = (rank_prev <= 100) & (days >= TOP100_FROM)[:, None]
    idx_ok = lambda k, mult: r[k] >= mult * ir[k]  # noqa: E731  (지수가 하락했으면 자동 충족)
    warn_trig = (((r[3] >= 1.0) & idx_ok(3, 5)) | ((r[5] >= 0.6) & idx_ok(5, 5)) | ((r[15] >= 1.0) & idx_ok(15, 3)))
    warn_trig = warn_trig & hi15 & traded & ~top100
    soft_trig = ((r[5] >= 0.45) | (r[15] >= 0.75)) & hi15 & traded & ~top100
    dA = (r[3] >= 0.45) & idx_ok(3, 5) & hi15 & traded
    dB = (r[5] >= 0.6) & idx_ok(5, 5) & hi15 & traded
    dC = (r[15] >= 1.0) & idx_ok(15, 3) & hi15 & traded
    up1 = traded & (ri > R[1])
    rel_block = (r[5] >= 0.6) | (r[15] >= 1.0) | hi15

    # ---- 시장경보 상태기계: 0 정상, 1 경고예고, 2 경고, 3 경고+위험예고, 4 위험, 5 해제 후 재지정 감시
    S_next = np.zeros((T, N), np.int8)
    C_next = np.zeros((T, N), np.int16)
    EV = np.zeros((T, N), np.int8)  # 1 예고 2 지정 3 경고중 정지 4 위험지정(정지) 5 해제 6 위험예고 7 재지정 8 위험→경고 9 위험중 정지
    P0_next = np.full((T, N), np.nan)
    s = np.zeros(N, np.int8)
    c = np.zeros(N, np.int16)
    win = np.zeros(N, np.int16)
    used = np.zeros(N, bool)
    up_run = np.zeros(N, np.int16)
    P0 = np.full(N, np.nan)
    PR = np.full(N, np.nan)
    for t in range(T):
        tr, ex, x = traded[t], exists[t], ri[t]
        ns, nw, nu = s.copy(), win.copy(), used.copy()
        nc = np.where(tr, c + 1, c).astype(np.int16)
        ev = np.zeros(N, np.int8)
        wt, rb = warn_trig[t], rel_block[t]
        dtg = (dA[t] & (c >= 3)) | (dB[t] & (c >= 5)) | (dC[t] & (c >= 15))
        m = (s == 0) & wt
        ns[m], nw[m], ev[m] = 1, 0, 1
        m1 = (s == 1) & tr
        nw[m1] = win[m1] + 1
        d = m1 & wt
        ns[d], nc[d], nu[d], ev[d] = 2, 0, False, 2
        P0[d] = x[d]
        ns[m1 & ~d & (nw >= 10)] = 0
        m23 = ((s == 2) | (s == 3)) & tr
        h = m23 & (c >= 1) & (r[2][t] >= 0.40) & (x > P0) & ~used
        ev[h], nu[h] = 3, True
        dn = (s == 2) & tr & (c >= 1) & dtg & ~h
        ns[dn], nw[dn], ev[dn] = 3, 0, 6
        rel = (s == 2) & tr & ~dn & ~h & (c + 1 >= 10) & ~rb
        ns[rel], nw[rel], ev[rel] = 5, 0, 5
        PR[rel] = x[rel]
        m3 = (s == 3) & tr & ~h
        nw[m3] = win[m3] + 1
        dd = m3 & dtg & (win >= 0) & ~(ev == 6)
        ns[dd], nc[dd], ev[dd] = 4, 0, 4
        up_run[dd] = 0
        ns[m3 & ~dd & (nw >= 10)] = 2
        m4 = (s == 4) & tr
        up_run[m4 & up1[t]] += 1
        up_run[m4 & ~up1[t]] = 0
        h4 = m4 & (up_run >= 3)
        ev[h4] = 9
        up_run[h4] = 0
        rel4 = m4 & (c + 1 >= 10) & ~rb
        ns[rel4], nc[rel4], nu[rel4], ev[rel4] = 2, 0, False, 8
        P0[rel4] = x[rel4]
        m5 = (s == 5) & tr
        nw[m5] = win[m5] + 1
        rd = m5 & (x > P0) & (x > PR) & (r[2][t] >= 0.40)
        ns[rd], nc[rd], nu[rd], ev[rd] = 2, 0, False, 7
        P0[rd] = x[rd]
        ns[m5 & ~rd & (nw >= 10)] = 0
        ns[top100[t] & (ns > 0)] = 0
        ns[~ex] = 0
        s, win, used = ns, nw, nu
        c = np.where(ns == 0, 0, nc).astype(np.int16)
        S_next[t], C_next[t], EV[t], P0_next[t] = s, c, ev, P0

    # ---- 내일 경보 요건까지 남은 상승률
    need_hi = _div(_own_rollmax(ri, traded, 14), ri) - 1
    need_A = _div(2.0 * R[2], ri) - 1
    need_B = _div(1.6 * R[4], ri) - 1
    need_C = _div(2.0 * R[14], ri) - 1
    warn_need = np.maximum(need_hi, np.fmin(np.fmin(need_A, need_B), need_C))
    halt_need = np.fmax(_div(1.4 * R[1], ri) - 1, _div(P0_next, ri) - 1)
    rel_need = np.where(S_next == 2, need_hi, np.nan)

    # ---- 단기과열종목
    turn = _div(np.where(traded, vol, 0.0), shares)
    volat = np.where(traded, _div(high - low, low), np.nan)
    oh_price = _div(ri, _shift(_roll(ri, 40, "mean", 30), 1))
    oh_turn = _div(_roll(turn, 2, "mean", 2), _shift(_roll(turn, 40, "mean", 30), 1))
    oh_vol = _div(_roll(volat, 2, "mean", 2), _shift(_roll(volat, 40, "mean", 30), 1))
    S_now = _shift(S_next.astype(float), 1)
    oh_trig = ((oh_price >= 1.3) & (oh_turn >= 6) & (oh_vol >= 1.5) & traded & ~(S_now >= 2) & ~(S_next >= 2)
               & ~(S_now == 4) & ~top100)
    parent = np.array([cd[:5] + "0" if (cd.isdigit() and not cd.endswith("0")) else "" for cd in codes])
    pidx = np.searchsorted(codes, parent)
    has_p = (parent != "") & (pidx < N) & (codes[np.minimum(pidx, N - 1)] == parent)
    prem = np.full((T, N), np.nan)
    cj, pj = np.flatnonzero(has_p), pidx[has_p]
    prem[:, cj] = _div(close[:, cj], np.where(close[:, pj] > 0, close[:, pj], np.nan)) - 1
    prem_on = (days >= PREF_PREM_FROM)[:, None]
    OH_next = np.zeros((T, N), np.int8)  # 0 없음 1 예고 2 단일가(일반) 3 단일가(우선주 괴리율)
    OH_left = np.zeros((T, N), np.int8)
    s = np.zeros(N, np.int8)
    c = np.zeros(N, np.int16)
    left = np.zeros(N, np.int16)
    ext = np.zeros(N, bool)
    ref = np.full(N, np.nan)
    pre = np.full(N, np.nan)
    for t in range(1, T):
        tr, x = traded[t], ri[t]
        ns, nl = s.copy(), left.copy()
        run = (s >= 2) & exists[t]
        nl[run] = left[run] - 1
        end = run & (nl <= 0)
        e2 = end & (s == 2) & ~ext & (x >= 1.2 * pre)
        nl[e2], ext[e2] = 3, True
        e3 = end & (s == 3) & (prem[t] > 0.5)
        nl[e3] = 3
        ns[end & ~e2 & ~e3] = 0
        m1 = (s == 1) & tr
        c[m1] += 1
        dsg = m1 & oh_trig[t] & (x > ri[t - 1]) & (x > ref)
        ns[dsg], nl[dsg], ext[dsg] = 2, 3, False
        pre[dsg] = x[dsg]
        ns[m1 & ~dsg & (c >= 10)] = 0
        m0 = (s == 0) & oh_trig[t]
        ns[m0], c[m0] = 1, 0
        ref[m0] = ri[t - 1][m0]
        pm = (ns <= 1) & tr & prem_on[t] & (prem[t] > 0.5)
        ns[pm], nl[pm] = 3, 3
        ns[~exists[t]] = 0
        s, left = ns, nl
        OH_next[t], OH_left[t] = s, np.clip(left, 0, 9)

    n_desig250 = _roll(np.isin(EV, (2, 7)).astype(float), 250, "sum", 1)
    clip = lambda a, lo, hi: np.clip(a, lo, hi)  # noqa: E731
    return {
        "alert_lvl_next": S_next.astype(np.float32),
        "alert_days": np.minimum(C_next, 30).astype(np.float32),
        "alert_event": EV.astype(np.float32),
        "halt_pred": np.isin(EV, (3, 4, 9)).astype(np.float32),
        "warn_need": clip(warn_need, -0.5, 2.0),
        "halt_need": clip(np.where(np.isin(S_next, (2, 3)), halt_need, np.nan), -0.5, 2.0),
        "rel_need": clip(rel_need, -0.5, 2.0),
        "soft_trig": soft_trig.astype(np.float32),
        "n_desig250": n_desig250,
        "oh_price": clip(oh_price, 0, 10), "oh_turn": clip(oh_turn, 0, 200), "oh_vol": clip(oh_vol, 0, 50),
        "oh_lvl_next": OH_next.astype(np.float32), "oh_left": OH_left.astype(np.float32),
        "pref_prem": clip(np.log1p(np.where(prem > -0.99, prem, np.nan)), -3, 6),
    }
