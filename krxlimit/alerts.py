"""KRX 시장경보(투자주의·경고·위험)와 단기과열종목 지정을 일별 시세로 재구성한 특징.

KIND 공시 규정 문구를 따라 '오늘 장 마감 기준으로 내일 어떤 조치가 붙는가'를 상태기계로 추정한다.
- 투자경고: 가격 요건 예고(10거래일 관찰) → 15일 최고가·지수 배수 요건으로 지정 → (2일 +40% & 지정 전 종가 상회 시 1일 매매정지) → 위험예고 → 투자위험(지정일 정지)
  → 10거래일 뒤 해제 요건 → 해제 후 10일 재지정 감시
- 단기과열: 가격(40일 평균 대비 130%)·회전율(6배)·변동성(1.5배) 동시 충족 → 예고 → 3일 단일가매매(연장 1회),
  우선주는 보통주 대비 괴리율 50% 초과 시 바로 지정
- 날짜는 '해당 종목의 매매거래일' 기준. 행 t 의 값은 t일 종가까지의 정보만 사용.
시가총액 상위 100 종목 예외(2025-12-29~), 우선주 괴리율 기준(2020-09-28~) 같은 규정 변경일도 반영.
※ 실제 지정 이력 데이터가 아니라 규정을 가격으로 재현한 추정치다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _div, _roll, _shift  # noqa: F401 (features 모듈의 공통 함수 재사용)

ALERT_FEATURES = ["al_state_next", "al_nd", "al_event", "al_halt_next", "al_halt_used", "al_trip_halt",
                  "al_need_halt", "al_need_notice", "al_lim_hi15", "n_desig250"]
OVERHEAT_FEATURES = ["oh_price", "oh_turn", "oh_vol", "oh_lvl_next", "oh_left", "pref_prem"]
TOP100_FROM = "2025-12-29"  # 시가총액 상위 100 종목 시장경보 예외 시행일
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

    # ---- 종목 자신의 매매거래일 기준 수익률 (v4: 실제 다음날 정지와 대조해 검증한 규칙)
    R1, R2, R4, R14 = (_own_shift(ri, traded, k) for k in (1, 2, 4, 14))
    r = {1: _div(ri, R1) - 1, 2: _div(ri, R2) - 1}
    r.update({k: _div(ri, _own_shift(ri, traded, k)) - 1 for k in (3, 5, 15)})
    ir = {k: _div(IL, _own_shift(IL, traded, k)) - 1 for k in (3, 5, 15)}
    mx15 = _own_rollmax(ri, traded, 15)
    mx14 = _own_rollmax(ri, traded, 14)
    hi15 = traded & (ri >= mx15 * (1 - 1e-6))

    rank_prev = _shift(pd.DataFrame(np.where(exists, mcap, np.nan)).rank(axis=1, ascending=False).to_numpy(), 1)
    top100 = (rank_prev <= 100) & (days >= TOP100_FROM)[:, None]
    ge = lambda a, th: np.nan_to_num(a, nan=-9) >= th  # noqa: E731
    idx_ok = lambda k, m: ~(np.nan_to_num(r[k], nan=-9) < m * np.nan_to_num(ir[k], nan=0))  # noqa: E731
    # 투자경고 지정예고는 가격 요건만, 지정은 15일 최고가·지수 대비 배수까지
    w_notice = traded & ~top100 & (ge(r[3], 1.0) | ge(r[5], 0.6) | ge(r[15], 1.0))
    w_desig = traded & ~top100 & hi15 & ((ge(r[3], 1.0) & idx_ok(3, 5)) | (ge(r[5], 0.6) & idx_ok(5, 5))
                                         | (ge(r[15], 1.0) & idx_ok(15, 3)))
    dn3, dn5, dn15 = traded & ge(r[3], 0.45), traded & ge(r[5], 0.6), traded & ge(r[15], 1.0)
    dd3, dd5, dd15 = dn3 & hi15 & idx_ok(3, 5), dn5 & hi15 & idx_ok(5, 5), dn15 & hi15 & idx_ok(15, 3)
    DLAG, HND = -1, 1  # 위험 판단 창은 지정일 D-1 부터, 경고 중 정지는 D+1 부터 판단

    def dcond(a3, a5, a15, t, ndv):
        return (a3[t] & (ndv - 3 >= DLAG)) | (a5[t] & (ndv - 5 >= DLAG)) | (a15[t] & (ndv - 15 >= DLAG))

    w_block = ge(r[5], 0.45) | ge(r[15], 0.75) | hi15   # 경고 해제 불가 요건
    d_block = ge(r[5], 0.6) | ge(r[15], 1.0) | hi15     # 위험 해제 불가 요건
    up1 = traded & (np.nan_to_num(r[1], nan=0) > 0)
    r2 = np.nan_to_num(r[2], nan=-9)

    # ---- 상태기계: 0 정상, 1 경고예고, 2 경고, 3 경고+위험예고, 4 위험, 5 해제 후 재지정 감시
    S = np.zeros((T, N), np.int8)       # 오늘 판단 후 다음 거래일에 적용될 상태
    ND = np.zeros((T, N), np.int16)     # 지정일 D(=0) 이후 종목 매매거래일 수
    HALT = np.zeros((T, N), np.int8)    # 내일 정지 예측: 1 경고 2일+40% 2 위험 지정 3 위험 3일 연속 상승
    EVT = np.zeros((T, N), np.int8)     # 1 경고예고 2 경고지정 3 위험예고 4 위험지정 5 경고해제 6 재지정 7 위험해제
    USED = np.zeros((T, N), bool)
    P0A = np.full((T, N), np.nan)
    s = np.zeros(N, np.int8)
    nd = np.full(N, -1, np.int16)
    win = np.zeros(N, np.int16)
    dwin = np.full(N, -1, np.int16)
    used = np.zeros(N, bool)
    uprun = np.zeros(N, np.int16)
    P0 = np.full(N, np.nan)
    PR = np.full(N, np.nan)
    for t in range(T):
        tr, x = traded[t], ri[t]
        ev = np.zeros(N, np.int8)
        hl = np.zeros(N, np.int8)
        nd = np.where(tr & (s >= 2) & (s <= 4), nd + 1, nd).astype(np.int16)
        win = np.where(tr & ((s == 1) | (s == 5)), win + 1, win).astype(np.int16)
        dwin = np.where(tr & (dwin >= 0), dwin + 1, dwin).astype(np.int16)
        ns = s.copy()
        m1 = (s == 1) & tr                               # 예고 기간(10일) 중 지정 판단
        d = m1 & w_desig[t]
        ns[d], ev[d] = 2, 2
        P0[d], used[d], uprun[d], nd[d], dwin[d] = x[d], False, 0, -1, -1
        ns[m1 & ~d & (win >= 10)] = 0
        win[m1 & ~d & w_notice[t]] = 0                   # 예고 요건 재충족 → 창 다시 시작
        m0 = (s == 0) & w_notice[t]
        ns[m0], ev[m0], win[m0] = 1, 1, 0
        m23 = ((s == 2) | (s == 3)) & tr & (nd >= 0)     # 경고 중 2일 +40% & 지정 전 종가 상회 → 1일 정지(1회)
        h = m23 & (nd >= HND) & (r2[t] >= 0.40) & (x > P0) & ~used
        hl[h], used[h] = 1, True
        m3 = (s == 3) & tr & (dwin >= 1)                 # 위험 지정 판단
        dd = m3 & dcond(dd3, dd5, dd15, t, nd)
        ns[dd], ev[dd], hl[dd] = 4, 4, 2
        nd[dd], uprun[dd] = 0, 0
        m3x = m3 & ~dd & (dwin >= 10)
        ns[m3x], dwin[m3x] = 2, -1
        dn = (s == 2) & tr & (nd >= 0) & dcond(dn3, dn5, dn15, t, nd)   # 위험 예고 (정지 판단일과 겹칠 수 있음)
        ns[dn], ev[dn], dwin[dn] = 3, 3, 0
        rel = (s == 2) & tr & ~dn & ~h & (nd >= 9) & ~w_block[t]       # 경고 해제
        ns[rel], ev[rel], win[rel] = 5, 5, 0
        PR[rel] = x[rel]
        m4 = (s == 4) & tr                               # 위험: 3일 연속 상승 → 1일 정지
        uprun[m4 & up1[t]] += 1
        uprun[m4 & ~up1[t]] = 0
        h4 = m4 & (uprun >= 3)
        hl[h4 & (hl == 0)] = 3
        uprun[h4] = 0
        rel4 = m4 & (nd >= 10) & ~d_block[t]             # 위험 해제 → 경고로
        ns[rel4], ev[rel4] = 2, 7
        P0[rel4], used[rel4], nd[rel4], dwin[rel4] = x[rel4], False, -1, -1
        m5 = (s == 5) & tr                               # 해제 후 재지정 감시
        rd = m5 & (win >= 2) & (x > P0) & (x > PR) & (r2[t] >= 0.40)
        ns[rd], ev[rd] = 2, 6
        P0[rd], used[rd], nd[rd], dwin[rd] = x[rd], False, -1, -1
        ns[m5 & ~rd & (win >= 10)] = 0
        rn5 = m5 & ~rd & w_notice[t]
        ns[rn5], ev[rn5], win[rn5] = 1, 1, 0
        ns[top100[t] & (ns > 0)] = 0
        ns[~exists[t]] = 0
        s = ns
        nd = np.where(s == 0, -1, nd).astype(np.int16)
        dwin = np.where(s == 3, dwin, -1).astype(np.int16)
        S[t], ND[t], HALT[t], EVT[t], USED[t], P0A[t] = s, nd, hl, ev, used, P0

    # ---- 내일 종가가 상한가라면 어떤 규정에 걸리는가 (t일까지 정보만)
    lim = 1.3 * ri
    in_warn = (S == 2) | (S == 3)
    trip_halt = in_warn & ~USED & (ND + 1 >= HND) & (_div(lim, R1) >= 1.4) & (lim > P0A)
    need_halt = np.where(in_warn & ~USED, np.fmax(_div(1.4 * R1, ri), _div(P0A, ri)) - 1, np.nan)
    need_notice = np.fmin(np.fmin(_div(2.0 * R2, ri), _div(1.6 * R4, ri)), _div(2.0 * R14, ri)) - 1
    lim_hi15 = lim >= mx14
    S_next = S  # 아래 단기과열 블록에서 사용

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

    n_desig250 = _roll(np.isin(EVT, (2, 6)).astype(float), 250, "sum", 1)
    clip = lambda a, lo, hi: np.clip(a, lo, hi)  # noqa: E731
    return {
        "al_state_next": S.astype(np.float32),
        "al_nd": np.where(S >= 2, ND, -1).astype(np.float32),
        "al_event": EVT.astype(np.float32),
        "al_halt_next": HALT.astype(np.float32),
        "al_halt_used": (USED & in_warn).astype(np.float32),
        "al_trip_halt": trip_halt.astype(np.float32),
        "al_need_halt": clip(need_halt, -0.5, 2.0),
        "al_need_notice": clip(need_notice, -0.5, 2.0),
        "al_lim_hi15": lim_hi15.astype(np.float32),
        "n_desig250": n_desig250,
        "oh_price": clip(oh_price, 0, 10), "oh_turn": clip(oh_turn, 0, 200), "oh_vol": clip(oh_vol, 0, 50),
        "oh_lvl_next": OH_next.astype(np.float32), "oh_left": OH_left.astype(np.float32),
        "pref_prem": clip(np.log1p(np.where(prem > -0.99, prem, np.nan)), -3, 6),
    }
