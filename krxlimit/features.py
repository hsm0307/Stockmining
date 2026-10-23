"""종목-일 단위 특징(feature)과 다음 거래일 라벨 생성.

모든 특징은 '당일 장 마감 후' 알 수 있는 정보만 사용한다(룩어헤드 금지).
라벨은 다음 거래일(t+1)의 상한가/하한가 여부.
계산은 (일자 × 종목) 행렬로 한 번에 처리해 속도를 확보한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# 모델에 들어가는 특징 그룹 (ablation 실험 시 그룹 단위로 끄고 켤 수 있음)
FEATURE_GROUPS: dict[str, list[str]] = {
    "candle": ["ret1", "gap", "body", "ushadow", "lshadow", "hl_range", "close_pos", "dist_up", "high_ret"],
    "momentum": ["ret5", "ret20", "ret60", "ret120", "ma5_gap", "ma20_gap", "ma60_gap", "ma120_gap",
                 "hi250_pos", "lo250_pos", "box60"],
    "limit_hist": ["up_close", "up_touch", "up_jeom", "up_fail", "down_close", "down_touch", "up_streak",
                   "down_streak", "n_up20", "n_up60", "n_up250", "n_down20", "n_down60", "n_uptouch20",
                   "days_since_up"],
    "volume": ["log_value", "turnover", "turn20", "vr1_20", "vr5_60", "value_rank", "vol_spike60",
               "low_vol_share60"],
    "volatility": ["volat20", "volat60"],
    "accumulation": ["cmf20", "obv_slope20", "upvol_ratio20", "shadow_test60"],
    "size_type": ["log_price", "penny", "log_mcap", "mcap_rank", "is_kosdaq", "is_pref", "is_spac",
                  "is_reit", "is_mgmt", "is_caution", "log_age"],
    "market": ["mkt_ret", "mkt_ret5", "mkt_n_up", "mkt_n_up5", "mkt_n_down", "mkt_breadth", "all_n_up"],
    "peer": ["peer_ret", "peer_n_up", "peer_max_ret", "peer_corr"],
    "calendar": ["dow"],
    # ---- v2 후보 특징 (improve 실험에서 그룹 단위로 효과 검증)
    "rel_strength": ["turn_rank_in_up", "value_rank_in_up", "mcap_rank_in_up"],  # 오늘 상한가 종목끼리 비교한 강도
    "pref_link": ["parent_ret1", "parent_up_close", "child_max_ret1", "child_up_close"],  # 우선주 ↔ 보통주
    "stock_hist": ["hist_cont_rate", "hist_n_cont500"],  # 이 종목의 과거 연상 성향
    "market2": ["mkt_n_fail", "all_n_up_prev", "mkt_jeom_ratio"],  # 시장 과열/실패 분위기
    "peer2": ["peer_n_touch"],  # 동조 종목의 상한가 터치 수
    # 시장경보(투자주의·경고·위험) 상태기계 재구성 — krxlimit/alerts.py
    "market_alert": ["al_state_next", "al_nd", "al_event", "al_halt_next", "al_halt_used", "al_trip_halt",
                     "al_need_halt", "al_need_notice", "al_lim_hi15", "n_desig250"],
    "overheat": ["oh_price", "oh_turn", "oh_vol", "oh_lvl_next", "oh_left", "pref_prem"],  # 단기과열종목 요건
}
V1_GROUPS = ["candle", "momentum", "limit_hist", "volume", "volatility", "accumulation", "size_type", "market",
             "peer", "calendar"]  # 1차 모델(v1) 특징 그룹
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
AGE_CAP = 400       # 상장 경과 거래일 상한
LIVE_LOOKBACK = 560  # 실시간 예측 시 사용하는 최근 거래일 수 (가장 긴 창 500일 + 여유)
NOLIMIT_DAYS = 7    # 최근 이만큼 거래일 안에 '가격제한 없는 거래일'이 있으면 정리매매 등으로 보고 예측 대상에서 뺀다
UNIVERSE_COLS = ["is_nolimit"]  # 모델 특징이 아니라 예측 대상 필터(model.universe_mask)용 표시
LABELS = ["y_up", "y_down", "y_uptouch", "next_ret", "next_open_ret", "next_tidx"]
ID_COLS = ["date", "code", "name", "market", "close", "upper", "tidx"]


class Wide:
    """long 패널 ↔ (T 일자 × N 종목) numpy 행렬 변환기."""

    def __init__(self, panel: pd.DataFrame):
        self.days = np.sort(panel["date"].unique())
        codes = panel["code"].astype(str).to_numpy()
        self.codes = np.unique(codes)
        self.ti = np.searchsorted(self.days, panel["date"].to_numpy())
        self.ci = np.searchsorted(self.codes, codes)
        self.shape = (len(self.days), len(self.codes))

    def wide(self, values, fill=np.nan, dtype=np.float32) -> np.ndarray:
        arr = np.full(self.shape, fill, dtype=dtype)
        arr[self.ti, self.ci] = np.asarray(values, dtype=dtype)
        return arr

    def take(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.ti, self.ci]


class _Gather:
    """(T×N) 행렬 특징을 계산하는 즉시 필요한 (일자, 종목) 위치만 뽑아 float32 로 보관 → 메모리 절약."""

    def __init__(self, ti, ci):
        self.ti, self.ci, self.data = ti, ci, {}

    def __setitem__(self, name, arr):
        self.data[name] = np.asarray(arr)[self.ti, self.ci].astype(np.float32)

    def __getitem__(self, name):
        return self.data[name]

    def update(self, d):
        for k, v in d.items():
            self[k] = v


def _roll(arr, win, fn="mean", min_periods=None):
    r = pd.DataFrame(arr).rolling(win, min_periods=min_periods or win)
    return getattr(r, fn)().to_numpy()


def _shift(arr, k):
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = arr[:-k]
    elif k < 0:
        out[:k] = arr[-k:]
    else:
        out[:] = arr
    return out


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _xs_rank(arr, mask):
    """일자별 횡단면 백분위 순위 (mask 밖은 NaN)."""
    a = np.where(mask, arr, np.nan)
    return pd.DataFrame(a).rank(axis=1, pct=True).to_numpy()


def peer_features(ret: np.ndarray, up: np.ndarray, days: np.ndarray, k: int = 10,
                  lookback: int = 120, min_obs: int = 80, touch: np.ndarray | None = None,
                  exists: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """'테마' 대용 특징: 직전 lookback 거래일 수익률 상관이 가장 높은 k개 종목(동조 종목)의 당일 움직임.

    매월 첫 거래일에 그 이전 데이터로만 동조 종목을 다시 고르므로 미래 정보가 섞이지 않는다.
    우선주는 대개 보통주가, 테마주는 같은 테마 종목이 동조 종목으로 잡힌다.
    """
    T, N = ret.shape
    names = ["peer_ret", "peer_n_up", "peer_max_ret", "peer_corr"] + (["peer_n_touch"] if touch is not None else [])
    out = {n: np.full((T, N), np.nan, dtype=np.float32) for n in names}
    month = pd.DatetimeIndex(days).to_period("M")
    starts = np.flatnonzero(np.r_[True, month[1:] != month[:-1]])
    ends = np.r_[starts[1:], T]
    for t0, t1 in zip(starts, ends):
        if t0 < lookback:
            continue
        R = ret[t0 - lookback:t0]
        # 직전 거래일에 상장돼 있던 종목 (그날 거래정지였어도 포함)
        alive = exists[t0 - 1] if exists is not None else ~np.isnan(ret[t0 - 1])
        ok = (np.sum(~np.isnan(R), axis=0) >= min_obs) & alive
        idx = np.flatnonzero(ok)
        if len(idx) <= k + 1:
            continue
        Z = np.nan_to_num(R[:, idx]).astype(np.float32)
        Z -= Z.mean(axis=0)
        sd = Z.std(axis=0)
        good = sd > 0
        idx, Z = idx[good], Z[:, good] / sd[good]
        C = (Z.T @ Z) / Z.shape[0]
        np.fill_diagonal(C, -np.inf)
        top = np.argpartition(-C, k, axis=1)[:, :k]
        corr_top = np.take_along_axis(C, top, axis=1)
        Rm = np.nan_to_num(ret[t0:t1][:, idx]).astype(np.float32)
        Um = np.nan_to_num(up[t0:t1][:, idx]).astype(np.float32)
        G = Rm[:, top]  # (D, n, k)
        out["peer_ret"][t0:t1, idx] = G.mean(axis=2)
        out["peer_max_ret"][t0:t1, idx] = G.max(axis=2)
        out["peer_n_up"][t0:t1, idx] = Um[:, top].sum(axis=2)
        out["peer_corr"][t0:t1, idx] = corr_top.mean(axis=1)
        if touch is not None:
            Tm = np.nan_to_num(touch[t0:t1][:, idx]).astype(np.float32)
            out["peer_n_touch"][t0:t1, idx] = Tm[:, top].sum(axis=2)
    return out


def market_daily(panel: pd.DataFrame) -> pd.DataFrame:
    """시장별 일간 집계: 동일가중 수익률, 상/하한가 종목 수, 상승 비율."""
    p = panel[panel["traded"] & panel["ret"].notna()]
    g = p.groupby(["date", "market"], observed=True)
    out = g.agg(n=("code", "size"), ew_ret=("ret", "mean"), n_up=("up_close", "sum"),
                n_down=("down_close", "sum"), n_up_jeom=("up_jeom", "sum"), n_down_jeom=("down_jeom", "sum"),
                n_up_fail=("up_fail", "sum"), n_down_fail=("down_fail", "sum"),
                breadth=("ret", lambda s: float((s > 0).mean())))
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out


def build_features(panel: pd.DataFrame, save: bool = True, peer: bool = True,
                   keep_from: str | None = None) -> pd.DataFrame:
    """특징 테이블 생성. keep_from(YYYY-MM-DD) 이전 행은 계산에만 쓰고 결과에서 뺀다(실시간 예측용)."""
    W = Wide(panel)
    T, N = W.shape
    w = lambda col, dtype=np.float32: W.wide(panel[col].to_numpy(), dtype=dtype)  # noqa: E731

    close, open_, high, low, base = w("close"), w("open"), w("high"), w("low"), w("base")
    upper = w("upper")
    vol, value, mcap, shares = w("volume"), w("value"), w("mktcap"), w("shares")
    ret = w("ret")
    traded = W.wide(panel["traded"].to_numpy(), fill=False, dtype=bool)
    exists = W.wide(np.ones(len(panel)), fill=0, dtype=np.int8).astype(bool)
    ri = pd.DataFrame(w("ri")).ffill(limit=5).to_numpy()
    flags = {c: W.wide(panel[c].to_numpy(), fill=0, dtype=np.float32) for c in
             ["up_close", "down_close", "up_touch", "down_touch", "up_jeom", "up_fail"]}
    is_kosdaq_w = W.wide(panel["is_kosdaq"].to_numpy(), fill=False, dtype=bool)

    keep = (panel["traded"] & ~panel["is_first_day"] & (panel["base"] > 0)).to_numpy()
    if keep_from is not None:
        keep = keep & (panel["date"] >= pd.Timestamp(keep_from)).to_numpy()
    f = _Gather(W.ti[keep], W.ci[keep])
    # ---- 캔들
    t_ret = np.where(traded, ret, np.nan)
    f["ret1"] = ret
    f["gap"] = np.where(traded, _div(open_, base) - 1, np.nan)
    f["body"] = np.where(traded, _div(close - open_, base), np.nan)
    f["ushadow"] = np.where(traded, _div(high - np.maximum(open_, close), base), np.nan)
    f["lshadow"] = np.where(traded, _div(np.minimum(open_, close) - low, base), np.nan)
    f["hl_range"] = np.where(traded, _div(high - low, base), np.nan)
    f["close_pos"] = np.where(traded, _div(close - low, high - low), np.nan)
    f["dist_up"] = _div(upper - close, base)
    high_ret = np.where(traded, _div(high, base) - 1, np.nan)
    f["high_ret"] = high_ret

    # ---- 모멘텀 / 가격 위치 (수익률지수 ri 사용 → 분할 등 영향 없음)
    for k in (5, 20, 60, 120):
        f[f"ret{k}"] = _div(ri, _shift(ri, k)) - 1
        f[f"ma{k}_gap"] = _div(ri, _roll(ri, k, "mean")) - 1
    f["hi250_pos"] = _div(ri, _roll(ri, 250, "max", 60))
    f["lo250_pos"] = _div(ri, _roll(ri, 250, "min", 60)) - 1
    f["box60"] = _div(_roll(ri, 60, "max"), _roll(ri, 60, "min")) - 1

    # ---- 상/하한가 이력
    for c in ["up_close", "up_touch", "up_jeom", "up_fail", "down_close", "down_touch"]:
        f[c] = np.where(exists, flags[c], np.nan)
    f["up_streak"] = w("up_streak")
    f["down_streak"] = w("down_streak")
    for k in (20, 60, 250):
        f[f"n_up{k}"] = _roll(flags["up_close"], k, "sum", 1)
    for k in (20, 60):
        f[f"n_down{k}"] = _roll(flags["down_close"], k, "sum", 1)
    f["n_uptouch20"] = _roll(flags["up_touch"], 20, "sum", 1)
    rows = np.arange(T)[:, None]
    last_up = np.maximum.accumulate(np.where(flags["up_close"] > 0, rows, -1), axis=0)
    f["days_since_up"] = np.where(last_up >= 0, np.minimum(rows - last_up, 260), 260).astype(np.float64)

    # ---- 거래량 / 유동성
    tvol = np.where(traded, vol, 0.0)
    tvol = np.where(exists, tvol, np.nan)
    f["log_value"] = np.log1p(np.where(exists, value, np.nan))
    turnover = _div(tvol, shares)
    f["turnover"] = turnover
    f["turn20"] = _roll(turnover, 20, "mean", 10)
    f["vr1_20"] = _div(tvol, _shift(_roll(tvol, 20, "mean", 10), 1))
    f["vr5_60"] = _div(_roll(tvol, 5, "mean"), _roll(tvol, 60, "mean", 30))
    f["value_rank"] = _xs_rank(value, traded)
    med60 = _shift(_roll(tvol, 60, "median", 30), 1)
    f["vol_spike60"] = _roll(np.where(exists, (tvol > 3 * med60).astype(float), np.nan), 60, "sum", 20)
    f["low_vol_share60"] = _roll(np.where(exists, (tvol < 0.5 * med60).astype(float), np.nan), 60, "mean", 20)

    # ---- 변동성
    f["volat20"] = _roll(t_ret, 20, "std", 10)
    f["volat60"] = _roll(t_ret, 60, "std", 30)

    # ---- 매집 흔적 대용 지표
    clv = np.where(traded, _div((close - low) - (high - close), high - low), np.nan)
    clv = np.where(traded & np.isnan(clv), 0.0, clv)
    f["cmf20"] = _div(_roll(np.nan_to_num(clv) * np.nan_to_num(tvol), 20, "sum", 10), _roll(tvol, 20, "sum", 10))
    signed_vol = np.where(exists, np.sign(np.nan_to_num(t_ret)) * np.nan_to_num(tvol), np.nan)
    f["obv_slope20"] = _div(_roll(signed_vol, 20, "sum"), 20 * _roll(tvol, 60, "mean", 30))  # OBV 20일 변화량
    f["upvol_ratio20"] = _div(_roll(np.where(t_ret > 0, tvol, 0.0), 20, "sum", 10), _roll(tvol, 20, "sum", 10))
    test_day = np.where(exists, ((high_ret >= 0.10) & (t_ret <= 0.03)).astype(float), np.nan)
    f["shadow_test60"] = _roll(test_day, 60, "sum", 20)

    # ---- 규모 / 종목 유형
    f["log_price"] = np.log(np.where(close > 0, close, np.nan))
    f["penny"] = np.where(exists, (close < 1000).astype(float), np.nan)
    f["log_mcap"] = np.log(np.where(mcap > 0, mcap, np.nan))
    f["mcap_rank"] = _xs_rank(mcap, exists)
    for c in ["is_kosdaq", "is_pref", "is_spac", "is_reit", "is_mgmt", "is_caution"]:
        f[c] = np.where(exists, W.wide(panel[c].to_numpy(), fill=0, dtype=np.float32), np.nan)
    # 정리매매·재상장 등 가격제한(±30%) 없이 거래되는 종목은 상·하한가가 정의상 불가능하다.
    # 상장 첫날이 아닌데 가격제한이 적용되지 않은 거래일이 최근 NOLIMIT_DAYS 거래일(당일 포함) 안에 있으면 1.
    # 당일 종가로 판정하므로 예측 시점(D 장 마감)에 알 수 있는 정보다.
    nolim = W.wide((panel["traded"] & ~panel["limit_valid"] & ~panel["is_first_day"]).to_numpy(),
                   fill=0, dtype=np.float32)
    f["is_nolimit"] = np.where(exists, (_roll(nolim, NOLIMIT_DAYS, "max", 1) > 0).astype(np.float32), np.nan)
    age = np.minimum(np.cumsum(exists, axis=0), AGE_CAP)  # 상한을 둬야 실시간 예측(최근 구간만 계산)과 값이 같다
    f["log_age"] = np.where(exists, np.log1p(age), np.nan)

    # ---- 시장 상황 (같은 시장의 당일 지표)
    md = market_daily(panel)
    day_pos = np.searchsorted(W.days, md["date"].to_numpy())
    mk = {}
    for mname in ["KOSPI", "KOSDAQ"]:
        sub = md[md["market"] == mname]
        arrs = {}
        for c in ["ew_ret", "n_up", "n_down", "breadth", "n_up_fail", "n_up_jeom"]:
            a = np.full(T, np.nan)
            a[day_pos[md["market"].to_numpy() == mname]] = sub[c].to_numpy()
            arrs[c] = a
        arrs["ew_ret5"] = pd.Series(arrs["ew_ret"]).rolling(5, min_periods=1).sum().to_numpy()
        arrs["n_up5"] = pd.Series(arrs["n_up"]).rolling(5, min_periods=1).mean().to_numpy()
        mk[mname] = arrs
    both_up = np.nan_to_num(mk["KOSPI"]["n_up"]) + np.nan_to_num(mk["KOSDAQ"]["n_up"])
    sel = lambda c: np.where(is_kosdaq_w, mk["KOSDAQ"][c][:, None], mk["KOSPI"][c][:, None])  # noqa: E731
    f["mkt_ret"], f["mkt_ret5"] = sel("ew_ret"), sel("ew_ret5")
    f["mkt_n_up"], f["mkt_n_up5"] = sel("n_up"), sel("n_up5")
    f["mkt_n_down"], f["mkt_breadth"] = sel("n_down"), sel("breadth")
    f["all_n_up"] = np.broadcast_to(both_up[:, None], (T, N))
    # v2: 시장 분위기 — 상한가 터치 후 실패 수, 전일 전체 상한가 수, 상한가 중 점상 비율
    f["mkt_n_fail"] = sel("n_up_fail")
    f["all_n_up_prev"] = np.broadcast_to(_shift(both_up[:, None], 1), (T, N))
    f["mkt_jeom_ratio"] = _div(sel("n_up_jeom"), sel("n_up"))

    # ---- 동조 종목(테마 대용)
    if peer:
        f.update(peer_features(t_ret, flags["up_close"], W.days, touch=flags["up_touch"], exists=exists))
    else:
        for c in FEATURE_GROUPS["peer"] + FEATURE_GROUPS["peer2"]:
            f[c] = np.full((T, N), np.nan, dtype=np.float32)

    # ---- 시장경보·단기과열 지정 재구성 (규정 기반 추정, 행 t 는 t일 종가까지만 사용)
    from .alerts import market_alert_features
    f.update(market_alert_features(
        W.days, W.codes, ri, ret, mcap, close, high, low, vol, shares, traded, exists, is_kosdaq_w,
        W.wide(panel["is_pref"].to_numpy(), fill=False, dtype=bool),
        W.wide(panel["is_spac"].to_numpy(), fill=False, dtype=bool)))

    # ---- v2: 오늘 상한가 종목끼리의 상대 강도 (회전율·거래대금이 낮을수록 매물이 잠긴 '강한' 상한가)
    up_today = flags["up_close"] > 0
    f["turn_rank_in_up"] = _xs_rank(turnover, up_today)
    f["value_rank_in_up"] = _xs_rank(value, up_today)
    f["mcap_rank_in_up"] = _xs_rank(mcap, up_today)

    # ---- v2: 우선주 ↔ 보통주 연결 (우선주 코드 끝자리를 0으로 바꾸면 보통주)
    codes = W.codes
    parent = np.array([c[:5] + "0" if (c.isdigit() and not c.endswith("0")) else "" for c in codes])
    pidx = np.searchsorted(codes, parent)
    has_parent = (parent != "") & (pidx < N) & (codes[np.minimum(pidx, N - 1)] == parent)
    child_cols, parent_cols = np.flatnonzero(has_parent), pidx[has_parent]
    up_f = flags["up_close"]
    parent_ret1 = np.full((T, N), np.nan, dtype=np.float32)
    parent_up = np.full((T, N), np.nan, dtype=np.float32)
    parent_ret1[:, child_cols] = t_ret[:, parent_cols]
    parent_up[:, child_cols] = np.where(exists[:, parent_cols], up_f[:, parent_cols], np.nan)
    child_max = np.full((T, N), np.nan, dtype=np.float32)
    child_up = np.full((T, N), np.nan, dtype=np.float32)
    for cj, pj in zip(child_cols, parent_cols):
        child_max[:, pj] = np.fmax(child_max[:, pj], t_ret[:, cj])
        child_up[:, pj] = np.fmax(child_up[:, pj], np.where(exists[:, cj], up_f[:, cj], np.nan))
    f["parent_ret1"], f["parent_up_close"] = parent_ret1, parent_up
    f["child_max_ret1"], f["child_up_close"] = child_max, child_up

    # ---- v2: 이 종목의 과거 연상 성향 (최근 500거래일: 상한가 다음 날도 상한가였던 비율)
    cont = up_f * _shift(up_f, 1)  # 오늘과 전날 모두 상한가
    n_cont = _roll(np.nan_to_num(cont), 500, "sum", 1)
    n_up_prev = _shift(_roll(up_f, 500, "sum", 1), 1)  # 결과를 아는 과거 상한가 수 (오늘 제외)
    f["hist_n_cont500"] = np.where(exists, n_cont, np.nan)
    f["hist_cont_rate"] = np.where(exists & (n_up_prev > 0), _div(n_cont, n_up_prev), np.nan)

    f["dow"] = np.broadcast_to(pd.DatetimeIndex(W.days).dayofweek.to_numpy()[:, None], (T, N)).astype(float)

    # ---- 라벨: 다음 거래일(t+1)
    # 다음 날 거래정지(시장경보 등)·가격제한 무효(정리매매)·상장폐지로 상/하한가가 불가능한 종목도 '실패(0)'로 둔다.
    # 예측 시점에는 이런 사정을 모르므로, 후보에서 빼 버리면 백테스트가 실전보다 좋아 보인다.
    # 데이터 마지막 날만 결과를 몰라서 NaN.
    has_next = np.zeros((T, 1), dtype=bool)
    has_next[:-1] = True
    nxt_exists = _shift(exists.astype(float), -1) > 0
    nxt_traded = _shift(traded.astype(float), -1) > 0
    for lab, flag in [("y_up", "up_close"), ("y_down", "down_close"), ("y_uptouch", "up_touch")]:
        f[lab] = np.where(has_next, np.nan_to_num(_shift(flags[flag], -1)), np.nan)
    f["next_ret"] = np.where(nxt_exists, _shift(ret, -1), np.nan)  # 거래정지일은 0, 정리매매는 실제 손익
    f["next_open_ret"] = np.where(nxt_traded, _shift(_div(open_, base) - 1, -1), np.nan)

    # ---- long 형태로 모으기
    out = panel.loc[keep, ID_COLS].reset_index(drop=True)
    out["code"] = out["code"].astype(str)
    cols = ALL_FEATURES + UNIVERSE_COLS + ["y_up", "y_down", "y_uptouch", "next_ret", "next_open_ret"]
    out = pd.concat([out, pd.DataFrame({name: f.data.pop(name) for name in cols})], axis=1)
    out["next_tidx"] = (out["tidx"] + 1).astype(np.int32)

    if save:
        config.ensure_dirs()
        out.to_parquet(config.FEATURES_PATH, index=False)
        print(f"특징 저장: {config.FEATURES_PATH} ({len(out):,}행 × {len(ALL_FEATURES)}개 특징)")
    return out


def load_features() -> pd.DataFrame:
    if not config.FEATURES_PATH.exists():
        raise FileNotFoundError("특징 파일이 없습니다. 먼저 `python -m krxlimit features` 를 실행하세요.")
    return pd.read_parquet(config.FEATURES_PATH)
