"""목표 4 보강: 연속 상한가·하한가 종목의 최고점/최저점 대비 가격 변화.

연속 상한가(연상)
- 최고점: 연상 시작일 ~ 종료 후 20거래일 사이 최고 종가(수익률지수 기준 → 분할·증자 영향 없음)
- 최고점 이후 1주·1달·3달·6달·1년 시점 가격의 최고점 대비 변화, 1년 내 최대 낙폭
- 상승분 반납률 = (최고점 − 1년 후 가격) ÷ (최고점 − 연상 직전 가격). 100% 이상이면 연상 전보다 낮아짐
- 1년 안에 최고점을 다시 넘은 비율
연속 하한가(연하): 위와 대칭(최저점, 반등률, 1년 내 최대 반등, 하락 전 가격 회복 비율)
상장폐지 종목은 마지막 거래 가격으로 확정, 데이터 끝을 넘는 구간은 제외(NaN).
종료 후 20거래일 창이 데이터 끝에 잘리는 최근 연속 구간은 극점을 정할 수 없어 제외한다.
주의: 극점은 사후에 고른 창의 극값이라, 추세 없는 무작위 가격에서도 극점 뒤에는 되돌림이 나온다(선택 효과).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis_streak import streak_table
from .events import EventContext
from .viz import NEUTRAL, plt, save_fig, save_table

PEAK_WIN = 20  # 종료 후 최고/최저점을 찾는 기간(거래일)
HZ = {"1주": 5, "1달": 20, "3달": 60, "6달": 120, "1년": 250}
PATH_MAX = PEAK_WIN + 250


def _group(lengths: np.ndarray, side: str) -> np.ndarray:
    cap = 5 if side == "up" else 3
    g = np.minimum(lengths, cap).astype(int).astype(str)
    return np.where(lengths >= cap, f"{cap}+", g)


def _streaks(panel: pd.DataFrame, ctx: EventContext, side: str) -> pd.DataFrame:
    """극점 창(종료 후 20거래일)이 데이터 안에 온전히 있는 연속 구간만."""
    st = streak_table(panel, side)
    return st[~st["데이터끝"] & (st["tidx"] + PEAK_WIN < ctx.T)].reset_index(drop=True)


def streak_extremes(panel: pd.DataFrame, ctx: EventContext, side: str = "up") -> pd.DataFrame:
    """연속 구간별 최고점(최저점)과 이후 가격 변화 지표."""
    st = _streaks(panel, ctx, side)
    e_ti = st["tidx"].to_numpy()
    s_ti = np.searchsorted(ctx.days, st["시작일"].to_numpy())
    ci = np.searchsorted(ctx.W.codes, st["code"].astype(str).to_numpy())
    pre = ctx.ri[np.maximum(s_ti - 1, 0), ci]
    pre = np.where(s_ti >= 1, pre, np.nan)
    ok_pre = ~np.isnan(pre)
    end_px = ctx.ri[e_ti, ci]
    offs = np.arange(0, PATH_MAX + 1)
    P = ctx.path(e_ti, ci, offs, ctx.ri)  # 종료일 기준 경로 (0 = 종료일)
    win = P[:, :PEAK_WIN + 1]
    n = len(st)
    if side == "up":
        ext_off = np.nanargmax(np.where(np.isnan(win), -np.inf, win), axis=1)
    else:
        ext_off = np.nanargmin(np.where(np.isnan(win), np.inf, win), axis=1)
    ext = P[np.arange(n), ext_off]
    rows = np.arange(n)[:, None]

    def after(h):  # 최고/최저점 h일 뒤 가격
        idx = ext_off + h
        v = P[np.arange(n), np.minimum(idx, PATH_MAX)]
        return np.where(idx <= PATH_MAX, v, np.nan)

    out = pd.DataFrame({"code": st["code"].astype(str), "name": st["name"], "market": st["market"].astype(str),
                        "시작일": st["시작일"], "종료일": st["종료일"], "길이": st["길이"].astype(int),
                        "첫날점상": st["첫날점상"]})
    out["길이그룹"] = _group(out["길이"].to_numpy(), side)
    out["연속중변화"] = end_px / pre - 1
    out["극점까지변화"] = ext / pre - 1
    out["극점도달(종료후일)"] = ext_off
    for name, h in HZ.items():
        out[f"극점후{name}"] = after(h) / ext - 1
    # 극점 이후 1년 구간 (극점 다음날 ~ 250일 뒤)
    seg_idx = ext_off[:, None] + np.arange(1, 251)[None, :]
    seg = np.where(seg_idx <= PATH_MAX, P[rows, np.minimum(seg_idx, PATH_MAX)], np.nan)
    full_year = ~np.isnan(seg[:, -1])
    # 가격은 float32 라 1e-9 로는 같은 값을 못 거른다. 1e-6 은 float32 오차보다 크고 호가 한 단위보다 작다.
    TOL = 1e-6
    with np.errstate(all="ignore"):
        if side == "up":
            out["1년내최대낙폭"] = np.where(full_year, np.nanmin(seg, axis=1) / ext - 1, np.nan)
            # 최고점을 '넘어서야' 회복 (거래정지일은 직전 종가 그대로라 같음으로 잡히면 안 됨)
            out["1년내최고점회복"] = np.where(full_year, (np.nanmax(seg, axis=1) > ext * (1 + TOL)).astype(float), np.nan)
            out["상승분반납률(1년)"] = (ext - after(250)) / (ext - pre)
            out["1년후_연상전보다낮음"] = np.where(full_year & ok_pre, (after(250) < pre).astype(float), np.nan)
        else:
            out["1년내최대반등"] = np.where(full_year, np.nanmax(seg, axis=1) / ext - 1, np.nan)
            out["1년내하락전가격회복"] = np.where(full_year & ok_pre,
                                         (np.nanmax(seg, axis=1) >= pre * (1 - TOL)).astype(float), np.nan)
            out["하락분회복률(1년)"] = (after(250) - ext) / (pre - ext)
    # 극점 뒤 1년 안에 상장폐지. 1년 창이 데이터 끝을 넘으면 알 수 없음(NaN) → 거래정지 비율과 같은 분모
    delisted = ctx.last_t[ci] < np.minimum(e_ti + ext_off + 250, ctx.T - 1)
    out["상장폐지"] = np.where(full_year, delisted.astype(float), np.nan)
    # 극점 6달·1년 뒤 거래정지 여부 (가격이 멈춰 있어 실제로 팔 수 없는 경우)
    for name, h in [("6달", 120), ("1년", 250)]:
        t2 = e_ti + ext_off + h
        ok = t2 < ctx.T
        t2c = np.minimum(t2, ctx.T - 1)
        halted = ctx.exists[t2c, ci] & ~ctx.traded[t2c, ci]
        out[f"극점후{name}_거래정지"] = np.where(ok, halted.astype(float), np.nan)
    return out


def summarize(df: pd.DataFrame, side: str) -> pd.DataFrame:
    med = lambda c: (c, "median")  # noqa: E731
    aggs = {"n": ("길이", "size"), "연속중변화(중앙)": med("연속중변화"), "극점까지변화(중앙)": med("극점까지변화"),
            "극점도달일(중앙)": med("극점도달(종료후일)")}
    for name in HZ:
        aggs[f"극점후{name}(중앙)"] = med(f"극점후{name}")
    if side == "up":
        aggs.update({"1년내최대낙폭(중앙)": med("1년내최대낙폭"), "1년내최고점회복(비율)": ("1년내최고점회복", "mean"),
                     "상승분반납률1년(중앙)": med("상승분반납률(1년)"),
                     "1년후연상전보다낮음(비율)": ("1년후_연상전보다낮음", "mean")})
    else:
        aggs.update({"1년내최대반등(중앙)": med("1년내최대반등"), "1년내하락전가격회복(비율)": ("1년내하락전가격회복", "mean"),
                     "하락분회복률1년(중앙)": med("하락분회복률(1년)")})
    aggs["극점후6달 거래정지(비율)"] = ("극점후6달_거래정지", "mean")
    aggs["극점후1년 거래정지(비율)"] = ("극점후1년_거래정지", "mean")
    aggs["상장폐지(비율)"] = ("상장폐지", "mean")
    t = df.groupby("길이그룹").agg(**aggs)
    t.loc["전체"] = pd.Series({k: (len(df) if f == "size" else getattr(df[c], f)()) for k, (c, f) in aggs.items()})
    return t


def run(panel: pd.DataFrame, ctx: EventContext | None = None) -> dict:
    ctx = ctx or EventContext(panel)
    res = {}
    for side in ["up", "down"]:
        df = streak_extremes(panel, ctx, side)
        tab = summarize(df, side)
        save_table(tab.round(4).reset_index(names="연속일수"), f"g4_peak_{side}")
        num = df.select_dtypes("number").columns
        save_table(df.assign(**{c: df[c].round(4) for c in num}), f"g4_peak_detail_{side}")
        if side == "up":
            bm = df[df["길이"] >= 2].groupby("market").agg(
                n=("길이", "size"), 최고점후1달=("극점후1달", "median"), 최고점후1년=("극점후1년", "median"),
                최대낙폭=("1년내최대낙폭", "median"), 최고점회복=("1년내최고점회복", "mean"))
            save_table(bm.round(4).reset_index(), "g4_peak_up_by_market_2plus")
            # 같은 종목이 20일 안에 다시 연상하면 같은 최고점을 공유하므로, 최고점 날짜가 같은 구간은 하나만
            top = df.assign(극점키=df["code"] + "_" + (st_end_ti(df, ctx) + df["극점도달(종료후일)"].to_numpy()).astype(str))
            top = top.sort_values("극점까지변화", ascending=False).drop_duplicates("극점키").head(20)[
                ["code", "name", "market", "시작일", "길이", "극점까지변화", "극점후1년", "1년내최대낙폭"]]
            save_table(top.round(4), "g4_peak_up_top20")
        res[side] = (df, tab)
        print(f"[연속 {'상' if side == 'up' else '하'}한가 극점 분석] {len(df):,}건")
        print(tab.round(3).to_string())
    _plot(panel, ctx, res)
    return res


def st_end_ti(df: pd.DataFrame, ctx: EventContext) -> np.ndarray:
    return np.searchsorted(ctx.days, df["종료일"].to_numpy())


def _plot(panel, ctx, res) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    offs = np.arange(-PEAK_WIN, 251)
    for r, side in enumerate(["up", "down"]):
        df, tab = res[side]
        st = _streaks(panel, ctx, side)
        e_ti = st["tidx"].to_numpy()
        ci = np.searchsorted(ctx.W.codes, st["code"].astype(str).to_numpy())
        ext_ti = e_ti + df["극점도달(종료후일)"].to_numpy()
        P = ctx.path(ext_ti, ci, offs, ctx.ri)
        rel = P / ctx.ri[np.minimum(ext_ti, ctx.T - 1), ci][:, None] - 1
        groups = sorted(df["길이그룹"].unique(), key=lambda g: int(g.rstrip("+")))
        cmap = plt.get_cmap("Reds" if side == "up" else "Blues")
        for k, g in enumerate(groups):
            m = (df["길이그룹"] == g).to_numpy()
            if m.sum() < 10:
                continue
            ax[r, 0].plot(offs, np.nanmedian(rel[m], axis=0), color=cmap(0.35 + 0.65 * k / max(len(groups) - 1, 1)),
                          label=f"{g}일 연속 (n={int(m.sum())})")
        ax[r, 0].axvline(0, color="k", lw=0.8)
        ax[r, 0].axhline(0, color=NEUTRAL, lw=0.8)
        word = "최고점" if side == "up" else "최저점"
        ax[r, 0].set_title(f"연속 {'상' if side == 'up' else '하'}한가: {word} 기준 가격 경로 (중앙값)")
        ax[r, 0].set_xlabel(f"{word} 이후 거래일")
        ax[r, 0].legend(fontsize=8)
        cols = [f"극점후{h}(중앙)" for h in HZ]
        t = tab.drop(index="전체")
        x = np.arange(len(cols))
        w = 0.8 / len(t)
        for k, (g, row) in enumerate(t.iterrows()):
            ax[r, 1].bar(x + k * w - 0.4 + w / 2, row[cols].to_numpy(dtype=float), w,
                         color=cmap(0.35 + 0.65 * k / max(len(t) - 1, 1)), label=f"{g}일")
        ax[r, 1].set_xticks(x, [f"{word}후 {h}" for h in HZ])
        ax[r, 1].axhline(0, color=NEUTRAL, lw=0.8)
        ax[r, 1].set_title(f"{word} 대비 가격 변화 (연속 일수별 중앙값)")
        ax[r, 1].legend(fontsize=8)
    save_fig(fig, "g4_peak")
