"""목표 3. 상한가·하한가를 따라 산 투자자(상따/하따)의 이후 수익률.

진입 방식
- 당일 진입: 이벤트 당일 상한가(하한가) 가격에 매수. 상한가 마감 종목은 종가와 같고,
  장중 터치 후 이탈 종목은 '상한가에 샀다가 물린' 경우가 된다.
- 익일 시가: 다음 거래일 시가 매수 (뉴스·커뮤니티를 보고 따라 들어가는 경우)
매도: 진입일 기준 h 거래일 뒤 종가 (익일=1, 1주=5, 1달=20, 3달=60, 반년=120, 1년=250)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .events import EventContext
from .viz import DOWN_COLOR, NEUTRAL, UP_COLOR, plt, save_fig, save_table

UP_TYPES = {
    "상따(상한가 터치 전체)": lambda p: p["up_touch"],
    "상한가 마감": lambda p: p["up_close"],
    "점상": lambda p: p["up_jeom"],
    "상한가(점상 제외)": lambda p: p["up_close"] & ~p["up_jeom"],
    "상한터치후이탈": lambda p: p["up_fail"],
    "첫 상한가": lambda p: p["up_close"] & (p["up_streak"] == 1),
    "2연상 이상": lambda p: p["up_streak"] >= 2,
}
DOWN_TYPES = {
    "하따(하한가 터치 전체)": lambda p: p["down_touch"],
    "하한가 마감": lambda p: p["down_close"],
    "점하": lambda p: p["down_jeom"],
    "하한가(점하 제외)": lambda p: p["down_close"] & ~p["down_jeom"],
    "하한터치후회복": lambda p: p["down_fail"],
    "첫 하한가": lambda p: p["down_close"] & (p["down_streak"] == 1),
    "2연하 이상": lambda p: p["down_streak"] >= 2,
}
ENTRIES = {"당일(제한가 매수)": "price", "익일 시가": "next_open"}
PATH_MAX_EVENTS = 30_000


def _stats(r: np.ndarray, mr: np.ndarray, delisted: np.ndarray) -> dict:
    ok = ~np.isnan(r)
    r, mr, delisted = r[ok], mr[ok], delisted[ok]
    if len(r) == 0:
        return {"n": 0}
    lo, hi = np.quantile(r, [0.01, 0.99])
    trimmed = r[(r >= lo) & (r <= hi)]
    return {
        "n": len(r), "평균": r.mean(), "중앙값": np.median(r), "절사평균(1%)": trimmed.mean(),
        "승률": (r > 0).mean(), "p10": np.quantile(r, 0.1), "p90": np.quantile(r, 0.9),
        "시장초과평균": np.nanmean(r - mr), "비용차감평균": r.mean() - config.ROUND_TRIP_COST,
        "상장폐지비율": delisted.mean(),
    }


def event_returns(ctx: EventContext, ev: pd.DataFrame, entry: str, side: str) -> pd.DataFrame:
    ti, ci, mi = ctx.locate(ev)
    price = (ev["upper"] if side == "up" else ev["lower"]).to_numpy(dtype=float)
    out = {}
    for hname, h in config.HORIZONS.items():
        r, _, dl = ctx.fwd_return(ti, ci, h, entry=entry, entry_price=price)
        out[f"r_{hname}"] = r
        out[f"m_{hname}"] = ctx.mkt_fwd(ti, mi, h)
        out[f"dl_{hname}"] = dl
    return pd.DataFrame(out, index=ev.index)


def run(panel: pd.DataFrame, ctx: EventContext | None = None) -> dict:
    ctx = ctx or EventContext(panel)
    cols = ["date", "code", "market", "upper", "lower", "up_close", "up_touch", "up_jeom", "up_fail", "down_close",
            "down_touch", "down_jeom", "down_fail", "up_streak", "down_streak", "is_spac", "is_pref"]
    p = panel.loc[panel["limit_valid"], cols]
    rows = []
    cache: dict[tuple, pd.DataFrame] = {}
    for side, types in [("up", UP_TYPES), ("down", DOWN_TYPES)]:
        for tname, rule in types.items():
            ev = p[rule(p)]
            for ename, entry in ENTRIES.items():
                rr = event_returns(ctx, ev, entry, side)
                cache[(tname, ename)] = rr
                for hname in config.HORIZONS:
                    st = _stats(rr[f"r_{hname}"].to_numpy(), rr[f"m_{hname}"].to_numpy(),
                                rr[f"dl_{hname}"].to_numpy())
                    rows.append({"방향": side, "이벤트": tname, "진입": ename, "보유기간": hname, **st})
    full = pd.DataFrame(rows)
    save_table(full.round(4), "g3_returns_full")

    # 발표용 요약: 평균/중앙값/승률 피벗
    piv = {}
    for metric in ["평균", "중앙값", "승률", "시장초과평균"]:
        t = full.pivot_table(index=["방향", "이벤트", "진입"], columns="보유기간", values=metric, sort=False)
        t = t[list(config.HORIZONS)]
        piv[metric] = t
        save_table(t.round(4).reset_index(), f"g3_returns_{metric}")

    # 세부 분해: 상한가 마감 / 하한가 마감, 당일 진입
    mcap_q = np.ceil(panel.groupby("date")["mktcap"].rank(pct=True) * 5)
    breakdowns = []
    for side, tname in [("up", "상한가 마감"), ("down", "하한가 마감")]:
        rule = UP_TYPES.get(tname) or DOWN_TYPES[tname]
        ev = p[rule(p)].copy()
        rr = cache[(tname, "당일(제한가 매수)")]
        ev = ev.join(rr)
        ev["연도"] = ev["date"].dt.year
        ev["시총5분위"] = mcap_q.reindex(ev.index)
        ev["유형"] = np.select([ev["is_spac"], ev["is_pref"]], ["스팩", "우선주"], "보통주")
        streak = ev["up_streak"] if side == "up" else ev["down_streak"]
        ev["연속일수"] = np.where(streak >= 3, "3+", streak.astype(str))
        for by in ["market", "연도", "시총5분위", "유형", "연속일수"]:
            for hname in ["1d", "1m", "6m", "1y"]:
                g = ev.groupby(by, observed=True)[f"r_{hname}"]
                b = pd.DataFrame({"n": g.count(), "평균": g.mean(), "중앙값": g.median(),
                                  "승률": g.apply(lambda s: (s.dropna() > 0).mean())})
                b = b.reset_index().rename(columns={by: "그룹값"})
                b.insert(0, "보유기간", hname)
                b.insert(0, "분해기준", by)
                b.insert(0, "이벤트", tname)
                breakdowns.append(b)
    bd = pd.concat(breakdowns, ignore_index=True)
    save_table(bd.round(4), "g3_breakdown")

    # 평균 누적수익률 경로 (0~250 거래일)
    offs = np.arange(0, 251)
    paths = {}
    for side, tname in [("up", "상따(상한가 터치 전체)"), ("up", "점상"), ("up", "상한터치후이탈"),
                        ("up", "2연상 이상"), ("down", "하따(하한가 터치 전체)"), ("down", "하한가 마감")]:
        rule = UP_TYPES.get(tname) or DOWN_TYPES[tname]
        ev = p[rule(p)]
        if ev.empty:
            continue
        if len(ev) > PATH_MAX_EVENTS:  # 경로 평균은 표본으로 충분 → 메모리 절약
            ev = ev.sample(PATH_MAX_EVENTS, random_state=config.RANDOM_SEED)
        ti, ci, mi = ctx.locate(ev)
        price = (ev["upper"] if side == "up" else ev["lower"]).to_numpy(dtype=float)
        rp = ctx.path(ti, ci, offs, ctx.ri) / ctx.ri[ti, ci][:, None] * (ctx.close[ti, ci] / price)[:, None] - 1
        mp = ctx.mi[np.clip(ti[:, None] + offs, 0, ctx.T - 1), mi[:, None]] / ctx.mi[ti, mi][:, None] - 1
        mp[ti[:, None] + offs >= ctx.T] = np.nan
        paths[tname] = pd.DataFrame({"평균": np.nanmean(rp, 0), "중앙값": np.nanmedian(rp, 0),
                                     "시장초과평균": np.nanmean(rp - mp, 0),
                                     "표본수": np.sum(~np.isnan(rp), 0)}, index=offs)
    path_df = pd.concat(paths, names=["이벤트", "경과일"]).reset_index()
    save_table(path_df.round(5), "g3_paths")

    # ---------------- 그림
    hz = list(config.HORIZONS)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, metric in enumerate(["평균", "중앙값", "승률"]):
        t = piv[metric]
        for (side, tname, ename), row in t.iterrows():
            if tname not in ("상따(상한가 터치 전체)", "점상", "상한터치후이탈", "하따(하한가 터치 전체)", "하한가 마감"):
                continue
            color = UP_COLOR if side == "up" else DOWN_COLOR
            ls = "-" if ename.startswith("당일") else "--"
            ax[k].plot(hz, row.to_numpy(), ls, marker="o", ms=3, color=color,
                       alpha=0.4 if "이탈" in tname or "점" in tname else 1,
                       label=f"{tname}/{ename}")
        ax[k].axhline(0.5 if metric == "승률" else 0, color=NEUTRAL, lw=0.8)
        ax[k].set_title(f"보유기간별 {metric}")
    ax[0].legend(fontsize=7)
    save_fig(fig, "g3_returns_by_horizon")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for tname, d in paths.items():
        color = DOWN_COLOR if "하" in tname[:2] else UP_COLOR
        alpha = 1.0 if "전체" in tname else 0.55
        ax[0].plot(d.index, d["평균"], color=color, alpha=alpha, label=tname)
        ax[1].plot(d.index, d["시장초과평균"], color=color, alpha=alpha, label=tname)
    ax[0].set_title("이벤트 후 평균 누적수익률 (제한가 매수 기준)")
    ax[1].set_title("시장(동일가중) 대비 초과수익률")
    for a in ax:
        a.axhline(0, color=NEUTRAL, lw=0.8)
        a.set_xlabel("경과 거래일")
        a.legend(fontsize=8)
    save_fig(fig, "g3_paths")

    up = cache[("상한가 마감", "당일(제한가 매수)")]
    dn = cache[("하한가 마감", "당일(제한가 매수)")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data, labels = [], []
    for hname in ["1d", "1m", "6m", "1y"]:
        for nm, rr in [("상", up), ("하", dn)]:
            v = rr[f"r_{hname}"].dropna().clip(-1, 3)
            data.append(v.to_numpy())
            labels.append(f"{nm}한가\n{hname}")
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor(UP_COLOR if k % 2 == 0 else DOWN_COLOR)
        b.set_alpha(0.5)
    ax.axhline(0, color=NEUTRAL, lw=0.8)
    ax.set_title("상한가/하한가 마감 종목 보유기간별 수익률 분포 (이상치 제외)")
    save_fig(fig, "g3_box")
    return {"full": full, "pivot": piv, "paths": path_df, "breakdown": bd}
