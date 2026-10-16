"""목표 1. 상한가·점상·하한가·점하(+장중 터치 후 이탈) 건수 분석."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import market_daily
from .viz import DOWN_COLOR, UP_COLOR, date_axis, plt, save_fig, save_table

EVENT_COLS = {
    "up_close": "상한가", "up_jeom": "점상", "up_fail": "상한터치후이탈",
    "down_close": "하한가", "down_jeom": "점하", "down_fail": "하한터치후회복",
}
PRICE_BINS = [0, 1_000, 5_000, 20_000, 100_000, np.inf]
PRICE_LABELS = ["<1천(동전주)", "1천~5천", "5천~2만", "2만~10만", "10만 이상"]


def _stock_type(df: pd.DataFrame) -> pd.Series:
    t = pd.Series("보통주", index=df.index)
    t[df["is_pref"]] = "우선주"
    t[df["is_reit"]] = "리츠"
    t[df["is_spac"]] = "스팩"
    return t


def _count_table(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = g[list(EVENT_COLS)].sum().rename(columns=EVENT_COLS)
    out.insert(0, "종목일수", g.size())
    out["상한가/만종목일"] = (out["상한가"] / out["종목일수"] * 1e4).round(2)
    out["하한가/만종목일"] = (out["하한가"] / out["종목일수"] * 1e4).round(2)
    return out


def run(panel: pd.DataFrame) -> dict:
    cols = ["date", "code", "name", "market", "sect", "base", "mktcap", "is_pref", "is_spac", "is_reit"]
    p = panel.loc[panel["limit_valid"], cols + list(EVENT_COLS)]  # 필요한 열만 복사 (메모리 절약)
    p["year"] = p["date"].dt.year
    p["month"] = p["date"].dt.to_period("M").dt.to_timestamp()
    p["유형"] = _stock_type(p)
    p["가격대"] = pd.cut(p["base"], PRICE_BINS, labels=PRICE_LABELS, right=False)
    p["시총5분위"] = np.ceil(p.groupby("date")["mktcap"].rank(pct=True) * 5).clip(1, 5).astype("Int64")
    res = {}

    # 1) 전체 요약
    total = p[list(EVENT_COLS)].sum().rename(EVENT_COLS)
    extra = {
        "상한가 경험 종목 수": p.loc[p["up_close"], "code"].nunique(),
        "하한가 경험 종목 수": p.loc[p["down_close"], "code"].nunique(),
        "분석 거래일 수": p["date"].nunique(),
        "분석 종목 수": p["code"].nunique(),
        "기간": f"{p['date'].min():%Y-%m-%d} ~ {p['date'].max():%Y-%m-%d}",
    }
    summary = pd.DataFrame({"건수": pd.concat([total.astype(object), pd.Series(extra, dtype=object)])})
    res["summary"] = summary
    save_table(summary.reset_index(names="항목"), "g1_summary")

    # 2) 연도 × 시장
    yearly = _count_table(p, ["year", "market"]).reset_index()
    res["yearly"] = yearly
    save_table(yearly, "g1_yearly_by_market")
    yearly_all = _count_table(p, "year").reset_index()
    save_table(yearly_all, "g1_yearly")

    # 3) 유형 / 가격대 / 시총 / 소속부
    for key, col in [("type", "유형"), ("price", "가격대"), ("mcap", "시총5분위")]:
        t = _count_table(p, ["market", col]).reset_index()
        res[key] = t
        save_table(t, f"g1_by_{key}")
    kq = p[p["market"] == "KOSDAQ"].copy()
    kq["소속부"] = kq["sect"].astype(str).replace("", "(미분류)")
    save_table(_count_table(kq, "소속부").reset_index(), "g1_kosdaq_by_sect")

    # 4) 상/하한가 다발 종목
    for flag, label in [("up_close", "상한가"), ("down_close", "하한가")]:
        e = p[p[flag]]
        top = (e.groupby("code", observed=True)
                .agg(종목명=("name", "last"), 시장=("market", "last"), 건수=("date", "size"),
                     첫발생=("date", "min"), 마지막발생=("date", "max"))
                .sort_values("건수", ascending=False).head(30).reset_index())
        save_table(top, f"g1_top_stocks_{flag}")
        res[f"top_{flag}"] = top

    # 5) 요일 / 월 (거래일 1일당 평균 건수)
    daily = market_daily(panel)
    d = daily.groupby("date")[["n_up", "n_down", "n_up_jeom", "n_down_jeom"]].sum()
    dow = d.groupby(d.index.dayofweek).mean().rename(index=dict(enumerate("월화수목금")))
    mon = d.groupby(d.index.month).mean()
    save_table(dow.round(2).reset_index(names="요일"), "g1_by_weekday")
    save_table(mon.round(2).reset_index(names="월"), "g1_by_month")

    # 6) 상/하한가가 가장 많았던 날
    for c, label in [("n_up", "상한가"), ("n_down", "하한가")]:
        top_days = d.sort_values(c, ascending=False).head(20)[[c]].rename(columns={c: f"{label} 종목수"})
        save_table(top_days.reset_index(), f"g1_top_days_{c}")
        res[f"top_days_{c}"] = top_days

    # ---------------- 그림
    ya = yearly_all.set_index("year")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.38
    x = np.arange(len(ya))
    ax[0].bar(x - w / 2, ya["상한가"], w, color=UP_COLOR, alpha=0.45, label="상한가(전체)")
    ax[0].bar(x - w / 2, ya["점상"], w, color=UP_COLOR, label="└ 점상")
    ax[0].bar(x + w / 2, ya["하한가"], w, color=DOWN_COLOR, alpha=0.45, label="하한가(전체)")
    ax[0].bar(x + w / 2, ya["점하"], w, color=DOWN_COLOR, label="└ 점하")
    ax[0].set_xticks(x, ya.index)
    ax[0].set_title("연도별 상·하한가 건수")
    ax[0].legend()
    ax[1].plot(ya.index, ya["상한가/만종목일"], marker="o", color=UP_COLOR, label="상한가")
    ax[1].plot(ya.index, ya["하한가/만종목일"], marker="o", color=DOWN_COLOR, label="하한가")
    ax[1].set_title("발생률 (종목·거래일 1만 건당)")
    ax[1].legend()
    save_fig(fig, "g1_yearly")

    m = p.groupby(["month", "market"], observed=True)[["up_close", "down_close"]].sum().unstack("market")
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for i, (flag, label, color) in enumerate([("up_close", "상한가", UP_COLOR), ("down_close", "하한가", DOWN_COLOR)]):
        for mk, ls in [("KOSPI", "--"), ("KOSDAQ", "-")]:
            if (flag, mk) in m.columns:
                ax[i].plot(m.index, m[(flag, mk)], ls, color=color, label=mk)
        ax[i].set_title(f"월별 {label} 건수")
        ax[i].legend()
        date_axis(ax[i])
    save_fig(fig, "g1_monthly")

    tp = res["type"].groupby("유형")[["상한가", "하한가", "종목일수"]].sum()
    tp_rate = (tp[["상한가", "하한가"]].div(tp["종목일수"], axis=0) * 1e4)
    pr = res["price"].groupby("가격대", observed=True)[["상한가", "하한가", "종목일수"]].sum()
    pr_rate = (pr[["상한가", "하한가"]].div(pr["종목일수"], axis=0) * 1e4)
    mc = res["mcap"].groupby("시총5분위")[["상한가", "하한가", "종목일수"]].sum()
    mc_rate = (mc[["상한가", "하한가"]].div(mc["종목일수"], axis=0) * 1e4)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, rate, title in [(ax[0], tp_rate, "종목 유형별"), (ax[1], pr_rate, "가격대별"),
                           (ax[2], mc_rate, "시가총액 5분위별 (1=소형)")]:
        rate.plot.bar(ax=a, color=[UP_COLOR, DOWN_COLOR], rot=0)
        a.set_title(f"{title} 발생률 (1만 종목일당)")
        a.set_xlabel("")
    save_fig(fig, "g1_by_group")
    return res
