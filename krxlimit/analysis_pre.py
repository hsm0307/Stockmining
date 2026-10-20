"""목표 2. 상·하한가 직전 20/60 거래일에 특이사항(매집 흔적 등)이 있었는가.

방법
1) 사건연구(event study): '첫' 상한가(직전 20일간 상한가 없음) 전후 -60~+20일의
   초과수익률(AR, 시장 동일가중 대비)과 비정상 거래량(AV, 과거 [-120,-61]일 중앙값 대비 로그배율)을
   같은 날·같은 시장·같은 시가총액 5분위의 '대조군' 종목과 비교.
2) 직전 20/60일 요약지표를 이벤트군 vs 대조군으로 비교 (표준화 평균차 SMD, AUC, Mann-Whitney 검정).
3) 매집 의심 점수(규칙 기반 0~5점)와 이후 수익률 (펌프앤덤프 여부).
4) K-means 군집으로 상한가 직전 패턴 유형화.
※ 투자자별(개인/외국인/기관) 매매 데이터가 없으므로 '매집'은 가격·거래량 패턴으로만 추정한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .events import EventContext
from .viz import DOWN_COLOR, NEUTRAL, UP_COLOR, plt, save_fig, save_table

METRIC_DESC = {
    "car": "누적초과수익률", "ret": "누적수익률", "abvol": "평균 비정상거래량(로그배율)",
    "volat": "일간 변동성", "range": "고저폭(박스권 폭)", "vol_trend": "거래량 추세(기울기)",
    "spikes": "거래량 급증일수(3배↑)", "lowvol": "거래량 고갈일 비율(0.5배↓)",
    "shadow": "장대윗꼬리(고가+10%↑ 종가+3%↓) 일수", "cmf": "자금흐름지표 CMF", "obv": "OBV 순매수비율",
    "uptouch": "상한가 터치 횟수", "downtouch": "하한가 터치 횟수",
}


def _window_metrics(ctx: EventContext, ti, ci, mi, W: int, up_touch_w, down_touch_w, logvol, base_med) -> dict:
    offs = np.arange(-W, 0)
    P = lambda arr: ctx.path(ti, ci, offs, arr)  # noqa: E731
    ret = P(ctx.ret_t)
    mret = ctx.mkt_ret[np.clip(ti[:, None] + offs, 0, ctx.T - 1), mi[:, None]]
    lv = P(logvol)
    vol = P(ctx.vol_t)
    hi, lo, cl, bs = P(ctx.high), P(ctx.low), P(ctx.close), P(ctx.base)
    ri = P(ctx.ri)
    medvol = np.expm1(base_med)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = offs - offs.mean()
        yc = lv - np.nanmean(lv, axis=1, keepdims=True)
        m = ~np.isnan(lv)
        slope = np.nansum(np.where(m, x * yc, 0), 1) / np.nansum(np.where(m, x ** 2, 0), 1)
        clv = ((cl - lo) - (hi - cl)) / (hi - lo)
        clv = np.where(hi == lo, 0, clv)
        vsum = np.nansum(vol, 1)
        out = {
            "car": np.nansum(ret - mret, 1),
            "ret": ctx.ri[np.maximum(ti - 1, 0), ci] / ctx.ri[np.maximum(ti - 1 - W, 0), ci] - 1,
            "abvol": np.nanmean(lv - base_med[:, None], 1),
            "volat": np.nanstd(ret, 1),
            "range": np.nanmax(ri, 1) / np.nanmin(ri, 1) - 1,
            "vol_trend": slope,
            "spikes": np.nansum(vol > 3 * medvol, 1).astype(float),
            "lowvol": np.nanmean(np.where(np.isnan(vol), np.nan, vol < 0.5 * medvol), 1),
            "shadow": np.nansum((hi / bs - 1 >= 0.10) & (ret <= 0.03), 1).astype(float),
            "cmf": np.nansum(np.nan_to_num(clv) * vol, 1) / vsum,
            "obv": np.nansum(np.sign(np.nan_to_num(ret)) * vol, 1) / vsum,
            "uptouch": np.nansum(P(up_touch_w), 1),
            "downtouch": np.nansum(P(down_touch_w), 1),
        }
    bad = ti - W - 1 < 0
    for k in out:
        out[k] = np.where(bad | ~np.isfinite(out[k]), np.nan, out[k])
    return out


def _window_desc(c: str) -> str:
    key, w = c.rsplit("_", 1)
    return f"{METRIC_DESC.get(key, key)} (직전 {w}일)"


def compare_groups(ev: pd.DataFrame, ct: pd.DataFrame, cols, desc=_window_desc,
                   names=("이벤트", "대조")) -> pd.DataFrame:
    """두 집단의 지표 비교: 평균·중앙값, 표준화 평균차(SMD), AUC(무작위 a > b 확률), Mann-Whitney p값."""
    rows = []
    for c in cols:
        a, b = ev[c].dropna().to_numpy(float), ct[c].dropna().to_numpy(float)
        if len(a) < 5 or len(b) < 5:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2)
        u, pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"지표": c, "설명": desc(c),
                     f"{names[0]}_평균": a.mean(), f"{names[1]}_평균": b.mean(),
                     f"{names[0]}_중앙값": np.median(a), f"{names[1]}_중앙값": np.median(b),
                     "SMD": (a.mean() - b.mean()) / sd if sd > 0 else np.nan,
                     "AUC": u / (len(a) * len(b)), "p값": pval})
    cols_out = ["지표", "설명", f"{names[0]}_평균", f"{names[1]}_평균", f"{names[0]}_중앙값",
                f"{names[1]}_중앙값", "SMD", "AUC", "p값"]
    return pd.DataFrame(rows, columns=cols_out)


def _accum_score(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["F1_거래량증가+횡보"] = (df["abvol_60"] >= np.log(1.5)) & (df["ret_60"].abs() <= 0.10)
    f["F2_박스권(폭≤30%)"] = df["range_60"] <= 0.30
    f["F3_OBV매수우위+가격정체"] = (df["obv_60"] >= 0.10) & (df["ret_60"] <= 0.05)
    f["F4_윗꼬리시세테스트"] = df["shadow_60"] >= 1
    f["F5_소형주(시총하위20%)"] = df["mcap_q"] == 1
    f["매집점수"] = f.sum(axis=1)
    return f


def run(panel: pd.DataFrame, ctx: EventContext | None = None, n_controls: int = 3) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 관측치가 없는 구간의 nanmean 경고
        return _run(panel, ctx or EventContext(panel), n_controls)


def _run(panel: pd.DataFrame, ctx: EventContext, n_controls: int) -> dict:
    W_ = ctx.W
    ctx.ret_t = np.where(ctx.traded, ctx.ret, np.nan)
    ctx.vol_t = np.where(ctx.traded, ctx.volume, np.nan)
    logvol = np.log1p(ctx.vol_t)
    up_c = W_.wide(panel["up_close"].to_numpy(), fill=0, dtype=np.float32)
    dn_c = W_.wide(panel["down_close"].to_numpy(), fill=0, dtype=np.float32)
    up_t = W_.wide(panel["up_touch"].to_numpy(), fill=0, dtype=np.float32)
    dn_t = W_.wide(panel["down_touch"].to_numpy(), fill=0, dtype=np.float32)
    prev20_up = pd.DataFrame(up_c).rolling(20, min_periods=1).sum().shift(1).fillna(0).to_numpy()
    prev20_dn = pd.DataFrame(dn_c).rolling(20, min_periods=1).sum().shift(1).fillna(0).to_numpy()

    p = panel.loc[panel["limit_valid"], ["date", "code", "market", "mktcap", "up_close", "up_touch",
                                          "down_close", "down_touch"]]
    p["mcap_q"] = np.ceil(p.groupby("date")["mktcap"].rank(pct=True) * 5)
    ti_all, ci_all, _ = ctx.locate(p)
    p["prev20_up"] = prev20_up[ti_all, ci_all]
    p["prev20_dn"] = prev20_dn[ti_all, ci_all]
    p = p[ti_all >= 61]  # 직전 60일 이상 관측 가능한 경우만

    rng = np.random.default_rng(config.RANDOM_SEED)
    key = ["date", "market", "mcap_q"]
    res = {}
    paths = {}
    for side in ["up", "down"]:
        if side == "up":
            ev = p[p["up_close"] & (p["prev20_up"] == 0)]
            pool = p[~p["up_touch"] & ~p["down_touch"] & (p["prev20_up"] == 0)]
        else:
            ev = p[p["down_close"] & (p["prev20_dn"] == 0)]
            pool = p[~p["up_touch"] & ~p["down_touch"] & (p["prev20_dn"] == 0)]
        need = (ev.groupby(key, observed=True).size() * n_controls).rename("need").reset_index()
        cand = pool[key].rename_axis("row").reset_index().merge(need, on=key)
        cand["r"] = rng.random(len(cand))
        cand["k"] = cand.groupby(key, observed=True)["r"].rank(method="first")
        ct = pool.loc[cand.loc[cand["k"] <= cand["need"], "row"].to_numpy()]

        frames = {}
        for grp, df in [("event", ev), ("control", ct)]:
            ti, ci, mi = ctx.locate(df)
            base_med = np.nanmedian(ctx.path(ti, ci, np.arange(-120, -60), logvol), axis=1)
            m = {}
            for W in (20, 60):
                for k, v in _window_metrics(ctx, ti, ci, mi, W, up_t, dn_t, logvol, base_med).items():
                    m[f"{k}_{W}"] = v
            out = pd.DataFrame(m, index=df.index)
            out["mcap_q"] = df["mcap_q"].to_numpy()
            for hname in ["1m", "3m", "6m"]:
                out[f"post_{hname}"] = ctx.fwd_return(ti, ci, config.HORIZONS[hname])[0]
            frames[grp] = out

            # 사건연구 경로
            offs = np.arange(-60, 21)
            ar = ctx.path(ti, ci, offs, ctx.ret_t) - ctx.mkt_ret[np.clip(ti[:, None] + offs, 0, ctx.T - 1), mi[:, None]]
            av = ctx.path(ti, ci, offs, logvol) - base_med[:, None]
            paths[(side, grp)] = pd.DataFrame({
                "AR평균": np.nanmean(ar, 0), "CAR": np.nancumsum(np.nanmean(ar, 0)),
                "AV평균": np.nanmean(av, 0), "AV중앙값": np.nanmedian(av, 0)}, index=offs)

        cols = [c for c in frames["event"].columns if c[-3:] in ("_20", "_60")]
        cmp_ = compare_groups(frames["event"], frames["control"], cols)
        save_table(cmp_.round(4), f"g2_compare_{side}")
        res[f"compare_{side}"] = cmp_
        res[f"frames_{side}"] = frames
        print(f"[목표2] {side}: 이벤트 {len(ev):,}건, 대조군 {len(ct):,}건")

        # 매집 의심 점수
        sc_e, sc_c = _accum_score(frames["event"]), _accum_score(frames["control"])
        dist = pd.DataFrame({"이벤트": sc_e["매집점수"].value_counts(normalize=True),
                             "대조군": sc_c["매집점수"].value_counts(normalize=True)}).sort_index().fillna(0)
        dist["lift(이벤트/대조)"] = dist["이벤트"] / dist["대조군"]
        flag_rate = pd.DataFrame({"이벤트": sc_e.drop(columns="매집점수").mean(),
                                  "대조군": sc_c.drop(columns="매집점수").mean()})
        flag_rate["lift"] = flag_rate["이벤트"] / flag_rate["대조군"]
        post = frames["event"][["post_1m", "post_3m", "post_6m"]].assign(점수=sc_e["매집점수"].clip(upper=4))
        by_score = post.groupby("점수").agg(n=("post_1m", "size"), 수익률1m_중앙값=("post_1m", "median"),
                                          수익률3m_중앙값=("post_3m", "median"), 수익률6m_중앙값=("post_6m", "median"),
                                          급락비율_3m=("post_3m", lambda s: (s.dropna() <= -0.3).mean()))
        save_table(dist.round(4).reset_index(names="매집점수"), f"g2_accum_score_dist_{side}")
        save_table(flag_rate.round(4).reset_index(names="조건"), f"g2_accum_flags_{side}")
        save_table(by_score.round(4).reset_index(), f"g2_accum_post_{side}")
        res[f"accum_{side}"] = (dist, flag_rate, by_score)

    path_df = pd.concat(paths, names=["방향", "그룹", "상대일"]).reset_index()
    save_table(path_df.round(5), "g2_event_study_paths")

    # K-means 군집 (상한가 직전 패턴 유형화)
    res["clusters"] = _cluster(res["frames_up"]["event"])

    # ---------------- 그림
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for r, side in enumerate(["up", "down"]):
        color = UP_COLOR if side == "up" else DOWN_COLOR
        e, c = paths[(side, "event")], paths[(side, "control")]
        lbl = "첫 상한가" if side == "up" else "첫 하한가"
        ax[r, 0].plot(e.index, e["CAR"], color=color, label=lbl)
        ax[r, 0].plot(c.index, c["CAR"], color=NEUTRAL, ls="--", label="대조군")
        ax[r, 0].set_title(f"{lbl}: 누적초과수익률(CAR, -60일 기준)")
        ax[r, 1].plot(e.index, e["AV평균"], color=color, label=lbl)
        ax[r, 1].plot(c.index, c["AV평균"], color=NEUTRAL, ls="--", label="대조군")
        ax[r, 1].set_title(f"{lbl}: 비정상 거래량 (로그배율)")
        for a in ax[r]:
            a.axvline(0, color="k", lw=0.8)
            a.axvspan(-20, 0, color="orange", alpha=0.08)
            a.legend()
    ax[1, 0].set_xlabel("이벤트일 대비 거래일")
    ax[1, 1].set_xlabel("이벤트일 대비 거래일")
    save_fig(fig, "g2_event_study")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, side in enumerate(["up", "down"]):
        cm = res[f"compare_{side}"].set_index("설명")["SMD"].sort_values()
        ax[k].barh(cm.index, cm.to_numpy(), color=np.where(cm > 0, UP_COLOR, DOWN_COLOR))
        ax[k].axvline(0, color="k", lw=0.8)
        ax[k].set_title(f"{'상' if side == 'up' else '하'}한가 직전 지표: 이벤트 vs 대조군 (SMD)")
        ax[k].tick_params(axis="y", labelsize=7)
    save_fig(fig, "g2_smd")

    dist, flag_rate, by_score = res["accum_up"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    dist[["이벤트", "대조군"]].plot.bar(ax=ax[0], color=[UP_COLOR, NEUTRAL], rot=0)
    ax[0].set_title("매집 의심 점수 분포 (첫 상한가 vs 대조군)")
    by_score[["수익률1m_중앙값", "수익률3m_중앙값", "수익률6m_중앙값"]].plot.bar(ax=ax[1], rot=0)
    ax[1].set_title("매집 점수별 상한가 이후 수익률(중앙값)")
    ax[1].axhline(0, color="k", lw=0.8)
    save_fig(fig, "g2_accum_score")
    return res


def _cluster(ev: pd.DataFrame, k_range=range(3, 7)) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    cols = ["car_60", "abvol_60", "volat_60", "range_60", "vol_trend_60", "spikes_60", "lowvol_60",
            "shadow_60", "cmf_60", "obv_60", "uptouch_60", "car_20", "abvol_20"]
    X = ev[cols].dropna()
    if len(X) < 50:
        return pd.DataFrame()
    # 극단값에 강건하도록 순위 → 정규점수 변환
    Z = X.rank(pct=True).sub(0.5 / len(X)).apply(stats.norm.ppf)
    rng = np.random.default_rng(config.RANDOM_SEED)
    samp = Z.iloc[rng.choice(len(Z), min(len(Z), 5000), replace=False)]
    scores = {k: silhouette_score(samp, KMeans(k, n_init=10, random_state=0).fit_predict(samp)) for k in k_range}
    k_best = max(scores, key=scores.get)
    km = KMeans(k_best, n_init=10, random_state=0).fit(Z)
    lab = pd.Series(km.labels_, index=X.index, name="군집")
    centers = pd.DataFrame(km.cluster_centers_, columns=cols)
    desc = []
    for i, row in centers.iterrows():
        top = row.abs().sort_values(ascending=False).index[:3]
        desc.append(", ".join(f"{c}{'↑' if row[c] > 0 else '↓'}" for c in top))
    prof = ev.loc[X.index, cols + ["post_1m", "post_3m", "post_6m"]].groupby(lab).median()
    prof.insert(0, "n", lab.value_counts().sort_index())
    prof.insert(1, "특징(정규점수 중심 상위3)", desc)
    # Missing future returns are unknown outcomes, not non-crashes. Keep the
    # full cluster size above and expose the observed denominator separately.
    post3 = ev.loc[X.index, "post_3m"]
    observed = post3.notna().groupby(lab).sum()
    crashes = post3.le(-0.3).groupby(lab).sum()
    prof["3m수익률관측수"] = observed
    prof["3m수익률미관측수"] = prof["n"] - observed
    prof["급락건수_3m(≤-30%)"] = crashes
    prof["급락비율_3m(≤-30%)"] = crashes / observed.replace(0, np.nan)
    prof.attrs["silhouette"] = scores
    save_table(prof.round(4).reset_index(), "g2_clusters_up")
    save_table(pd.DataFrame({"k": list(scores), "silhouette": list(scores.values())}), "g2_cluster_silhouette")

    fig, ax = plt.subplots(figsize=(11, 0.6 * k_best + 2))
    im = ax.imshow(centers.to_numpy(), cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(k_best), [f"군집{i} (n={n})" for i, n in enumerate(prof["n"])])
    fig.colorbar(im, ax=ax, label="정규점수 중심")
    ax.set_title(f"첫 상한가 직전 60일 패턴 군집 (K-means, k={k_best})")
    ax.grid(False)
    save_fig(fig, "g2_clusters")
    return prof
