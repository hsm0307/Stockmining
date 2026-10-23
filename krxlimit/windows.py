"""학습 기간(창) 비교: 10년 데이터를 어떻게 나눠 학습하는 것이 예측에 가장 좋은가.

1) 워크포워드 비교 (실제 예측 방식, 분기마다 재학습, 2017~현재)
   ① 전체 누적 : 예측 시점 이전의 모든 데이터(2016~)
   ② 최근 1년  : 직전 1년만
   ③ 최근 3년  : 직전 3년만
   ④ 평균 앙상블: ①②③의 예측확률 평균
2) 교차표: 한 해(또는 3년 묶음)로만 만든 모델을 그 뒤의 각 해에 적용했을 때의 적중률
   → 과거 패턴이 얼마나 오래 통하는지(시장 환경 변화) 확인
모델 설정은 비교를 단순하게 하려고 v1(단일 LightGBM)을 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import V1_GROUPS
from .improve import PRIMARY_K, paired
from .model import TARGETS, evaluate, fit, predict_proba, rule_score, select_features, universe_mask, walk_forward
from .viz import NEUTRAL, plt, save_fig, save_table

STRATEGIES = {"전체 누적": None, "최근 1년": 1, "최근 3년": 3}
SLUG = {"전체 누적": "all", "최근 1년": "y1", "최근 3년": "y3"}
COLORS = {"전체 누적": "#1f77b4", "최근 1년": "#ff7f0e", "최근 3년": "#2ca02c", "평균 앙상블": "#d62728",
          "단순 규칙": NEUTRAL}
KEEP = ["date", "code", "name", "market", "next_ret", "next_open_ret", "up_close", "up_streak", "down_close",
        "down_streak", "up_touch", "ret1"]


def _yearly(daily: pd.DataFrame, k: int) -> pd.Series:
    g = daily.groupby(daily.index.year)
    return g[f"hit@{k}"].sum() / (k * g.size())


def compare_windows(feats: pd.DataFrame, target: str, start: str = "2017-01-01", freq: str = "QS",
                    verbose: bool = True) -> dict:
    k = PRIMARY_K.get(target, 10)
    pk = f"P@{k}"
    features = select_features(only_groups=V1_GROUPS)
    preds = {}
    for name, years in STRATEGIES.items():
        if verbose:
            print(f"  [{TARGETS[target]}] {name} 워크포워드 ({start}~, {freq})", flush=True)
        preds[name] = walk_forward(feats, target, start, freq=freq, kind="lgbm", neg_rate=0.1, features=features,
                                   train_years=years, verbose=verbose,
                                   cache=config.OUTPUT_DIR / "experiments" / "windows" / f"{target}_{SLUG[name]}")
    # ④ 앙상블: 같은 (날짜, 종목)의 확률 평균
    base = preds["전체 누적"].set_index(["date", "code"])
    probs = pd.concat({n: p.set_index(["date", "code"])["prob"] for n, p in preds.items()}, axis=1).dropna()
    ens = base.loc[probs.index].copy()
    ens["prob"] = probs.mean(axis=1)
    preds["평균 앙상블"] = ens.reset_index()
    rule = preds["전체 누적"].copy()
    rule["prob"] = rule_score(rule, target)
    preds["단순 규칙"] = rule

    dailies, rows = {}, []
    for name, p in preds.items():
        ev = evaluate(p, target)
        d = ev["daily"]
        dailies[name] = d
        for period, sel in [("2017~현재 (약 10년)", d.index >= d.index.min()),
                            ("2019~현재 (3년 창이 꽉 찬 구간)", d.index >= pd.Timestamp("2019-01-01")),
                            ("2024~현재", d.index >= pd.Timestamp("2024-01-01"))]:
            dd = d[sel]
            row = {"구간": period, "전략": name, "일수": len(dd)}
            for kk in (5, 10, 20):
                row[f"P@{kk}"] = dd[f"hit@{kk}"].sum() / (kk * len(dd))
            row["AP(전체)"] = ev["summary"]["AP"] if period.startswith("2017") else np.nan
            rows.append(row)
    table = pd.DataFrame(rows)
    yearly = pd.DataFrame({n: _yearly(d, k) for n, d in dailies.items()})
    yearly.index.name = "연도"
    comp = []
    for name in ["최근 1년", "최근 3년", "평균 앙상블", "단순 규칙"]:
        for period, cut in [("2017~현재", "2017-01-01"), ("2019~현재", "2019-01-01"), ("2024~현재", "2024-01-01")]:
            a, b = dailies[name], dailies["전체 누적"]
            g, t = paired(a[a.index >= cut], b[b.index >= cut], k)
            comp.append({"비교": f"{name} − 전체 누적", "구간": period, f"Δ{pk}": g, "t": t})
    comp = pd.DataFrame(comp)
    save_table(table.round(5), f"g5_window_{target}")
    save_table(yearly.round(4).reset_index(), f"g5_window_yearly_{target}")
    save_table(comp.round(5), f"g5_window_paired_{target}")
    if verbose:
        print(table.round(4).to_string(index=False), flush=True)
        print(comp.round(4).to_string(index=False), flush=True)
    return {"table": table, "yearly": yearly, "paired": comp, "dailies": dailies}


def cross_period(feats: pd.DataFrame, target: str, verbose: bool = True) -> pd.DataFrame:
    """한 해 / 3년 묶음으로만 학습한 모델을 이후 각 해에 적용한 적중률 교차표."""
    k = PRIMARY_K.get(target, 10)
    features = select_features(only_groups=V1_GROUPS)
    dates = feats["date"].to_numpy()
    base = feats[target].notna().to_numpy() & universe_mask(feats, (), target)
    last_year = int(pd.Timestamp(dates.max()).year)
    blocks = {str(y): (f"{y}-01-01", f"{y + 1}-01-01") for y in range(2016, last_year)}
    blocks.update({f"{y}~{y + 2}": (f"{y}-01-01", f"{y + 3}-01-01") for y in (2016, 2019, 2022)})
    rows = []
    for name, (s, e) in blocks.items():
        m = base & (dates >= np.datetime64(s)) & (dates < np.datetime64(e))
        if np.nansum(feats[target].to_numpy()[m]) < 20:
            continue
        model = fit(feats, target, features, "lgbm", None, 0.1, mask=m)
        tm = base & (dates >= np.datetime64(e))
        if not tm.any():
            continue
        pr = feats.loc[tm, KEEP + [target]].copy()
        pr["prob"] = predict_proba(model, feats.loc[tm, features], 0.1)
        y = _yearly(evaluate(pr, target)["daily"], k)
        for yr, v in y.items():
            rows.append({"학습": name, "평가연도": int(yr), f"P@{k}": v, "학습종류": "3년 묶음" if "~" in name else "1년"})
        if verbose:
            print(f"  [{TARGETS[target]}] {name} 모델 → " + ", ".join(f"{yr}:{v:.3f}" for yr, v in y.items()),
                  flush=True)
    out = pd.DataFrame(rows)
    save_table(out.round(5), f"g5_cross_{target}")
    return out


def plot(res: dict, cross: pd.DataFrame, target: str) -> None:
    k = PRIMARY_K.get(target, 10)
    pk = f"P@{k}"
    fig, ax = plt.subplots(1, 3, figsize=(19, 5), gridspec_kw={"width_ratios": [1.2, 1, 1.3]})
    yearly = res["yearly"]
    for name in yearly.columns:
        ax[0].plot(yearly.index, yearly[name], marker="o", ms=4, color=COLORS[name],
                   ls="--" if name == "단순 규칙" else "-", lw=2.2 if name == "평균 앙상블" else 1.5, label=name)
    ax[0].set_title(f"연도별 {pk} (분기 재학습 워크포워드)")
    ax[0].legend(fontsize=8)
    t = res["table"]
    sub = t[t["구간"].str.startswith("2019")].set_index("전략")[pk]
    ax[1].bar(sub.index, sub.to_numpy(), color=[COLORS[n] for n in sub.index])
    for x, v in enumerate(sub.to_numpy()):
        ax[1].text(x, v, f"{v:.2%}", ha="center", va="bottom", fontsize=9)
    ax[1].set_title(f"2019~현재 {pk} (모든 창이 꽉 찬 구간)")
    ax[1].tick_params(axis="x", rotation=20)
    if not cross.empty:
        mat = cross.pivot_table(index="학습", columns="평가연도", values=pk)
        order = [i for i in mat.index if "~" not in i] + [i for i in mat.index if "~" in i]
        mat = mat.loc[order]
        im = ax[2].imshow(mat.to_numpy(), cmap="Reds", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j]
                if not np.isnan(v):
                    ax[2].text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=7)
        ax[2].set_xticks(range(mat.shape[1]), mat.columns)
        ax[2].set_yticks(range(mat.shape[0]), mat.index)
        ax[2].set_xlabel("평가 연도")
        ax[2].set_ylabel("학습 데이터")
        ax[2].set_title(f"한 기간으로만 학습한 모델의 이후 연도 {pk} (%)")
        ax[2].grid(False)
        fig.colorbar(im, ax=ax[2])
    fig.suptitle(f"학습 기간 비교 — {TARGETS[target]}", y=1.02)
    save_fig(fig, f"g5_window_{target}")


def run(feats: pd.DataFrame, targets=("y_up", "y_down"), start: str = "2017-01-01", freq: str = "QS") -> dict:
    out = {}
    for t in targets:
        print(f"[학습 기간 비교] {TARGETS[t]}", flush=True)
        res = compare_windows(feats, t, start, freq)
        cross = cross_period(feats, t)
        plot(res, cross, t)
        out[t] = (res, cross)
    return out
