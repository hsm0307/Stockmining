"""목표 4. 연속 상한가(연상)·연속 하한가(연하) 종목의 특징과 예측 가능성.

1) 연속 길이 분포와 'k일 연속 → k+1일째도 이어질' 조건부 확률
2) 당일 상태 → 익일 상태 전이 행렬
3) 첫 상한가 다음 날 연상 성공 vs 실패 종목의 특징 비교
4) 연상 여부 예측: 로지스틱 회귀(계수 해석) / 의사결정나무(규칙) / 그래디언트 부스팅(성능)
   — 시간 기준 분할(과거 학습 → 최근 검증)
5) 연속 상한가가 끝난 뒤 수익률 (연상 길이별)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from . import config
from .analysis_pre import compare_groups
from .events import EventContext
from .features import ALL_FEATURES
from .model import default_kind, make_model
from .viz import DOWN_COLOR, NEUTRAL, UP_COLOR, plt, save_fig, save_table

STATES = ["상한가", "상한터치이탈", "+10%이상", "0~+10%", "하락(하한가 제외)", "하한가"]


def _state_today(f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [f["up_close"] == 1, f["up_touch"] == 1, f["ret1"] >= 0.10, f["ret1"] >= 0, f["down_close"] == 1],
        STATES[:4] + ["하한가"], STATES[4]), index=f.index)


def _state_next(f: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [f["y_up"] == 1, f["y_uptouch"] == 1, f["next_ret"] >= 0.10, f["next_ret"] >= 0, f["y_down"] == 1],
        STATES[:4] + ["하한가"], STATES[4]), index=f.index)


def streak_table(panel: pd.DataFrame, side: str = "up") -> pd.DataFrame:
    """연속 구간(streak) 목록: 종목, 시작일, 종료일, 길이, 첫날 점상 여부."""
    col, jeom = ("up_streak", "up_jeom") if side == "up" else ("down_streak", "down_jeom")
    t = panel.loc[panel["traded"], ["date", "code", "name", "market", col, jeom, "tidx"]].reset_index(drop=True)
    nxt_same = t["code"].shift(-1) == t["code"]
    nxt_streak = t[col].shift(-1).where(nxt_same, 0)
    ends = t[(t[col] > 0) & (nxt_streak == 0)].copy()
    ends["길이"] = ends[col]
    ends["종료일"] = ends["date"]
    start_pos = ends.index - ends[col] + 1
    ends["시작일"] = t.loc[start_pos, "date"].to_numpy()
    ends["첫날점상"] = t.loc[start_pos, jeom].to_numpy()
    ends["데이터끝"] = ~nxt_same.loc[ends.index] & (ends["tidx"] == panel["tidx"].max())
    return ends[["code", "name", "market", "시작일", "종료일", "길이", "첫날점상", "데이터끝", "tidx"]]


def continuation_table(st: pd.DataFrame, max_k: int = 8) -> pd.DataFrame:
    st = st[~st["데이터끝"]]
    rows = []
    for k in range(1, max_k + 1):
        reach = st["길이"] >= k
        n = int(reach.sum())
        if n == 0:
            break
        cont = int((st["길이"] >= k + 1).sum())
        p = cont / n
        rows.append({"k일 연속": k, "도달 건수": n, "k+1일 이어짐": cont, "연속확률": p,
                     "95%CI": 1.96 * np.sqrt(p * (1 - p) / n)})
    return pd.DataFrame(rows)


def _fit_models(df: pd.DataFrame, target: str, feats: list[str], split: pd.Timestamp, name: str) -> dict:
    tr, te = df[df["date"] < split], df[df["date"] >= split]
    ytr, yte = tr[target].astype(int), te[target].astype(int)
    if ytr.nunique() < 2 or yte.nunique() < 2:
        return {}
    out, metrics = {}, []

    def _m(label, p):
        prec10 = yte[p >= np.quantile(p, 0.9)].mean()
        metrics.append({"모델": label, "AUC": roc_auc_score(yte, p), "AP": average_precision_score(yte, p),
                        "기저율(검증)": yte.mean(), "상위10% 적중률": prec10, "Lift@10%": prec10 / yte.mean()})

    # 기준선: 점상 여부 + 연속일수만 쓰는 규칙
    side = "up" if target == "y_up" else "down"
    base_score = te[f"{side}_streak"] + (te["up_jeom"] * 0.5 if side == "up" else 0)
    _m("기준(연속일수+점상)", base_score.to_numpy(float) + np.random.default_rng(0).random(len(te)) * 1e-6)

    logit = make_model("logit", {"class_weight": "balanced"})
    logit.fit(tr[feats], ytr)
    _m("로지스틱 회귀", logit.predict_proba(te[feats])[:, 1])
    coef = pd.DataFrame({"특징": feats, "표준화계수": logit[-1].coef_[0]})
    coef = coef.reindex(coef["표준화계수"].abs().sort_values(ascending=False).index)
    save_table(coef.round(4), f"g4_logit_coef_{name}")
    out["coef"] = coef

    Xtr = tr[feats].fillna(tr[feats].median())
    Xte = te[feats].fillna(tr[feats].median())
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(30, len(tr) // 100),
                                  class_weight="balanced", random_state=0).fit(Xtr, ytr)
    _m("의사결정나무(깊이4)", tree.predict_proba(Xte)[:, 1])
    rules = export_text(tree, feature_names=feats, show_weights=True)
    (config.TABLE_DIR / f"g4_tree_rules_{name}.txt").write_text(rules, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(tree, feature_names=feats, class_names=["중단", "연속"], filled=True, proportion=True,
              impurity=False, fontsize=7, ax=ax)
    ax.set_title(f"연속 여부 결정나무 ({name})")
    save_fig(fig, f"g4_tree_{name}")

    kind = default_kind()
    gbm = make_model(kind, {"n_estimators": 400, "learning_rate": 0.03, "min_child_samples": 50}
                     if kind == "lgbm" else {"min_samples_leaf": 50})
    gbm.fit(tr[feats], ytr)
    pg = gbm.predict_proba(te[feats])[:, 1]
    _m(f"그래디언트부스팅({kind})", pg)

    from sklearn.inspection import permutation_importance
    samp = te.sample(min(len(te), 20000), random_state=0)
    pi = permutation_importance(gbm, samp[feats], samp[target].astype(int), scoring="roc_auc",
                                n_repeats=3, random_state=0, n_jobs=1)
    imp = (pd.DataFrame({"특징": feats, "AUC감소": pi.importances_mean})
           .sort_values("AUC감소", ascending=False).reset_index(drop=True))
    save_table(imp.round(5), f"g4_perm_importance_{name}")
    out["importance"] = imp

    dec = pd.DataFrame({"p": pg, "y": yte.to_numpy()})
    dec["예측10분위"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=range(1, 11))
    calib = dec.groupby("예측10분위", observed=True).agg(n=("y", "size"), 평균예측=("p", "mean"), 실제연속률=("y", "mean"))
    save_table(calib.round(4).reset_index(), f"g4_decile_{name}")
    out["decile"] = calib

    mt = pd.DataFrame(metrics)
    mt.insert(0, "학습기간", f"~{split:%Y-%m-%d}")
    mt.insert(1, "검증 n", len(te))
    save_table(mt.round(4), f"g4_model_metrics_{name}")
    out["metrics"] = mt
    return out


def run(panel: pd.DataFrame, feats_df: pd.DataFrame, ctx: EventContext | None = None,
        split: str | None = None) -> dict:
    ctx = ctx or EventContext(panel)
    res = {}
    # 1) 연속 길이 분포 / 조건부 연속 확률
    for side in ["up", "down"]:
        st = streak_table(panel, side)
        dist = st[~st["데이터끝"]]["길이"].clip(upper=10).value_counts().sort_index()
        save_table(dist.rename("건수").reset_index().rename(columns={"길이": "연속일수(10=10+)"}), f"g4_streak_dist_{side}")
        ct = continuation_table(st)
        save_table(ct.round(4), f"g4_continuation_{side}")
        res[f"streaks_{side}"], res[f"continuation_{side}"] = st, ct
        if side == "up":
            for lab, sub in [("첫날점상", st[st["첫날점상"]]), ("첫날일반", st[~st["첫날점상"]])]:
                save_table(continuation_table(sub).round(4), f"g4_continuation_up_{lab}")
            by_mkt = {m: continuation_table(st[st["market"] == m]) for m in ["KOSPI", "KOSDAQ"]}
            for m, t in by_mkt.items():
                save_table(t.round(4), f"g4_continuation_up_{m}")
        # 연속 종료 후 수익률
        sub = st[~st["데이터끝"]].copy()
        ti = sub["tidx"].to_numpy()
        ci = np.searchsorted(ctx.W.codes, sub["code"].astype(str).to_numpy())
        for hname, h in [("1d", 1), ("1w", 5), ("1m", 20), ("3m", 60)]:
            sub[f"종료후_{hname}"] = ctx.fwd_return(ti, ci, h)[0]
        sub["길이그룹"] = sub["길이"].clip(upper=5).astype(int).astype(str).replace("5", "5+")
        after = sub.groupby("길이그룹").agg(n=("길이", "size"), **{
            f"{h}_중앙값": (f"종료후_{h}", "median") for h in ["1d", "1w", "1m", "3m"]},
            **{f"{h}_평균": (f"종료후_{h}", "mean") for h in ["1d", "1m", "3m"]})
        save_table(after.round(4).reset_index(), f"g4_after_streak_{side}")
        res[f"after_{side}"] = after

    # 2) 상태 전이 행렬 (필요한 열만 사용 → 메모리 절약)
    labeled = feats_df[["y_up", "y_down", "y_uptouch", "next_ret"]].notna().all(axis=1).to_numpy()
    f = feats_df.loc[labeled, ["date", "up_close", "up_touch", "down_close", "ret1",
                               "y_up", "y_down", "y_uptouch", "next_ret"]]
    s_today, s_next = _state_today(f), _state_next(f)
    tm = pd.crosstab(s_today, s_next, normalize="index").reindex(index=STATES, columns=STATES)
    cnt = pd.crosstab(s_today, s_next).reindex(index=STATES, columns=STATES).sum(axis=1)
    tm.insert(0, "표본수", cnt)
    save_table(tm.round(4).reset_index(names="오늘\\내일"), "g4_transition")
    res["transition"] = tm

    # 3) 첫 상한가: 다음날 연상 성공 vs 실패 특징 비교 / 4) 예측 모델
    feats = [c for c in ALL_FEATURES if c not in ("dow",)]
    split_ts = pd.Timestamp(split) if split else f["date"].quantile(0.7).normalize()
    for side, target in [("up", "y_up"), ("down", "y_down")]:
        df = feats_df[labeled & (feats_df[f"{side}_close"] == 1).to_numpy()]
        first = df[df[f"{side}_streak"] == 1]
        cmp_ = compare_groups(first[first[target] == 1], first[first[target] == 0], feats,
                              desc=lambda c: c, names=("연속", "중단"))
        cmp_ = cmp_.reindex(cmp_["SMD"].abs().sort_values(ascending=False).index)
        save_table(cmp_.round(4), f"g4_first_day_compare_{side}")
        res[f"compare_{side}"] = cmp_
        res[f"models_{side}"] = _fit_models(df, target, feats, split_ts, side)

    # ---------------- 그림
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for side, color in [("up", UP_COLOR), ("down", DOWN_COLOR)]:
        st = res[f"streaks_{side}"]
        d = st[~st["데이터끝"]]["길이"].clip(upper=10).value_counts().sort_index()
        off = -0.2 if side == "up" else 0.2
        ax[0].bar(d.index + off, d.to_numpy(), 0.4, color=color, label="연상" if side == "up" else "연하")
        ct = res[f"continuation_{side}"]
        ct = ct[ct["도달 건수"] >= 20]  # 표본이 너무 적은 k 는 그림에서 제외 (표에는 남김)
        lo = (ct["연속확률"] - ct["95%CI"]).clip(lower=0)
        hi = (ct["연속확률"] + ct["95%CI"]).clip(upper=1)
        ax[1].errorbar(ct["k일 연속"], ct["연속확률"], yerr=[ct["연속확률"] - lo, hi - ct["연속확률"]], marker="o",
                       color=color, capsize=3, label="상한가" if side == "up" else "하한가")
    ax[0].set_yscale("log")
    ax[0].set_title("연속 일수 분포 (10=10일 이상)")
    ax[0].legend()
    ax[1].set_title("k일 연속 후 k+1일째도 이어질 확률 (도달 20건 이상)")
    ax[1].set_xlabel("k")
    ax[1].legend()
    after = res["after_up"]
    ax[2].bar(after.index, after["1m_중앙값"], color=UP_COLOR, alpha=0.7, label="1달 중앙값")
    ax[2].plot(after.index, after["1d_중앙값"], "o-", color="k", label="익일 중앙값")
    ax[2].axhline(0, color=NEUTRAL, lw=0.8)
    ax[2].set_title("연상 종료 후 수익률 (연상 길이별)")
    ax[2].legend()
    save_fig(fig, "g4_streaks")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(tm[STATES].to_numpy(), cmap="Reds", vmin=0, vmax=max(0.6, float(np.nanmax(tm[STATES].to_numpy()))))
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            v = tm[STATES].iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(STATES)), STATES, rotation=30, ha="right")
    ax.set_yticks(range(len(STATES)), [f"{s} (n={int(n):,})" for s, n in zip(STATES, tm["표본수"].fillna(0))])
    ax.set_xlabel("내일 상태")
    ax.set_ylabel("오늘 상태")
    ax.set_title("상태 전이 확률")
    ax.grid(False)
    fig.colorbar(im, ax=ax)
    save_fig(fig, "g4_transition")

    m = res.get("models_up") or {}
    if m:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        imp = m["importance"].head(15).iloc[::-1]
        ax[0].barh(imp["특징"], imp["AUC감소"], color=UP_COLOR)
        ax[0].set_title("연상 예측 순열 중요도 (AUC 감소)")
        dec = m["decile"]
        ax[1].bar(dec.index.astype(str), dec["실제연속률"], color=UP_COLOR, alpha=0.7, label="실제 연상률")
        ax[1].axhline(dec["실제연속률"].mean(), color=NEUTRAL, ls="--", label="평균")
        ax[1].set_xlabel("예측확률 10분위 (10=가장 높음)")
        ax[1].set_title("예측 10분위별 실제 연상 비율 (검증기간)")
        ax[1].legend()
        save_fig(fig, "g4_model_up")
    return res
