"""목표 5. 익일 상한가(하한가) 예측 모델: 워크포워드 백테스트, 최종 학습, 실험 기록.

평가 철학
- 상한가는 하루 전체 종목의 0.5% 안팎인 극단적 불균형 문제 → 정확도(accuracy)는 의미 없음.
- '매일 장 마감 후 내일 상한가 후보 상위 k개를 고른다'는 실제 사용 방식에 맞춰
  precision@k(적중률), lift(무작위 대비 배수), AUC-PR(AP)을 핵심 지표로 쓴다.
- 워크포워드: 과거 데이터로만 학습 → 다음 분기(또는 월) 예측 → 창을 앞으로 이동. 미래 정보 누출 없음.
"""
from __future__ import annotations

import json
import pickle
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .features import ALL_FEATURES, FEATURE_GROUPS
from .viz import NEUTRAL, UP_COLOR, date_axis, plt, save_fig, save_table

TARGETS = {"y_up": "익일 상한가 마감", "y_down": "익일 하한가 마감", "y_uptouch": "익일 상한가 터치"}
KS = (5, 10, 20)

LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                   verbose=-1, n_jobs=-1)


def default_kind() -> str:
    try:
        import lightgbm  # noqa: F401
        return "lgbm"
    except ImportError:
        return "hgb"


def make_model(kind: str, params: dict | None = None):
    params = params or {}
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**{**LGBM_PARAMS, "random_state": config.RANDOM_SEED, **params})
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**{
            "max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 200,
            "l2_regularization": 1.0, "early_stopping": True, "validation_fraction": 0.1,
            "n_iter_no_change": 40, "random_state": config.RANDOM_SEED, **params})
    if kind == "logit":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(**{"C": 0.1, "max_iter": 3000, **params}))
    raise ValueError(f"알 수 없는 모델 종류: {kind}")


def select_features(drop_groups=(), only_groups=None) -> list[str]:
    groups = only_groups or list(FEATURE_GROUPS)
    return [f for g in groups if g not in drop_groups for f in FEATURE_GROUPS[g]]


def universe_mask(df: pd.DataFrame, exclude=()) -> np.ndarray:
    """예측 대상 종목 필터. exclude 예: ('spac', 'pref', 'reit', 'mgmt')."""
    m = np.ones(len(df), dtype=bool)
    for e in exclude:
        m = m & (df[f"is_{e}"] != 1).to_numpy()
    return m


def universe(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    return df[universe_mask(df, exclude)] if exclude else df


def _downsample_mask(y: np.ndarray, neg_rate: float, seed: int) -> np.ndarray:
    if neg_rate >= 1:
        return np.ones(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    return (y == 1) | (rng.random(len(y)) < neg_rate)


def adjust_prob(p, neg_rate):
    """음성 표본을 neg_rate 비율로 줄여 학습했을 때의 확률 보정."""
    p = np.asarray(p, dtype=float)
    if neg_rate >= 1:
        return p
    return p * neg_rate / (p * neg_rate + (1 - p))


def fit(df: pd.DataFrame, target: str, feats: list[str], kind: str, params=None, neg_rate: float = 0.1,
        valid_days: int = 60, seed: int = config.RANDOM_SEED, mask: np.ndarray | None = None):
    """df[mask] 로 학습. 시간 순 마지막 valid_days 거래일을 검증셋으로 조기종료(early stopping).
    큰 테이블을 통째로 복사하지 않도록 행 선택은 불리언 마스크로 처리한다."""
    mask = np.ones(len(df), dtype=bool) if mask is None else mask
    dates = df["date"].to_numpy()
    days = np.unique(dates[mask])
    cut = days[-valid_days] if len(days) > valid_days * 3 else days[-1] + np.timedelta64(1, "D")
    y = df[target].to_numpy()
    tr_m = mask & (dates < cut) & _downsample_mask(y, neg_rate, seed)
    va_m = mask & (dates >= cut)
    tr = df.loc[tr_m, feats + [target]]
    va = df.loc[va_m, feats + [target]]
    model = make_model(kind, params)
    if kind == "lgbm" and len(va) and va[target].sum() > 0:
        import lightgbm as lgb
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lightgbm 4.7+ eval_set 경고 (구버전 호환을 위해 eval_set 유지)
            model.fit(tr[feats], tr[target].astype(int), eval_set=[(va[feats], va[target].astype(int))],
                      eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        if kind == "lgbm":  # 검증셋이 없으면 조기종료 불가 → 보수적인 반복수
            model.set_params(n_estimators=300)
        model.fit(tr[feats], tr[target].astype(int))
    return model


def predict_proba(model, X: pd.DataFrame, neg_rate: float) -> np.ndarray:
    return adjust_prob(model.predict_proba(X)[:, 1], neg_rate)


# ------------------------------------------------------------------ 평가
def evaluate(preds: pd.DataFrame, target: str, score: str = "prob", ks=KS) -> dict:
    """preds: date, code, score, target, next_ret, next_open_ret 포함."""
    d = preds.dropna(subset=[target]).copy()
    y = d[target].to_numpy().astype(int)
    s = d[score].to_numpy()
    base = y.mean()
    summary = {"n": len(d), "일수": d["date"].nunique(), "양성수": int(y.sum()), "기저율": base,
               "AUC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
               "AP": average_precision_score(y, s) if y.sum() > 0 else np.nan}
    d["rank"] = d.groupby("date")[score].rank(ascending=False, method="first")
    d["oc_ret"] = (1 + d["next_ret"]) / (1 + d["next_open_ret"]) - 1  # 익일 시가→종가
    daily = d.groupby("date").agg(n=(target, "size"), 양성=(target, "sum"))
    for k in ks:
        top = d[d["rank"] <= k]
        g = top.groupby("date")
        daily[f"hit@{k}"] = g[target].sum()
        daily[f"ret_cc@{k}"] = g["next_ret"].mean()      # 당일 종가 매수 → 익일 종가 매도
        daily[f"ret_oc@{k}"] = g["oc_ret"].mean()        # 익일 시가 매수 → 익일 종가 매도
        summary[f"P@{k}"] = daily[f"hit@{k}"].sum() / (k * len(daily))
        summary[f"Lift@{k}"] = summary[f"P@{k}"] / base if base > 0 else np.nan
        pos_days = daily["양성"] > 0
        summary[f"Recall@{k}"] = (daily.loc[pos_days, f"hit@{k}"].sum() / daily.loc[pos_days, "양성"].sum()
                                  if pos_days.any() else np.nan)
        summary[f"적중1+일비율@{k}"] = (daily[f"hit@{k}"] > 0).mean()
        summary[f"수익_종가진입@{k}"] = daily[f"ret_cc@{k}"].mean() - config.ROUND_TRIP_COST
        summary[f"수익_시가진입@{k}"] = daily[f"ret_oc@{k}"].mean() - config.ROUND_TRIP_COST
    d["month"] = d["date"].dt.to_period("M").astype(str)
    monthly = daily.assign(month=daily.index.to_period("M").astype(str)).groupby("month").agg(
        일수=("n", "size"), 양성=("양성", "sum"), **{f"hit@{k}": (f"hit@{k}", "sum") for k in ks})
    for k in ks:
        monthly[f"P@{k}"] = monthly[f"hit@{k}"] / (k * monthly["일수"])
    monthly["AP"] = d.groupby("month").apply(
        lambda g: average_precision_score(g[target], g[score]) if g[target].sum() > 0 else np.nan)
    thr = []
    for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        sel = d[score] >= t
        thr.append({"확률임계값": t, "예측건수": int(sel.sum()), "일평균": sel.sum() / max(len(daily), 1),
                    "정밀도": d.loc[sel, target].mean() if sel.any() else np.nan,
                    "재현율": d.loc[sel, target].sum() / max(y.sum(), 1)})
    return {"summary": summary, "daily": daily, "monthly": monthly, "threshold": pd.DataFrame(thr)}


def rule_score(df: pd.DataFrame, target: str) -> np.ndarray:
    """비교용 단순 규칙: 오늘 상한가(하한가) → 연속일수 → 오늘 등락률 순으로 줄 세우기."""
    if target == "y_down":
        return df["down_close"] * 10 + df["down_streak"] - df["ret1"]
    return df["up_close"] * 10 + df["up_streak"] + df["up_touch"] * 2 + df["ret1"]


# ------------------------------------------------------------------ 워크포워드 백테스트
def walk_forward(feats_df: pd.DataFrame, target: str = "y_up", start: str = "2019-01-01", end: str | None = None,
                 freq: str = "QS", kind: str | None = None, params: dict | None = None, neg_rate: float = 0.1,
                 features: list[str] | None = None, exclude=(), train_years: float | None = None,
                 verbose: bool = True) -> pd.DataFrame:
    kind = kind or default_kind()
    features = features or ALL_FEATURES
    df = feats_df
    base = df[target].notna().to_numpy() & universe_mask(df, exclude)
    dates, tidx = df["date"].to_numpy(), df["tidx"].to_numpy()
    end_ts = pd.Timestamp(end) if end else df["date"].max() + pd.Timedelta(days=1)
    bounds = list(pd.date_range(start, end_ts, freq=freq))
    if not bounds or bounds[0] > pd.Timestamp(start):
        bounds = [pd.Timestamp(start)] + bounds
    if bounds[-1] < end_ts:
        bounds.append(end_ts)
    keep_cols = ["date", "code", "name", "market", target, "next_ret", "next_open_ret", "up_close",
                 "up_streak", "down_close", "down_streak", "up_touch", "ret1"]
    out = []
    y = df[target].to_numpy()
    for ps, pe in zip(bounds[:-1], bounds[1:]):
        test_m = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
        if not test_m.any():
            continue
        t0 = tidx[test_m].min()
        train_m = base & (tidx < t0)  # t0-1 행의 라벨(t0일)은 t0 장 마감 시 확정 → 사용 가능
        if train_years:
            train_m &= dates >= np.datetime64(ps - pd.DateOffset(years=train_years))
        n_pos = int(np.nansum(y[train_m]))
        if n_pos < 20:
            continue
        t = time.time()
        model = fit(df, target, features, kind, params, neg_rate, mask=train_m)
        test = df.loc[test_m, keep_cols + [c for c in features if c not in keep_cols]]
        pr = test[keep_cols].copy()
        pr["prob"] = predict_proba(model, test[features], neg_rate)
        out.append(pr)
        if verbose:
            print(f"  {ps:%Y-%m}~{pe:%Y-%m}: 학습 {int(train_m.sum()):,}행(양성 {n_pos:,}) "
                  f"→ 예측 {len(test):,}행, {time.time() - t:.0f}초")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=keep_cols + ["prob"])


def log_experiment(tag: str, target: str, cfg: dict, summary: dict) -> pd.DataFrame:
    path = config.TABLE_DIR / "experiments.csv"
    row = {"시각": datetime.now().strftime("%Y-%m-%d %H:%M"), "tag": tag, "target": target,
           **{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v) for k, v in cfg.items()},
           **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in summary.items()}}
    old = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    config.ensure_dirs()
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return new


def run_backtest(feats_df: pd.DataFrame, target: str = "y_up", tag: str | None = None, start: str = "2019-01-01",
                 freq: str = "QS", kind: str | None = None, params: dict | None = None, neg_rate: float = 0.1,
                 drop_groups=(), exclude=(), baselines: bool = True) -> dict:
    kind = kind or default_kind()
    feats = select_features(drop_groups)
    tag = tag or f"{target}_{kind}_{freq}" + (f"_drop-{'-'.join(drop_groups)}" if drop_groups else "")
    print(f"[백테스트] {tag}: 특징 {len(feats)}개, 모델 {kind}, 재학습 주기 {freq}, 시작 {start}")
    preds = walk_forward(feats_df, target, start, freq=freq, kind=kind, params=params, neg_rate=neg_rate,
                         features=feats, exclude=exclude)
    preds.to_parquet(config.OUTPUT_DIR / f"backtest_{tag}.parquet", index=False)
    ev = evaluate(preds, target)
    cfg = {"model": kind, "freq": freq, "start": start, "neg_rate": neg_rate, "n_features": len(feats),
           "drop_groups": list(drop_groups), "exclude": list(exclude), "params": params or {}}
    log_experiment(tag, target, cfg, ev["summary"])
    save_table(ev["monthly"].round(4).reset_index(), f"g5_monthly_{tag}")
    save_table(ev["threshold"].round(4), f"g5_threshold_{tag}")
    results = {"model": ev}
    if baselines:
        b = preds.copy()
        b["rule"] = rule_score(b, target)
        evb = evaluate(b, target, score="rule")
        log_experiment(f"{tag}__baseline_rule", target, {"model": "rule", "start": start}, evb["summary"])
        results["rule"] = evb
    comp = pd.DataFrame({k: v["summary"] for k, v in results.items()})
    save_table(comp.round(4).reset_index(names="지표"), f"g5_summary_{tag}")
    print(comp.round(4).to_string())
    _plot_backtest(results, target, tag)
    return {"preds": preds, **results}


def _plot_backtest(results: dict, target: str, tag: str) -> None:
    ev = results["model"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    m = ev["monthly"]
    for k, c in zip(KS, ["#d62728", "#ff7f0e", "#2ca02c"]):
        ax[0].plot(pd.PeriodIndex(m.index, freq="M").to_timestamp(), m[f"P@{k}"], color=c, label=f"P@{k}")
    if "rule" in results:
        mr = results["rule"]["monthly"]
        ax[0].plot(pd.PeriodIndex(mr.index, freq="M").to_timestamp(), mr["P@10"], color=NEUTRAL, ls="--",
                   label="규칙 P@10")
    ax[0].set_title(f"월별 적중률 (상위 k개 중 실제 {TARGETS[target]})")
    ax[0].legend()
    d = ev["daily"]
    for k, c in zip(KS, ["#d62728", "#ff7f0e", "#2ca02c"]):
        ax[1].plot(d.index, (d[f"ret_oc@{k}"].fillna(0) - config.ROUND_TRIP_COST).cumsum(), color=c,
                   label=f"상위{k} 익일 시가→종가")
    ax[1].set_title("단순 전략 누적수익(합산, 비용 차감)")
    ax[1].legend()
    date_axis(ax[0])
    date_axis(ax[1])
    th = ev["threshold"].dropna()
    ax[2].bar(th["확률임계값"].astype(str), th["정밀도"], color=UP_COLOR)
    for x, (n, p) in enumerate(zip(th["일평균"], th["정밀도"])):
        ax[2].text(x, p, f"일{n:.1f}건", ha="center", va="bottom", fontsize=8)
    ax[2].set_title("예측확률 임계값별 정밀도")
    save_fig(fig, f"g5_backtest_{tag}")


# ------------------------------------------------------------------ 최종 학습/저장
def train_final(feats_df: pd.DataFrame, target: str = "y_up", kind: str | None = None, params: dict | None = None,
                neg_rate: float = 0.1, drop_groups=(), exclude=(), name: str | None = None) -> dict:
    """최종 모델 학습·저장. name 기본값은 target (예: y_up). 비교용 버전은 y_up_v2 처럼 이름을 다르게."""
    kind = kind or default_kind()
    feats = select_features(drop_groups)
    df = feats_df
    m = df[target].notna().to_numpy() & universe_mask(df, exclude)
    model = fit(df, target, feats, kind, params, neg_rate, mask=m)
    best_iter = None
    if kind == "lgbm":
        best_iter = int(model.best_iteration_ or model.n_estimators)
        # 검증 구간까지 포함해 최적 반복수로 다시 학습
        model = make_model(kind, {**(params or {}), "n_estimators": max(best_iter, 50)})
        tr = df.loc[m & _downsample_mask(df[target].to_numpy(), neg_rate, config.RANDOM_SEED), feats + [target]]
        model.fit(tr[feats], tr[target].astype(int))
    meta = {"name": name or target, "target": target, "kind": kind, "features": feats, "neg_rate": neg_rate, "params": params or {},
            "exclude": list(exclude), "trained_through": str(df.loc[m, "date"].max().date()),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "best_iter": best_iter,
            "n_train": int(m.sum()), "n_pos": int(np.nansum(df[target].to_numpy()[m]))}
    config.ensure_dirs()
    name = name or target
    with open(config.MODEL_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump({"model": model, "meta": meta}, fh)
    (config.MODEL_DIR / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    imp = feature_importance(model, kind, feats)
    if imp is not None:
        save_table(imp.round(5), f"g5_importance_{name}")
        top = imp.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top["feature"], top["importance"], color=UP_COLOR)
        ax.set_title(f"특징 중요도 (gain) - {TARGETS.get(target, target)}")
        save_fig(fig, f"g5_importance_{name}")
    print(f"모델 저장: {config.MODEL_DIR / (name + '.pkl')} (학습 데이터 ~{meta['trained_through']})")
    return {"model": model, "meta": meta}


def feature_importance(model, kind: str, feats: list[str]) -> pd.DataFrame | None:
    if kind == "lgbm":
        gain = model.booster_.feature_importance("gain")
        return (pd.DataFrame({"feature": feats, "importance": gain / gain.sum()})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    if kind == "logit":
        coef = model[-1].coef_[0]
        return (pd.DataFrame({"feature": feats, "importance": np.abs(coef), "coef": coef})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    return None


def load_model(name: str = "y_up") -> dict:
    path = config.MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 `python -m krxlimit train --target {name}` 를 실행하세요.")
    with open(path, "rb") as fh:
        return pickle.load(fh)
