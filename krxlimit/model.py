"""목표 5. 익일 상한가(하한가) 예측 모델: 워크포워드 백테스트, 최종 학습, 실험 기록.

평가 철학
- 상한가는 하루 전체 종목의 0.5% 안팎인 극단적 불균형 문제 → 정확도(accuracy)는 의미 없음.
- '매일 장 마감 후 내일 상한가 후보 상위 k개를 고른다'는 실제 사용 방식에 맞춰
  precision@k(적중률), lift(무작위 대비 배수), AUC-PR(AP)을 핵심 지표로 쓴다.
- 워크포워드: 과거 데이터로만 학습 → 다음 분기(또는 월) 예측 → 창을 앞으로 이동. 미래 정보 누출 없음.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import pickle
import shutil
import time
from pathlib import Path
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
RANK_PARAMS = dict(objective="lambdarank", n_estimators=2000, learning_rate=0.05, num_leaves=31,
                   min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                   reg_lambda=1.0, verbose=-1, n_jobs=-1)
KINDS_HELP = ("lgbm: 이진분류 / rank: 날짜별 순위학습(LambdaRank) / bag3: 음성표본을 달리한 lgbm 3개 평균 / "
              "ens: lgbm+rank 평균 / hgb / logit")


class CalibratedRanker:
    """LightGBM LambdaRank(하루 단위로 순위를 직접 학습) + 검증셋 Platt 보정으로 실제 확률 출력."""

    def __init__(self, ranker, calibrator):
        self.ranker, self.calibrator = ranker, calibrator

    def true_proba(self, X) -> np.ndarray:
        s = self.ranker.predict(X).reshape(-1, 1)
        return self.calibrator.predict_proba(s)[:, 1]


def hard_mask(df: pd.DataFrame) -> np.ndarray:
    """'헷갈리는' 음성 후보: 당일 ±10% 이상 움직였거나, 고가 +15% 이상이거나, 상·하한가를 터치한 종목."""
    m = np.zeros(len(df), dtype=bool)
    if "ret1" in df:
        m |= np.abs(np.nan_to_num(df["ret1"].to_numpy(dtype=float))) >= 0.10
    if "high_ret" in df:
        m |= np.nan_to_num(df["high_ret"].to_numpy(dtype=float)) >= 0.15
    for c in ("up_touch", "down_touch"):
        if c in df:
            m |= df[c].to_numpy() == 1
    return m


HARD_COLS = ["ret1", "high_ret", "up_touch", "down_touch"]


class HardNegModel:
    """헷갈리는 음성은 전부, 나머지 음성은 neg_rate 만 뽑아 학습한 분류기. 행마다 다른 추출률로 확률을 되돌린다."""

    def __init__(self, model, neg_rate: float):
        self.model, self.neg_rate = model, neg_rate

    def true_proba(self, X) -> np.ndarray:
        p = self.model.predict_proba(X)[:, 1]
        r = np.where(hard_mask(X), 1.0, self.neg_rate)
        return p * r / (p * r + (1 - p))


class EnsembleModel:
    """여러 모델의 (보정된) 확률 평균."""

    def __init__(self, members: list[tuple[object, float]]):
        self.members = members  # (모델, 학습 시 음성 표본 비율)

    def true_proba(self, X) -> np.ndarray:
        return np.mean([predict_proba(m, X, nr) for m, nr in self.members], axis=0)


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
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(**{"C": 0.1, "max_iter": 3000, **params}))
    raise ValueError(f"알 수 없는 모델 종류: {kind}")


def select_features(drop_groups=(), only_groups=None) -> list[str]:
    groups = only_groups or list(FEATURE_GROUPS)
    return [f for g in groups if g not in drop_groups for f in FEATURE_GROUPS[g]]


# 운영 규칙: 가격제한 없이 거래 중인 종목(정리매매 등, features.is_nolimit)은 상·하한가가 불가능하므로
# 학습·백테스트·실전 예측 모두에서 항상 뺀다. (2024~ 보류 구간 상위 10개 중 73자리가 이런 종목이었고 적중 0)
ALWAYS_EXCLUDE = ("nolimit",)


def universe_mask(df: pd.DataFrame, exclude=()) -> np.ndarray:
    """예측 대상 종목 필터. exclude 예: ('spac', 'pref', 'reit', 'mgmt'). ALWAYS_EXCLUDE 는 항상 적용
    (해당 표시 열이 없는 옛 특징 파일이면 건너뛴다)."""
    m = np.ones(len(df), dtype=bool)
    for e in tuple(exclude) + tuple(x for x in ALWAYS_EXCLUDE if x not in exclude):
        col = f"is_{e}"
        if e in ALWAYS_EXCLUDE and col not in df.columns:
            continue
        m = m & (df[col] != 1).to_numpy()
    return m


def universe(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    return df[universe_mask(df, exclude)]


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


def sample_mask(df: pd.DataFrame, target: str, neg_rate: float, seed: int, hard_neg: bool) -> np.ndarray:
    m = _downsample_mask(df[target].to_numpy(), neg_rate, seed)
    return m | hard_mask(df) if hard_neg else m


def fit(df: pd.DataFrame, target: str, feats: list[str], kind: str, params=None, neg_rate: float = 0.1,
        valid_days: int = 60, seed: int = config.RANDOM_SEED, mask: np.ndarray | None = None,
        hard_neg: bool = False):
    """df[mask] 로 학습. 시간 순 마지막 valid_days 거래일을 검증셋으로 조기종료(early stopping).
    큰 테이블을 통째로 복사하지 않도록 행 선택은 불리언 마스크로 처리한다.
    hard_neg=True 이면 '헷갈리는' 음성(급등락·터치 종목)은 추출하지 않고 전부 학습에 쓴다."""
    if hard_neg and not set(HARD_COLS) <= set(feats):
        raise ValueError(f"hard_neg 를 쓰려면 특징에 {HARD_COLS} 가 있어야 합니다.")
    mask = np.ones(len(df), dtype=bool) if mask is None else mask
    dates = df["date"].to_numpy()
    days = np.unique(dates[mask])
    cut = days[-valid_days] if len(days) > valid_days * 3 else days[-1] + np.timedelta64(1, "D")
    tr_m = mask & (dates < cut) & sample_mask(df, target, neg_rate, seed, hard_neg)
    va_m = mask & (dates >= cut)
    if kind == "rank":
        return _fit_ranker(df, target, feats, params, tr_m, va_m, seed)
    if kind.startswith("bag"):  # 음성 표본 추출과 시드를 달리한 lgbm n개 평균 → 분산 감소
        n = int(kind[3:] or 3)
        return EnsembleModel([(fit(df, target, feats, "lgbm", params, neg_rate, valid_days, seed + i, mask, hard_neg),
                               neg_rate) for i in range(n)])
    if kind == "ens":
        return EnsembleModel([(fit(df, target, feats, "lgbm", params, neg_rate, valid_days, seed, mask, hard_neg),
                               neg_rate),
                              (fit(df, target, feats, "rank", None, neg_rate, valid_days, seed, mask, hard_neg), 1.0)])
    tr = df.loc[tr_m, feats + [target]]
    va = df.loc[va_m, feats + [target]]
    model = make_model(kind, {**(params or {}), "random_state": seed} if kind == "lgbm" else params)
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
    return HardNegModel(model, neg_rate) if hard_neg and kind == "lgbm" else model


def _fit_ranker(df, target, feats, params, tr_m, va_m, seed):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    cols = ["date"] + feats + [target]
    tr = df.loc[tr_m, cols].sort_values("date", kind="mergesort")
    va = df.loc[va_m, cols].sort_values("date", kind="mergesort")
    ranker = lgb.LGBMRanker(**{**RANK_PARAMS, "random_state": seed, **(params or {})})
    g_tr = tr.groupby("date", sort=True).size().to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(va) and va[target].sum() > 0:
            g_va = va.groupby("date", sort=True).size().to_numpy()
            ranker.fit(tr[feats], tr[target].astype(int), group=g_tr,
                       eval_set=[(va[feats], va[target].astype(int))], eval_group=[g_va], eval_at=[10],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
            cal = va
        else:
            ranker.set_params(n_estimators=300)
            ranker.fit(tr[feats], tr[target].astype(int), group=g_tr)
            cal = tr
    s = ranker.predict(cal[feats]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e4, max_iter=1000).fit(s, cal[target].astype(int))
    return CalibratedRanker(ranker, calibrator)


def predict_proba(model, X: pd.DataFrame, neg_rate: float) -> np.ndarray:
    """실제 확률 척도의 예측값. 음성 표본 추출로 학습한 분류기는 확률을 되돌려 보정한다."""
    if hasattr(model, "true_proba"):
        return model.true_proba(X)
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
def _cache_source_fingerprint() -> str:
    """캐시를 만든 계산 코드가 바뀌면 이전 예측을 재사용하지 않는다."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_data_fingerprint(df: pd.DataFrame, columns: list[str]) -> str:
    """행 순서·결측값까지 포함한 입력 지문. 큰 특징 테이블을 복제하지 않고 열별로 해시한다."""
    digest = hashlib.sha256()
    digest.update(str(len(df)).encode())
    for column in sorted(set(columns)):
        if column not in df:
            digest.update(json.dumps([column, "absent"]).encode())
            continue
        values = df[column]
        digest.update(json.dumps([column, str(values.dtype)]).encode())
        digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _cache_runtime_versions() -> dict:
    result = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _walk_forward_columns(target: str) -> list[str]:
    return ["date", "code", "name", "market", target, "next_ret", "next_open_ret", "up_close",
            "up_streak", "down_close", "down_streak", "up_touch", "ret1"]


def walk_forward_cache_signature(df: pd.DataFrame, target: str, start: str, end: str | None = None,
                                 freq: str = "QS", kind: str | None = None, params: dict | None = None,
                                 neg_rate: float = 0.1, features: list[str] | None = None, exclude=(),
                                 train_years: float | None = None, hard_neg: bool = False) -> str:
    """분기 캐시와 이를 감싸는 전체기간 캐시가 함께 쓰는 유효성 서명."""
    features = features or ALL_FEATURES
    end_ts = pd.Timestamp(end) if end else df["date"].max() + pd.Timedelta(days=1)
    input_columns = _walk_forward_columns(target) + features + ["tidx", "next_tidx"]
    input_columns += [f"is_{e}" for e in exclude]
    if hard_neg:
        input_columns += HARD_COLS
    return json.dumps({"schema": 2, "rows": int(len(df)), "last": str(df["date"].max().date()),
                       "data": _cache_data_fingerprint(df, input_columns),
                       "source": _cache_source_fingerprint(), "runtime": _cache_runtime_versions(),
                       "defaults": {"seed": config.RANDOM_SEED, "lgbm": LGBM_PARAMS, "rank": RANK_PARAMS},
                       "target": target, "kind": kind or default_kind(), "params": params,
                       "neg_rate": neg_rate, "hard_neg": hard_neg, "train_years": train_years,
                       "exclude": list(exclude), "start": start, "end": end_ts.isoformat(),
                       "freq": freq, "features": features}, sort_keys=True, default=str)


def walk_forward(feats_df: pd.DataFrame, target: str = "y_up", start: str = "2019-01-01", end: str | None = None,
                 freq: str = "QS", kind: str | None = None, params: dict | None = None, neg_rate: float = 0.1,
                 features: list[str] | None = None, exclude=(), train_years: float | None = None,
                 verbose: bool = True, hard_neg: bool = False, cache: Path | str | None = None) -> pd.DataFrame:
    """cache: 기간별 예측을 parquet 로 저장하는 폴더. 같은 설정·데이터로 다시 부르면 끝난 기간은 건너뛴다
    (긴 워크포워드가 중간에 끊겨도 이어서 돌리기 위함). 데이터나 설정이 바뀌면 폴더를 비운다."""
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
    keep_cols = _walk_forward_columns(target)
    out = []
    y = df[target].to_numpy()
    if cache:
        cache = Path(cache)
        sig = walk_forward_cache_signature(df, target, start, end, freq, kind, params, neg_rate,
                                           features, exclude, train_years, hard_neg)
        sig_path = cache / "signature.json"
        if cache.exists() and (not sig_path.exists() or sig_path.read_text() != sig):
            shutil.rmtree(cache)
        cache.mkdir(parents=True, exist_ok=True)
        sig_path.write_text(sig)
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
        cached = cache / f"{ps:%Y%m%d}.parquet" if cache else None
        if cached is not None and cached.exists():
            out.append(pd.read_parquet(cached))
            continue
        t = time.time()
        model = fit(df, target, features, kind, params, neg_rate, mask=train_m, hard_neg=hard_neg)
        test = df.loc[test_m, keep_cols + [c for c in features if c not in keep_cols]]
        pr = test[keep_cols].copy()
        pr["prob"] = predict_proba(model, test[features], neg_rate)
        out.append(pr)
        if cached is not None:
            pr.to_parquet(cached, index=False)
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
                neg_rate: float = 0.1, drop_groups=(), exclude=(), name: str | None = None,
                features: list[str] | None = None, train_years: float | None = None, hard_neg: bool = False) -> dict:
    """최종 모델 학습·저장. name 기본값은 target (예: y_up). 비교용 버전은 y_up_v2 처럼 이름을 다르게."""
    kind = kind or default_kind()
    feats = features or select_features(drop_groups)
    df = feats_df
    m = df[target].notna().to_numpy() & universe_mask(df, exclude)
    if train_years:
        m &= (df["date"] >= df["date"].max() - pd.DateOffset(years=train_years)).to_numpy()
    model = fit(df, target, feats, kind, params, neg_rate, mask=m, hard_neg=hard_neg)
    # 조기종료로 정한 반복수를 그대로 쓰고, 검증에 떼어 둔 마지막 60거래일까지 포함해 다시 학습한다.
    # (배깅·앙상블·순위학습도 마찬가지 — 그래야 실전 예측 모델이 가장 최근 데이터를 배운다)
    model, best_iter = _refit_all(model, df, m, target, feats, params, neg_rate, hard_neg)
    meta = {"name": name or target, "target": target, "kind": kind, "features": feats, "neg_rate": neg_rate, "params": params or {},
            "exclude": list(exclude), "train_years": train_years, "hard_neg": hard_neg,
            "trained_through": str(df.loc[m, "date"].max().date()),
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


def _refit_lgbm(model, df, m, target, feats, params, neg_rate, hard_neg, seed):
    inner = model.model if isinstance(model, HardNegModel) else model
    best = int(inner.best_iteration_ or inner.n_estimators)
    new = make_model("lgbm", {**(params or {}), "n_estimators": max(best, 50), "random_state": seed})
    tr = df.loc[m & sample_mask(df, target, neg_rate, seed, hard_neg), feats + [target]]
    new.fit(tr[feats], tr[target].astype(int))
    return (HardNegModel(new, neg_rate) if hard_neg else new), best


def _refit_ranker(model: CalibratedRanker, df, m, target, feats, params, neg_rate, hard_neg, seed):
    import lightgbm as lgb
    best = int(model.ranker.best_iteration_ or model.ranker.n_estimators)
    tr = df.loc[m & sample_mask(df, target, neg_rate, seed, hard_neg), ["date"] + feats + [target]]
    tr = tr.sort_values("date", kind="mergesort")
    rk = lgb.LGBMRanker(**{**RANK_PARAMS, "random_state": seed, **(params or {}), "n_estimators": max(best, 50)})
    rk.fit(tr[feats], tr[target].astype(int), group=tr.groupby("date", sort=True).size().to_numpy())
    return CalibratedRanker(rk, model.calibrator), best  # 확률 보정은 조기종료 때 검증셋으로 맞춘 것을 유지


def _refit_all(model, df, m, target, feats, params, neg_rate, hard_neg, seed=config.RANDOM_SEED):
    """조기종료 모델(들)을 전체 기간으로 다시 학습. 반환: (모델, 반복수 정보)."""
    if isinstance(model, EnsembleModel):
        # fit() 과 같은 시드 규칙: 배깅(bagN)은 구성원마다 seed+i, lgbm+rank 앙상블은 모두 seed
        is_bag = not any(isinstance(x, CalibratedRanker) for x, _ in model.members)
        members, iters = [], []
        for i, (mem, nr) in enumerate(model.members):
            s = seed + i if is_bag else seed
            if isinstance(mem, CalibratedRanker):
                new, best = _refit_ranker(mem, df, m, target, feats, None, neg_rate, hard_neg, s)
            else:
                new, best = _refit_lgbm(mem, df, m, target, feats, params, neg_rate, hard_neg, s)
            members.append((new, nr))
            iters.append(best)
        return EnsembleModel(members), iters
    if isinstance(model, CalibratedRanker):
        return _refit_ranker(model, df, m, target, feats, params, neg_rate, hard_neg, seed)
    if isinstance(model, (HardNegModel,)) or hasattr(model, "best_iteration_"):
        return _refit_lgbm(model, df, m, target, feats, params, neg_rate, hard_neg, seed)
    return model, None  # hgb, logit: 재학습 없음


def retrain_like(feats_df: pd.DataFrame, meta: dict) -> dict:
    """저장된 모델과 같은 설정(특징·모델·파라미터)으로 최신 데이터까지 다시 학습."""
    return train_final(feats_df, meta["target"], kind=meta["kind"], params=meta.get("params") or None,
                       neg_rate=meta["neg_rate"], exclude=tuple(meta.get("exclude", ())), name=meta["name"],
                       features=meta["features"], train_years=meta.get("train_years"),
                       hard_neg=meta.get("hard_neg", False))


def feature_importance(model, kind: str, feats: list[str]) -> pd.DataFrame | None:
    if isinstance(model, EnsembleModel):
        model, kind = model.members[0][0], "lgbm"
    if isinstance(model, HardNegModel):
        model = model.model
    if isinstance(model, CalibratedRanker):
        model, kind = model.ranker, "lgbm"
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

