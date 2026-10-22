"""상한가 예측을 종목 구간별로 다르게 다루면 나아지는가: 우선주 / 시총 300억 미만 보통주 / 나머지.

    python scripts/segment_model_study.py run           # S1 구간 전용 모델 (v5 설정 그대로)
    python scripts/segment_model_study.py run-tuned     # 우선주·소형 전용 모델을 작은 데이터용 설정으로
    python scripts/segment_model_study.py run-weighted  # S4 한 모델 + 우선주·소형 음성 표본을 3배 더 쓰기
    python scripts/segment_model_study.py report        # 방법별 비교표·판정

모든 run 은 분기별로 저장해 끊겨도 이어서 돈다. 기준(G)은 운영 v5 의 분기별 워크포워드 예측
(outputs/evolve/v2-haltlabel/ten_y_up_f2e5035c75.parquet)이고, 모든 방법은 예측일 D 장 마감까지 아는 정보만 쓴다.

방법
  S1  구간 전용 모델(v5 설정)을 따로 학습해 확률을 합침        H   우선주·소형만 S1, 나머지는 G
  Ht  우선주·소형만 조정 모델(잎 15, 최소표본 50, 음성 30%)   W   G 상위 10개의 구간 구성은 그대로, 구간 안 순서만 S1
  S2  (G+S1)/2                                               B   대조군 (G+v4)/2 — 일반 모델 평균만의 효과
  S3  구간별 재보정: 직전 8분기 G 표본외 상위 30위의 (실제 적중/예측 확률 합) 배수, 전체 배수 대비 상대값, 0.5~2배
  S4  한 모델에서 우선주·소형 음성 표본만 30%(나머지 10%) 추출 후 행별로 확률 보정
  Q   직전 8분기 실제 상한가의 구간 비중대로 10자리 할당(최대 나머지 방식)
  R   최근 7거래일 안에 가격제한 없는 날(상장 첫날 제외)이 있었던 종목 제외 (정리매매 등)

판정 규칙 (결과를 보기 전에 정함)
  - 판정 기간은 개발 2019~2023 뿐이다. 2024~ 는 구간 경계(300억)와 R 규칙을 고를 때 이미 본 기간이라 참고용.
  - '도움 됨' = 개발 기간 ΔP@10 >= +0.002, 분기 단위로 묶은 t >= 2.4, 개발 5개 연도 중 4개 이상 양(+),
    그리고 2024~ 에서 ΔP@10 >= 0 (비열등).
  - S2·H·W 처럼 두 모델을 섞는 방법은 대조군 B 대비로도 같은 조건을 넘어야 구간 효과로 본다.
  - 참고 지표: 매수 가능 P@10 (상한가이면서 다음 날 시가가 +29% 미만으로 시작).
비교용 실험이며 운영 모델과 무관하다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krxlimit import config  # noqa: E402

CFG_PATH = ROOT / "configs" / "y_up_v5.json"
STATE = config.OUTPUT_DIR / "evolve" / "v2-haltlabel"
V5_KEY, V4_KEY = "f2e5035c75", "fe91875e23"   # 운영 v5, 직전 운영 v4 (대조군 B)
EXP = config.OUTPUT_DIR / "experiments"
CACHE, CACHE_T, CACHE_W = EXP / "segment_models", EXP / "segment_models_tuned", EXP / "segment_models_weighted"
SMALL_CAP = 3e10          # 300억 원
SEGS = ("pref", "small", "rest")
SEG_KO = {"pref": "우선주", "small": "300억 미만", "rest": "나머지"}
START, HOLD = "2019-01-01", "2024-01-01"
TUNED = {"params": {"num_leaves": 15, "min_child_samples": 50, "learning_rate": 0.05}, "neg_rate": 0.3}
WEIGHT_RATES = {"pref": 0.3, "small": 0.3, "rest": 0.1}


def _cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def segment_of(df: pd.DataFrame) -> np.ndarray:
    """예측일 D 의 구간. 시총은 D 종가 기준(log_mcap = log(시가총액 원))."""
    mcap = np.exp(df["log_mcap"].to_numpy(dtype=float))
    pref = df["is_pref"].to_numpy() == 1
    return np.where(pref, "pref", np.where(np.nan_to_num(mcap, nan=np.inf) < SMALL_CAP, "small", "rest"))


def _quarters(start, last_date) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    b = list(pd.date_range(start, last_date, freq="QS"))
    if not b or b[0] > pd.Timestamp(start):
        b = [pd.Timestamp(start)] + b
    b.append(pd.Timestamp(last_date) + pd.Timedelta(days=1))
    return list(zip(b[:-1], b[1:]))


def _setup():
    from krxlimit.__main__ import _features
    from krxlimit.model import select_features, universe_mask
    cfg = _cfg()
    feats = select_features(only_groups=cfg["groups"])
    f = _features()
    seg = segment_of(f)
    base = f["y_up"].notna().to_numpy() & universe_mask(f, tuple(cfg["exclude"]))
    return cfg, feats, f, seg, base


def _masks(f, base, cfg, ps, pe):
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    test_q = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
    if not test_q.any():
        return None, None
    t0 = tidx[test_q].min()
    train_q = base & (tidx < t0)  # t0-1 행의 라벨은 t0 장 마감에 확정 → 사용 가능 (model.walk_forward 와 같은 규칙)
    if cfg["train_years"]:
        train_q &= dates >= np.datetime64(ps - pd.DateOffset(years=cfg["train_years"]))
    return train_q, test_q


def cmd_run(start: str, quarters: int | None, tuned: bool) -> None:
    from krxlimit.model import fit, predict_proba
    cfg, feats, f, seg, base = _setup()
    y = f["y_up"].to_numpy()
    cache, segs = (CACHE_T, ("pref", "small")) if tuned else (CACHE, SEGS)
    params = {**(cfg["params"] or {}), **TUNED["params"]} if tuned else (cfg["params"] or None)
    neg = TUNED["neg_rate"] if tuned else cfg["neg_rate"]
    cache.mkdir(parents=True, exist_ok=True)
    for ps, pe in _quarters(start, f["date"].max())[: quarters or None]:
        train_q, test_q = _masks(f, base, cfg, ps, pe)
        if test_q is None:
            continue
        for s in segs:
            path = cache / f"{s}_{ps:%Y%m%d}.parquet"
            if path.exists():
                continue
            tr, te = train_q & (seg == s), test_q & (seg == s)
            n_pos = int(np.nansum(y[tr]))
            t = time.time()
            out = f.loc[te, ["date", "code"]].copy()
            if n_pos < 20 or not te.any():
                out["p_seg"] = np.nan
            else:
                model = fit(f, "y_up", feats, cfg["kind"], params, neg, mask=tr, hard_neg=cfg.get("hard_neg", False))
                out["p_seg"] = predict_proba(model, f.loc[te, feats], neg)
            out["seg"] = s
            out.to_parquet(path, index=False)
            print(f"{'조정 ' if tuned else ''}{ps:%Y-%m} {SEG_KO[s]}: 학습 {int(tr.sum()):,}행(양성 {n_pos:,}) "
                  f"→ 예측 {int(te.sum()):,}행, {time.time() - t:.0f}초", flush=True)


def _fit_weighted(f, feats, params, mask, rate_row, seed):
    """한 모델, 행마다 다른 음성 추출 비율. 분할·인내·시드는 model.fit 의 lgbm 경로와 같고, 조기종료는
    행별 비율로 보정한 확률의 AP 로 한다 (보정이 구간 사이 순서를 바꾸므로 평가와 같은 순위로 학습량을 고른다)."""
    import warnings
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from krxlimit.model import make_model
    dates = f["date"].to_numpy()
    days = np.unique(dates[mask])
    cut = days[-60] if len(days) > 180 else days[-1] + np.timedelta64(1, "D")
    y = f["y_up"].to_numpy()
    keep = (y == 1) | (np.random.default_rng(seed).random(len(y)) < rate_row)
    tr_m, va_m = mask & (dates < cut) & keep, mask & (dates >= cut)
    model = make_model("lgbm", {**params, "random_state": seed, "metric": "None"})
    tr, va = f.loc[tr_m, feats + ["y_up"]], f.loc[va_m, feats + ["y_up"]]
    r_va = rate_row[va_m]

    def ap_corrected(y_true, p):
        return "ap_corrected", average_precision_score(y_true, p * r_va / (p * r_va + 1 - p)), True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(va) and va["y_up"].sum() > 0:
            model.fit(tr[feats], tr["y_up"].astype(int), eval_set=[(va[feats], va["y_up"].astype(int))],
                      eval_metric=ap_corrected, callbacks=[lgb.early_stopping(150, verbose=False)])
        else:
            model.set_params(n_estimators=300)
            model.fit(tr[feats], tr["y_up"].astype(int))
    return model


def cmd_run_weighted(start: str, quarters: int | None) -> None:
    cfg, feats, f, seg, base = _setup()
    rate_row = np.vectorize(WEIGHT_RATES.get)(seg).astype(float)
    n_bag = int(cfg["kind"][3:] or 3) if cfg["kind"].startswith("bag") else 1
    CACHE_W.mkdir(parents=True, exist_ok=True)
    for ps, pe in _quarters(start, f["date"].max())[: quarters or None]:
        path = CACHE_W / f"all_{ps:%Y%m%d}.parquet"
        train_q, test_q = _masks(f, base, cfg, ps, pe)
        if test_q is None or path.exists():
            continue
        if int(np.nansum(f["y_up"].to_numpy()[train_q])) < 20:
            out = f.loc[test_q, ["date", "code"]].copy()
            out["p_s4"] = np.nan
            out.to_parquet(path, index=False)
            continue
        t = time.time()
        X = f.loc[test_q, feats]
        r = rate_row[test_q]
        probs = []
        for i in range(n_bag):
            m = _fit_weighted(f, feats, cfg["params"] or {}, train_q, rate_row, config.RANDOM_SEED + i)
            raw = m.predict_proba(X)[:, 1]
            probs.append(raw * r / (raw * r + (1 - raw)))  # 행별 음성 추출 비율로 실제 확률 척도 복원
        out = f.loc[test_q, ["date", "code"]].copy()
        out["p_s4"] = np.mean(probs, axis=0)
        out.to_parquet(path, index=False)
        print(f"가중 {ps:%Y-%m}: 학습 {int(train_q.sum()):,}행 → 예측 {len(out):,}행, {time.time() - t:.0f}초", flush=True)


# ------------------------------------------------------------------ 보고
def _nolimit7() -> pd.DataFrame:
    pan = pd.read_parquet(config.DATA_DIR / "processed" / "panel.parquet",
                          columns=["date", "code", "tidx", "limit_valid", "traded", "is_first_day"])
    pan["code"] = pan["code"].astype(str)
    pan = pan.sort_values(["code", "tidx"])
    nolim = (pan["traded"].astype(bool) & ~pan["limit_valid"].astype(bool) & ~pan["is_first_day"].astype(bool))
    pan["nolimit7"] = nolim.astype(int).groupby(pan["code"]).transform(lambda s: s.rolling(7, min_periods=1).max())
    return pan[["date", "code", "nolimit7"]]


def _load_cache(cache: Path, col: str, segs, q_bounds, required: bool, rows: pd.DataFrame) -> pd.DataFrame | None:
    """분기 파일을 모으고, 평가할 (date, code) 행이 모두 들어 있는지 확인한다(학습 양성<20 으로 NaN 인 행은 '들어 있음')."""
    missing = [f"{s}_{ps:%Y%m%d}" for ps, _ in q_bounds for s in segs if not (cache / f"{s}_{ps:%Y%m%d}.parquet").exists()]
    if missing:
        if required:
            raise SystemExit(f"{cache.name}: 분기 파일 {len(missing)}개가 없습니다 (예: {missing[:3]}). run 을 끝까지 돌리세요.")
        print(f"{cache.name}: 분기 파일 {len(missing)}개 없음 → 이 방법은 건너뜀")
        return None
    x = pd.concat([pd.read_parquet(cache / f"{s}_{ps:%Y%m%d}.parquet") for ps, _ in q_bounds for s in segs],
                  ignore_index=True)
    x["code"] = x["code"].astype(str)
    x = x.rename(columns={"p_seg": col, "p_s4": col})[["date", "code", col]]
    lack = len(rows.merge(x[["date", "code"]], on=["date", "code"], how="left", indicator=True)
               .query("_merge == 'left_only'"))
    if lack:
        msg = f"{cache.name}: 평가 행 {lack:,}개가 캐시에 없습니다 (데이터가 바뀐 뒤 남은 옛 파일일 수 있음)."
        if required:
            raise SystemExit(msg + " 해당 분기 파일을 지우고 다시 돌리세요.")
        print(msg + " → 이 방법은 건너뜀")
        return None
    return x


def _daily_top(d: pd.DataFrame, col: str, k: int = 10, keep: np.ndarray | None = None) -> pd.DataFrame:
    x = d if keep is None else d[keep]
    top = x.sort_values(["date", col], ascending=[True, False], kind="mergesort").groupby("date", sort=True).head(k)
    return top.groupby("date").agg(hit=("y_up", "mean"), trade=("tradable", "mean"))


def _quota(d_all: pd.DataFrame, q_bounds) -> pd.DataFrame:
    """직전 8분기 실제 상한가의 구간 비중대로 10자리(최대 나머지 방식)를 G 확률 순으로 채운다."""
    res = {}
    for ps, pe in q_bounds:
        hist = d_all[(d_all["date"] < ps) & (d_all["date"] >= ps - pd.DateOffset(years=2))]
        share = hist.groupby("seg")["y_up"].sum().reindex(list(SEGS)).fillna(0)
        share = share / share.sum() if share.sum() > 0 else pd.Series(1 / 3, index=list(SEGS))
        raw = share * 10
        alloc = np.floor(raw).astype(int)
        for s in (raw - alloc).sort_values(ascending=False).index[: 10 - alloc.sum()]:
            alloc[s] += 1
        cur = d_all[(d_all["date"] >= ps) & (d_all["date"] < pe)]
        for dt, g in cur.groupby("date"):
            g = g.sort_values("p_G", ascending=False, kind="mergesort")
            pick = []
            for s in SEGS:
                pick += list(g.index[g["seg"] == s][: alloc[s]])
            if len(pick) < 10:
                taken = set(pick)
                pick += [i for i in g.index if i not in taken][: 10 - len(pick)]
            res[dt] = (g.loc[pick, "y_up"].mean(), g.loc[pick, "tradable"].mean())
    return pd.DataFrame.from_dict(res, orient="index", columns=["hit", "trade"]).sort_index()


def _within(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """G 상위 10개의 구간별 개수는 그대로 두고, 구간 안에서는 col 순서로 다시 고른다."""
    top_g = d.sort_values(["date", "p_G"], ascending=[True, False], kind="mergesort").groupby("date").head(10)
    cnt = top_g.groupby(["date", "seg"]).size().rename("n").reset_index()
    x = d.merge(cnt, on=["date", "seg"], how="inner")
    x = x.sort_values(["date", "seg", col], ascending=[True, True, False], kind="mergesort")
    x["r"] = x.groupby(["date", "seg"]).cumcount()
    sel = x[x["r"] < x["n"]]
    return sel.groupby("date").agg(hit=("y_up", "mean"), trade=("tradable", "mean"))


def _clustered_t(diff: pd.Series) -> float:
    """분기 단위 군집 표준오차로 평균 차이의 t."""
    q = diff.index.to_period("Q")
    n = len(diff)
    resid = diff - diff.mean()
    s = resid.groupby(q).sum()
    k = len(s)
    se = np.sqrt(k / (k - 1) * (s ** 2).sum()) / n if k > 1 else 0.0
    return float(diff.mean() / se) if se > 0 else 0.0


def cmd_report() -> None:
    g = pd.read_parquet(STATE / f"ten_y_up_{V5_KEY}.parquet", columns=["date", "code", "y_up", "prob", "next_open_ret"])
    v4 = pd.read_parquet(STATE / f"ten_y_up_{V4_KEY}.parquet", columns=["date", "code", "prob"]).rename(columns={"prob": "p_v4"})
    g = g[g["y_up"].notna()].rename(columns={"prob": "p_G"})
    g["code"] = g["code"].astype(str)
    v4["code"] = v4["code"].astype(str)
    feats = pd.read_parquet(config.DATA_DIR / "processed" / "features.parquet",
                            columns=["date", "code", "tidx", "log_mcap", "is_pref"])
    feats["code"] = feats["code"].astype(str)
    feats["seg"] = segment_of(feats)
    d_all = g.merge(feats[["date", "code", "tidx", "seg"]], on=["date", "code"], how="left") \
             .merge(v4, on=["date", "code"], how="left")
    assert d_all["seg"].notna().all() and d_all["p_v4"].notna().all(), "G·v4·피처 행이 맞지 않습니다"
    # 매수 가능 적중: 상한가로 끝났고 다음 날 시가가 그날 상한가보다 낮게 시작(시가에 살 수 있었음)
    pan = pd.read_parquet(config.DATA_DIR / "processed" / "panel.parquet", columns=["code", "tidx", "open", "upper"])
    pan["code"] = pan["code"].astype(str)
    pan["tidx"] = pan["tidx"] - 1
    d_all = d_all.merge(pan.rename(columns={"open": "n_open", "upper": "n_upper"}), on=["code", "tidx"], how="left")
    d_all["tradable"] = ((d_all["y_up"] == 1) & (d_all["n_open"] < d_all["n_upper"])).astype(float)
    q_bounds = _quarters(START, d_all["date"].max())

    eval_rows = d_all.loc[d_all["date"] >= START, ["date", "code"]]
    s1 = _load_cache(CACHE, "p_S1raw", SEGS, q_bounds, required=True, rows=eval_rows)
    st = _load_cache(CACHE_T, "p_Ht_seg", ("pref", "small"), q_bounds, required=False,
                     rows=d_all.loc[(d_all["date"] >= START) & d_all["seg"].isin(["pref", "small"]), ["date", "code"]])
    s4 = _load_cache(CACHE_W, "p_S4", ("all",), q_bounds, required=False, rows=eval_rows)
    d = d_all[d_all["date"] >= START].merge(s1, on=["date", "code"], how="left")
    for extra in (st, s4):
        if extra is not None:
            d = d.merge(extra, on=["date", "code"], how="left")
    d = d.merge(_nolimit7(), on=["date", "code"], how="left")
    cover = d.groupby(d["date"] >= HOLD)["p_S1raw"].apply(lambda s: s.notna().mean())
    print(f"S1 예측이 있는 행: 개발 {cover.get(False, np.nan):.2%}, 보류 {cover.get(True, np.nan):.2%} (없는 행은 학습 양성<20 구간)")
    d["p_S1"] = d["p_S1raw"].fillna(d["p_G"])
    small_or_pref = d["seg"].isin(["pref", "small"])
    d["p_H"] = np.where(small_or_pref, d["p_S1"], d["p_G"])
    d["p_S2"] = 0.5 * d["p_G"] + 0.5 * d["p_S1"]
    d["p_B"] = 0.5 * d["p_G"] + 0.5 * d["p_v4"]
    if st is not None:
        d["p_Ht"] = np.where(small_or_pref, d["p_Ht_seg"].fillna(d["p_G"]), d["p_G"])

    # S3: 직전 8분기 G 표본외 상위 30위의 구간별 (적중/예측합) ÷ 전체 (적중/예측합), 0.5~2배
    d_all["rk_G"] = d_all.groupby("date")["p_G"].rank(ascending=False, method="first")
    d["mult"] = 1.0
    mult_log = []
    for ps, pe in q_bounds:
        hist = d_all[(d_all["date"] < ps) & (d_all["date"] >= ps - pd.DateOffset(years=2)) & (d_all["rk_G"] <= 30)]
        m_all = hist["y_up"].sum() / hist["p_G"].sum()
        m = (hist.groupby("seg")["y_up"].sum() / hist.groupby("seg")["p_G"].sum() / m_all).clip(0.5, 2.0)
        sel = (d["date"] >= ps) & (d["date"] < pe)
        d.loc[sel, "mult"] = d.loc[sel, "seg"].map(m).fillna(1.0)
        mult_log.append({"분기": f"{ps:%Y-%m}", **{SEG_KO[s]: round(float(m.get(s, np.nan)), 2) for s in SEGS}})
    d["p_S3"] = d["p_G"] * d["mult"]

    keep_r = d["nolimit7"].fillna(0).to_numpy() == 0
    days = pd.Index(sorted(d["date"].unique()))
    methods = {"G 운영 v5": ("p_G", None), "B 대조군 (G+v4)/2": ("p_B", None), "S1 구간 전용 모델": ("p_S1", None),
               "H 우선주·소형만 S1": ("p_H", None), "S2 (G+S1)/2": ("p_S2", None), "S3 구간별 재보정": ("p_S3", None),
               "R 정리매매 등 제외": ("p_G", keep_r), "S3 + R": ("p_S3", keep_r)}
    if "p_Ht" in d:
        methods["Ht 우선주·소형 조정 모델"] = ("p_Ht", None)
    if "p_S4" in d:
        d["p_S4"] = d["p_S4"].fillna(d["p_G"])
        methods["S4 한 모델·우선주소형 음성 3배"] = ("p_S4", None)
    daily = {name: _daily_top(d, col, 10, m).reindex(days).fillna(0) for name, (col, m) in methods.items()}
    daily["W 구간 구성 유지·구간 안 S1 순서"] = _within(d, "p_S1").reindex(days).fillna(0)
    daily["Q 구간별 자리 할당"] = _quota(d_all, q_bounds).reindex(days).fillna(0)

    dev, hold = days < HOLD, days >= HOLD
    rows, years = [], sorted(set(days.year))
    for name, x in daily.items():
        row = {"방법": name}
        for ref_name in ["G 운영 v5", "B 대조군 (G+v4)/2"]:
            ref = daily[ref_name]
            tag = "G" if ref_name.startswith("G") else "B"
            if tag == "B" and not any(k in name for k in ("S2", "H ", "W ", "Ht", "B ")):
                continue
            diff = x["hit"] - ref["hit"]
            dev_d, dev_t, hold_d = diff[dev].mean(), _clustered_t(diff[dev]), diff[hold].mean()
            yr = diff[dev].groupby(days[dev].year).mean()
            ok = dev_d >= 0.002 and dev_t >= 2.4 and (yr > 0).sum() >= 4 and hold_d >= 0   # 반올림 전 값으로 판정
            row[f"개발 Δ vs {tag}"] = round(dev_d, 4)
            row[f"개발 군집t vs {tag}"] = round(dev_t, 2)
            row[f"개발 양(+)연도 vs {tag}"] = f"{int((yr > 0).sum())}/{len(yr)}"
            row[f"보류 Δ vs {tag}"] = round(hold_d, 4)
            if tag == "G":
                row["판정"] = "기준" if name.startswith("G") else ("도움 됨" if ok else "기준 미달")
            elif row.get("판정") == "도움 됨" and not ok:
                row["판정"] = "모델 평균 효과뿐"
        row["개발 P@10"] = round(x["hit"][dev].mean(), 4)
        row["보류 P@10"] = round(x["hit"][hold].mean(), 4)
        row["개발 매수가능 P@10"] = round(x["trade"][dev].mean(), 4)
        row["보류 매수가능 P@10"] = round(x["trade"][hold].mean(), 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    yearly = pd.DataFrame({name: (x["hit"] - daily["G 운영 v5"]["hit"]).groupby(days.year).mean().round(4)
                           for name, x in daily.items() if not name.startswith("G")})
    yearly.index.name = "연도 ΔP@10 vs G"

    inner = []
    for s in SEGS:
        for period, sel in [("개발", dev), ("보류", hold)]:
            pm = (d["seg"] == s).to_numpy() & d["date"].isin(days[sel]).to_numpy()
            cols = [("G", "p_G"), ("S1", "p_S1")] + ([("Ht", "p_Ht")] if "p_Ht" in d else [])
            res = {c: _daily_top(d, col, 3, pm)["hit"] for c, col in cols}
            base = res["G"]
            row = {"구간": SEG_KO[s], "기간": period, "일수": len(base), "G 구간내 P@3": round(base.mean(), 4)}
            for c in res:
                if c == "G":
                    continue
                diff = (res[c] - base).dropna()
                row[f"{c} 구간내 P@3"] = round(res[c].mean(), 4)
                row[f"{c}−G"] = round(diff.mean(), 4)
                row[f"{c} 군집t"] = round(_clustered_t(diff), 2) if len(diff) > 1 else np.nan
            inner.append(row)
    inner = pd.DataFrame(inner)

    comp = []
    for name, (col, m) in methods.items():
        x = d if m is None else d[m]
        top = x.sort_values(["date", col], ascending=[True, False], kind="mergesort").groupby("date").head(10)
        row = {"방법": name}
        for s in SEGS:
            ts = top[top["seg"] == s]
            row[f"{SEG_KO[s]} 비중"] = round(len(ts) / len(top), 3)
            row[f"{SEG_KO[s]} 적중률"] = round(ts["y_up"].mean(), 4) if len(ts) else np.nan
        comp.append(row)
    comp = pd.DataFrame(comp)
    config.ensure_dirs()
    for df_, name in [(table, "g5_segment_models_y_up"), (inner, "g5_segment_models_inner_y_up"),
                      (comp, "g5_segment_models_composition_y_up"), (pd.DataFrame(mult_log), "g5_segment_models_calibration_y_up")]:
        df_.to_csv(config.TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(config.TABLE_DIR / "g5_segment_models_yearly_y_up.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for df_ in (table, yearly, inner, comp, pd.DataFrame(mult_log).tail(8)):
        print(df_.to_string(index=df_ is yearly)); print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "run-tuned", "run-weighted"):
        r = sub.add_parser(name)
        r.add_argument("--start", default=START)
        r.add_argument("--quarters", type=int, default=None, help="앞에서부터 이만큼 분기만 (시험용)")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.start, a.quarters, tuned=False)
    elif a.cmd == "run-tuned":
        cmd_run(a.start, a.quarters, tuned=True)
    elif a.cmd == "run-weighted":
        cmd_run_weighted(a.start, a.quarters)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
