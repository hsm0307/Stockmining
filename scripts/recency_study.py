"""최신 데이터 위주 학습 실험('최신 데이터에 과적합'): 학습 창 축소·최근 가중·미세조정·혼합을 실전과 같은 방식으로 비교.

실전(train_final)과 똑같이 조기종료로 반복수를 정한 뒤 검증에 떼어 둔 마지막 60거래일까지 넣어 다시 학습한다(_refit_all).
(evolve 의 워크포워드는 다시 학습하지 않아 마지막 60거래일을 못 배운다 — 최근 자료 효과를 재려면 실전 방식이어야 함)
매월 1일에 그 전 자료로만 다시 학습해 그달을 예측한다(미래 정보 없음). 기준은 공식 기록 모델 v5 설정.

    python scripts/recency_study.py run                 # 모든 후보 (월별 예측은 outputs/experiments/recency/ 에 저장, 이어서 돌림)
    python scripts/recency_study.py run --only base,w3m
    python scripts/recency_study.py check               # 가중 학습 구현이 fit+_refit_all 과 같은지 한 달로 확인
    python scripts/recency_study.py report              # 선택 구간(2025-07~2026-06)·확인 구간(2026-07~) 비교표

후보를 고르는 것은 선택 구간에서만 하고, 확인 구간은 고른 뒤 결과를 보는 데만 쓴다(최근 몇 달에 맞춰 고르면 그 자체가 과적합).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krxlimit import config  # noqa: E402
from krxlimit.model import (EnsembleModel, _refit_all, fit, make_model, predict_proba, sample_mask,  # noqa: E402
                            select_features, universe_mask)

TARGET = "y_up"
CFG = json.loads((ROOT / "configs" / "y_up_v5.json").read_text(encoding="utf-8"))
FEATS = select_features(only_groups=CFG["groups"])
PARAMS, NEG = CFG["params"] or {}, CFG["neg_rate"]
CACHE = config.OUTPUT_DIR / "experiments" / "recency"
START = "2025-07-01"
SELECT_END, CONFIRM_START = "2026-07-01", "2026-07-01"
LOAD_FROM = "2022-01-01"   # 3년 창(2025-07 이전 3년)과 최근 가중 후보에 충분
SEED = config.RANDOM_SEED

# 이름: (학습 창 개월 수 또는 None=설정의 3년, 반감기(일) 또는 None, 덧붙일 미세조정, 파라미터 덧씌움, 고정 나무 수)
CANDS = {
    "base":     dict(months=None),                        # v5 그대로(3년, 조기종료 후 전체 재학습) — 실전과 같음
    "w12m":     dict(months=12),
    "w6m":      dict(months=6),
    "w3m":      dict(months=3),
    "w2m":      dict(months=2),
    "w1m":      dict(months=1),
    "dec30":    dict(months=None, half_life=30),          # 3년 창 + 최근일수록 큰 가중치(반감기 30일)
    "dec91":    dict(months=None, half_life=91),
    "dec182":   dict(months=None, half_life=182),
    "ft3m":     dict(months=None, finetune_months=3),     # v5 모델에 최근 3개월로 나무를 더 키움
    "w3m_big":  dict(months=3, params={"num_leaves": 127, "min_child_samples": 20}, trees=500),  # 말 그대로 과적합
}
MIXES = {"mix_w3m": ("base", "w3m"), "mix_w6m": ("base", "w6m"), "mix_dec91": ("base", "dec91")}


def load() -> pd.DataFrame:
    cols = list(dict.fromkeys(["date", "code", "tidx", TARGET, "up_close", "is_nolimit"] + FEATS))
    f = pd.read_parquet(config.FEATURES_PATH, columns=cols, filters=[("date", ">=", pd.Timestamp(LOAD_FROM))])
    return f.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def fit_weighted(f: pd.DataFrame, m: np.ndarray, w: np.ndarray | None, params: dict, trees: int | None = None,
                 valid_days: int = 60, n_bag: int = 3):
    """fit(kind='bag3') + _refit_all 과 같은 절차에 표본 가중치를 더한 것.
    구성원 i 는 seed+i 로 음성 표본을 뽑고, 거래일이 valid_days*3 보다 많으면 마지막 valid_days 거래일로 조기종료해
    반복수를 정한 뒤(max(best, 50)) 같은 표본 전체로 다시 학습한다. 적으면 조기종료 없이 300개(trees 로 바꿀 수 있음)."""
    import lightgbm as lgb
    dates, y = f["date"].to_numpy(), f[TARGET].to_numpy()
    days = np.unique(dates[m])
    es = trees is None and len(days) > valid_days * 3
    members, iters = [], []
    for i in range(n_bag):
        s = SEED + i
        sm = m & sample_mask(f, TARGET, NEG, s, False)
        if es:
            cut = days[-valid_days]
            tr, va = sm & (dates < cut), m & (dates >= cut)
            mdl = make_model("lgbm", {**params, "random_state": s})
            mdl.fit(f.loc[tr, FEATS], y[tr].astype(int), sample_weight=None if w is None else w[tr],
                    eval_set=[(f.loc[va, FEATS], y[va].astype(int))], eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(150, verbose=False)])
            best = int(mdl.best_iteration_ or mdl.n_estimators)
            n = max(best, 50)
        else:
            best, n = None, trees or 300
        final = make_model("lgbm", {**params, "n_estimators": n, "random_state": s})
        final.fit(f.loc[sm, FEATS], y[sm].astype(int), sample_weight=None if w is None else w[sm])
        members.append((final, NEG))
        iters.append(best)
    return EnsembleModel(members), iters


def finetune(model: EnsembleModel, f: pd.DataFrame, m_recent: np.ndarray, extra: int = 100, lr: float = 0.02):
    """구성원마다 최근 자료(같은 seed 규칙으로 음성 표본)로 나무 extra 개를 더 키운다(init_model)."""
    y = f[TARGET].to_numpy()
    members = []
    for i, (mem, nr) in enumerate(model.members):
        sm = m_recent & sample_mask(f, TARGET, NEG, SEED + i, False)
        new = make_model("lgbm", {**PARAMS, "n_estimators": extra, "learning_rate": lr, "random_state": SEED + i})
        new.fit(f.loc[sm, FEATS], y[sm].astype(int), init_model=mem.booster_)
        members.append((new, nr))
    return EnsembleModel(members)


def train_one(f: pd.DataFrame, base: np.ndarray, t0: int, ps: pd.Timestamp, spec: dict):
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    m = base & (tidx < t0)  # t0-1 행의 라벨(t0일)은 t0 장 마감 때 확정 → 사용 가능 (walk_forward 와 같음)
    months = spec.get("months")
    start = ps - (pd.DateOffset(months=months) if months else pd.DateOffset(years=CFG["train_years"]))
    m &= dates >= np.datetime64(start)
    params = {**PARAMS, **spec.get("params", {})}
    if spec.get("half_life"):
        age = (np.datetime64(ps) - dates).astype("timedelta64[D]").astype(float)
        w = np.power(0.5, age / spec["half_life"])
        w /= w[m].mean()  # 학습 표본 평균 1 로 맞춤 (가중치 합이 작으면 reg_lambda 가 상대적으로 너무 세짐)
        return fit_weighted(f, m, w, params)[0], m
    if spec.get("trees"):
        return fit_weighted(f, m, None, params, trees=spec["trees"])[0], m
    model = fit(f, TARGET, FEATS, CFG["kind"], params, NEG, mask=m)
    model, _ = _refit_all(model, f, m, TARGET, FEATS, params, NEG, False)
    if spec.get("finetune_months"):
        m_recent = m & (dates >= np.datetime64(ps - pd.DateOffset(months=spec["finetune_months"])))
        model = finetune(model, f, m_recent)
    return model, m


def _signature(f: pd.DataFrame) -> str:
    return json.dumps({"rows": len(f), "max_date": str(f["date"].max().date()), "pos": int(f[TARGET].sum()),
                       "cfg": CFG["name"], "feats": FEATS, "start": START}, sort_keys=True)


def cmd_run(only: list[str] | None) -> None:
    t_all = time.time()
    f = load()
    print(f"자료 {len(f):,}행 {f['date'].min().date()}~{f['date'].max().date()}, 특징 {len(FEATS)}개", flush=True)
    sig, sp = _signature(f), CACHE / "signature.json"
    if CACHE.exists() and (not sp.exists() or sp.read_text() != sig):
        shutil.rmtree(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)
    sp.write_text(sig)
    base = f[TARGET].notna().to_numpy() & universe_mask(f, tuple(CFG["exclude"]), TARGET)
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    bounds = list(pd.date_range(START, f["date"].max(), freq="MS")) + [f["date"].max() + pd.Timedelta(days=1)]
    for name in (only or list(CANDS)):
        spec = CANDS[name]
        for ps, pe in zip(bounds[:-1], bounds[1:]):
            path = CACHE / f"{name}_{ps:%Y%m}.parquet"
            if path.exists():
                continue
            test_m = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
            if not test_m.any():
                continue
            t = time.time()
            model, m = train_one(f, base, tidx[test_m].min(), ps, spec)
            pr = f.loc[test_m, ["date", "code", TARGET, "up_close"]].copy()
            pr["prob"] = predict_proba(model, f.loc[test_m, FEATS], NEG)
            pr["cand"] = name
            pr.to_parquet(path, index=False)
            print(f"{name:<8} {ps:%Y-%m}: 학습 {int(m.sum()):,}행(양성 {int(np.nansum(f[TARGET].to_numpy()[m])):,}) "
                  f"→ 예측 {len(pr):,}행, {time.time() - t:.0f}초", flush=True)
    print(f"끝 ({(time.time() - t_all) / 60:.1f}분)")


def cmd_check() -> None:
    """가중치 없이 fit_weighted 를 돌리면 fit(bag3)+_refit_all 과 같은 예측이 나와야 한다."""
    f = load()
    base = f[TARGET].notna().to_numpy() & universe_mask(f, tuple(CFG["exclude"]), TARGET)
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    ps = pd.Timestamp("2026-09-01")
    test_m = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64("2026-10-01"))
    t0 = tidx[test_m].min()
    m = base & (tidx < t0) & (dates >= np.datetime64(ps - pd.DateOffset(years=CFG["train_years"])))
    a = fit(f, TARGET, FEATS, CFG["kind"], PARAMS, NEG, mask=m)
    a, it_a = _refit_all(a, f, m, TARGET, FEATS, PARAMS, NEG, False)
    b, it_b = fit_weighted(f, m, None, PARAMS)
    pa, pb = predict_proba(a, f.loc[test_m, FEATS], NEG), predict_proba(b, f.loc[test_m, FEATS], NEG)
    print("반복수", it_a, it_b, "최대 차이", float(np.abs(pa - pb).max()))
    w1 = np.ones(len(f))
    c, it_c = fit_weighted(f, m, w1, PARAMS)
    pc = predict_proba(c, f.loc[test_m, FEATS], NEG)
    print("가중치 1 과의 최대 차이", float(np.abs(pa - pc).max()), it_c)


def _daily(d: pd.DataFrame) -> pd.DataFrame:
    """후보·날짜별: 1위 적중, P@5, P@10, 상한가 마감 종목 중 1위 적중."""
    d = d.sort_values(["cand", "date", "prob"], ascending=[True, True, False])
    d["rank"] = d.groupby(["cand", "date"]).cumcount() + 1
    g = d.groupby(["cand", "date"])
    out = pd.DataFrame({"hit1": d[d["rank"] == 1].set_index(["cand", "date"])[TARGET],
                        "P@5": g.apply(lambda x: x[TARGET].values[:5].mean()),
                        "P@10": g.apply(lambda x: x[TARGET].values[:10].mean())})
    lu = d[d["up_close"] == 1].groupby(["cand", "date"]).head(1).set_index(["cand", "date"])[TARGET]
    out["lu1"] = lu
    return out.reset_index()


def _budget(d: pd.DataFrame, n: int) -> tuple[float, int]:
    """기간 전체에서 확률 상위 n 개(같은 예산) 정밀도. 날짜별 순위가 아니라 확률 기준 — 고확신 규칙과 같은 방식."""
    top = d.nlargest(n, "prob")
    return float(top[TARGET].mean()) if n else float("nan"), int(top[TARGET].sum())


def cmd_report() -> None:
    files = sorted(CACHE.glob("*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    d = d[d[TARGET].notna()]
    for mix, (a, b) in MIXES.items():
        if {a, b} <= set(d["cand"]):
            x = d[d["cand"] == a].merge(d[d["cand"] == b][["date", "code", "prob"]], on=["date", "code"],
                                        suffixes=("", "_b"))
            d = pd.concat([d, x.assign(prob=(x["prob"] + x["prob_b"]) / 2, cand=mix).drop(columns="prob_b")])
    full = d.groupby("cand")["date"].nunique()
    cands = [c for c in d["cand"].unique() if full[c] == full["base"]]
    dropped = sorted(set(d["cand"].unique()) - set(cands))
    if dropped:
        print("기간이 다 차지 않아 뺀 후보:", dropped)
    d = d[d["cand"].isin(cands)]
    daily = _daily(d)
    segs = {"선택 2025-07~2026-06": (START, SELECT_END), "확인 2026-07~": (CONFIRM_START, "2100-01-01"),
            "전체": (START, "2100-01-01"), "실전 기간 10/22~": ("2026-10-22", "2100-01-01")}
    rows = []
    for seg, (s, e) in segs.items():
        dd = daily[(daily["date"] >= s) & (daily["date"] < e)]
        ds = d[(d["date"] >= s) & (d["date"] < e)]
        b = dd[dd["cand"] == "base"].set_index("date")
        n_hc = int((ds[ds["cand"] == "base"]["prob"] >= 0.15).sum())
        for c in cands:
            x = dd[dd["cand"] == c].set_index("date")
            r = {"구간": seg, "후보": c, "일수": len(x)}
            for k in ["P@10", "P@5", "hit1", "lu1"]:
                r[k] = round(x[k].mean(), 4)
                if c != "base":
                    diff = (x[k] - b[k]).dropna()
                    r[f"Δ{k}"] = round(diff.mean(), 4)
                    r[f"t {k}"] = round(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))), 2) if diff.std() > 0 else 0.0
            dc = ds[ds["cand"] == c]
            prec, hits = _budget(dc, n_hc)
            r.update({"같은예산 픽": n_hc, "같은예산 적중": hits, "같은예산 정밀도": round(prec, 4),
                      "자체 p≥0.15 픽": int((dc["prob"] >= 0.15).sum()),
                      "자체 p≥0.15 정밀도": round(float(dc.loc[dc["prob"] >= 0.15, TARGET].mean()), 4)
                      if (dc["prob"] >= 0.15).any() else float("nan"),
                      "하루 최고 prob 중앙값": round(float(dc.groupby("date")["prob"].max().median()), 4)})
            rows.append(r)
    t = pd.DataFrame(rows)
    config.ensure_dirs()
    t.to_csv(config.TABLE_DIR / "g5_recency_y_up.csv", index=False, encoding="utf-8-sig")
    monthly = daily.assign(month=daily["date"].dt.to_period("M").astype(str)).pivot_table(
        index="month", columns="cand", values="P@10", aggfunc="mean").round(4)
    monthly.to_csv(config.TABLE_DIR / "g5_recency_monthly_y_up.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for seg in segs:
        print(f"\n## {seg}")
        print(t[t["구간"] == seg].drop(columns="구간").to_string(index=False))
    print("\n## 월별 P@10")
    print(monthly.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", default="")
    sub.add_parser("check")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run([x for x in a.only.split(",") if x] or None)
    elif a.cmd == "check":
        cmd_check()
    else:
        cmd_report()


if __name__ == "__main__":
    main()
