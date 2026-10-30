"""최신 데이터 위주 학습 실험('최신 데이터에 과적합'): 학습 창 축소·최근 가중·미세조정·혼합을 실전과 같은 방식으로 비교.

실전(train_final)과 똑같이 조기종료로 반복수를 정한 뒤 검증에 떼어 둔 마지막 60거래일까지 넣어 다시 학습한다(_refit_all).
(evolve 의 워크포워드는 다시 학습하지 않아 마지막 60거래일을 못 배운다 — 최근 자료 효과를 재려면 실전 방식이어야 함)
재학습 주기(--freq): MS=매월 1일(전 후보 선별), W-MON=매주 월요일(실전의 매일 재학습에 가까움, 결선 후보만).
각 주기 시작일에 그 전 자료로만 다시 학습해 그 주기를 예측한다(미래 정보 없음). 기준은 공식 기록 모델 v5 설정.

    python scripts/recency_study.py check                          # 가중 학습 구현이 fit+_refit_all 과 같은지 한 달로 확인
    python scripts/recency_study.py run [--freq MS] [--only base,w3m] [--start 2025-07-01] [--end 2025-07-01]
    python scripts/recency_study.py report [--freq MS]

판정 규칙(결과를 보기 전에 정함): 주 지표는 선택 구간(2025-07~2026-06)의 기준 대비 ΔP@10.
 - 병행 기록 후보: 선택 구간 일별 t ≥ 2.7(후보 13개 Bonferroni) 이고 월평균 t ≥ 2, 12개월 중 9개월 이상 ΔP@10 > 0,
   1위 적중·고정 예산(하루 0.7개) 정밀도가 기준보다 1 표준오차 넘게 나쁘지 않고, 2026-07~ 의 ΔP@10 ≥ 0.
 - 탐색 단계(낮은 문턱): 선택 구간 일별 t ≥ 2 + 같은 보조 조건. 시드만 바꾼 base_s 의 |t| 보다 확실히 커야 함.
 - 2026-07~ 은 예전 recent_window 연구에서 이미 본 구간이라 확인용이 아니라 '음수가 아님'만 본다.
"""
from __future__ import annotations

import argparse
import hashlib
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
SELECT_END = "2026-07-01"
SEED = config.RANDOM_SEED
BUDGET_PER_DAY = 0.7   # 고정 예산: 하루 평균 0.7개 (live.py 고확신 설계 0.6~0.8개/일)

# 이름: months=학습 창 개월(None=설정의 3년), half_life=최근 가중 반감기(일), finetune_months, params 덧씌움, trees 고정 나무 수,
# seed_offset=시드만 바꾼 기준(순수 재학습 잡음 크기). 가중 학습 코드를 고치면 WEIGHTED_VER 를 올린다(그 후보 캐시만 무효).
WEIGHTED_VER = "w2"
CANDS = {
    "base":     dict(months=None),                        # v5 그대로(3년, 조기종료 후 전체 재학습) — 실전과 같음
    "base_s":   dict(months=None, seed_offset=100),       # 시드만 다른 v5 — 잡음 기준
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


def load_from(start: str) -> str:
    return f"{pd.Timestamp(start) - pd.DateOffset(years=CFG['train_years'], months=6):%Y-%m-%d}"


def load(start: str) -> pd.DataFrame:
    cols = list(dict.fromkeys(["date", "code", "tidx", TARGET, "up_close", "is_nolimit"] + FEATS))
    f = pd.read_parquet(config.FEATURES_PATH, columns=cols, filters=[("date", ">=", pd.Timestamp(load_from(start)))])
    return f.sort_values(["date", "code"], kind="mergesort").reset_index(drop=True)


def cand_key(name: str, start: str) -> str:
    """캐시 파일 이름에 넣는 후보 서명: 설정·가중 코드 버전·불러온 자료 시작일(음성 표본 추출이 행 수에 따라 달라짐)."""
    spec = CANDS[name]
    raw = json.dumps({**spec, "ver": WEIGHTED_VER if spec.get("half_life") else "v1", "load_from": load_from(start)},
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:6]


def fit_weighted(f: pd.DataFrame, m: np.ndarray, w: np.ndarray | None, params: dict, trees: int | None = None,
                 valid_days: int = 60, n_bag: int = 3, seed: int = SEED):
    """fit(kind='bag3') + _refit_all 과 같은 절차에 표본 가중치를 더한 것.
    구성원 i 는 seed+i 로 음성 표본을 뽑고, 거래일이 valid_days*3 보다 많으면 마지막 valid_days 거래일로 조기종료해
    반복수를 정한 뒤(max(best, 50)) 같은 표본 전체로 다시 학습한다. 적으면 조기종료 없이 300개(trees 로 바꿀 수 있음).
    가중치는 학습마다 그 학습 행에서 평균 1 로 맞춘다(조기종료 학습은 가중이 큰 마지막 60일이 빠져 평균이 작아지므로)."""
    import lightgbm as lgb
    dates, y = f["date"].to_numpy(), f[TARGET].to_numpy()
    days = np.unique(dates[m])
    es = trees is None and len(days) > valid_days * 3
    norm = (lambda r: None) if w is None else (lambda r: w[r] / w[r].mean())
    members, iters = [], []
    for i in range(n_bag):
        s = seed + i
        sm = m & sample_mask(f, TARGET, NEG, s, False)
        if es:
            cut = days[-valid_days]
            tr, va = sm & (dates < cut), m & (dates >= cut)
            mdl = make_model("lgbm", {**params, "random_state": s})
            mdl.fit(f.loc[tr, FEATS], y[tr].astype(int), sample_weight=norm(tr),
                    eval_set=[(f.loc[va, FEATS], y[va].astype(int))], eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(150, verbose=False)])
            best = int(mdl.best_iteration_ or mdl.n_estimators)
            n = max(best, 50)
        else:
            best, n = None, trees or 300
        final = make_model("lgbm", {**params, "n_estimators": n, "random_state": s})
        final.fit(f.loc[sm, FEATS], y[sm].astype(int), sample_weight=norm(sm))
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
    """반환: (모델, 학습 마스크, 메모). 메모는 반복수·유효 표본 크기 등 기록용."""
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    m = base & (tidx < t0)  # t0-1 행의 라벨(t0일)은 t0 장 마감 때 확정 → 사용 가능 (walk_forward 와 같음)
    months = spec.get("months")
    start = ps - (pd.DateOffset(months=months) if months else pd.DateOffset(years=CFG["train_years"]))
    m &= dates >= np.datetime64(start)
    params = {**PARAMS, **spec.get("params", {})}
    seed = SEED + spec.get("seed_offset", 0)
    if spec.get("half_life"):
        age = (np.datetime64(ps) - dates).astype("timedelta64[D]").astype(float)
        w = np.power(0.5, age / spec["half_life"])
        sm0 = m & sample_mask(f, TARGET, NEG, SEED, False)
        ess = w[sm0].sum() ** 2 / (w[sm0] ** 2).sum() / sm0.sum()
        model, iters = fit_weighted(f, m, w, params)
        return model, m, f"반복수 {iters}, 유효 표본 비율 {ess:.1%}"
    if spec.get("trees"):
        model, iters = fit_weighted(f, m, None, params, trees=spec["trees"])
        return model, m, f"나무 {spec['trees']}"
    model = fit(f, TARGET, FEATS, CFG["kind"], params, NEG, mask=m, seed=seed)
    model, iters = _refit_all(model, f, m, TARGET, FEATS, params, NEG, False, seed=seed)
    if spec.get("finetune_months"):
        m_recent = m & (dates >= np.datetime64(ps - pd.DateOffset(months=spec["finetune_months"])))
        model = finetune(model, f, m_recent)
    return model, m, f"반복수 {iters}"


def _signature(f: pd.DataFrame) -> str:
    """자료가 바뀌었는지(하루 추가·특징 재생성): 2022년 이후 행 수·양성 수·마지막 날짜·특징 목록."""
    g = f[f["date"] >= "2022-01-01"]
    return json.dumps({"rows_2022": len(g), "max_date": str(f["date"].max().date()), "pos_2022": int(g[TARGET].sum()),
                       "cfg": CFG["name"], "feats": FEATS}, sort_keys=True)


def _bounds(start: str, last: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    b = list(pd.date_range(start, last, freq=freq))
    if not b or b[0] > pd.Timestamp(start):
        b = [pd.Timestamp(start)] + b
    return b + [last + pd.Timedelta(days=1)]


def cmd_run(only: list[str] | None, freq: str, start: str, end: str | None = None) -> None:
    t_all = time.time()
    f = load(start)
    print(f"자료 {len(f):,}행 {f['date'].min().date()}~{f['date'].max().date()}, 특징 {len(FEATS)}개, 주기 {freq}", flush=True)
    sig, sp = _signature(f), CACHE / "signature.json"
    if CACHE.exists() and (not sp.exists() or sp.read_text() != sig):
        shutil.rmtree(CACHE)
    (CACHE / freq).mkdir(parents=True, exist_ok=True)
    sp.write_text(sig)
    base = f[TARGET].notna().to_numpy() & universe_mask(f, tuple(CFG["exclude"]), TARGET)
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    last = min(f["date"].max(), pd.Timestamp(end) - pd.Timedelta(days=1)) if end else f["date"].max()
    bounds = _bounds(start, last, freq)
    for name in (only or list(CANDS)):
        spec, key = CANDS[name], cand_key(name, start)
        for ps, pe in zip(bounds[:-1], bounds[1:]):
            path = CACHE / freq / f"{name}_{key}_{ps:%Y%m%d}.parquet"
            if path.exists():
                continue
            test_m = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
            if not test_m.any():
                continue
            t = time.time()
            model, m, memo = train_one(f, base, tidx[test_m].min(), ps, spec)
            pr = f.loc[test_m, ["date", "code", TARGET, "up_close"]].copy()
            pr["prob"] = predict_proba(model, f.loc[test_m, FEATS], NEG)
            pr["cand"] = name
            pr.to_parquet(path, index=False)
            print(f"{name:<8} {ps:%Y-%m-%d}: 학습 {int(m.sum()):,}행(양성 {int(np.nansum(f[TARGET].to_numpy()[m])):,}) "
                  f"→ 예측 {len(pr):,}행, {time.time() - t:.0f}초 ({memo})", flush=True)
    print(f"끝 ({(time.time() - t_all) / 60:.1f}분)")


def cmd_check() -> None:
    """가중치 없이(또 가중치 1 로) fit_weighted 를 돌리면 fit(bag3)+_refit_all 과 같은 예측이 나와야 한다."""
    f = load(START)
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
    c, it_c = fit_weighted(f, m, np.ones(len(f)), PARAMS)
    pc = predict_proba(c, f.loc[test_m, FEATS], NEG)
    print("가중치 1 과의 최대 차이", float(np.abs(pa - pc).max()), it_c)


def _daily(d: pd.DataFrame) -> pd.DataFrame:
    """후보·날짜별: 1위 적중, P@5, P@10, 상한가 마감 종목 중 1위 적중(lu1)."""
    d = d.sort_values(["cand", "date", "prob"], ascending=[True, True, False])
    d["rank"] = d.groupby(["cand", "date"]).cumcount() + 1
    g = d[d["rank"] <= 10].groupby(["cand", "date"])[TARGET]
    out = pd.DataFrame({"hit1": d[d["rank"] == 1].set_index(["cand", "date"])[TARGET],
                        "P@5": d[d["rank"] <= 5].groupby(["cand", "date"])[TARGET].mean(),
                        "P@10": g.mean()})
    out["lu1"] = d[d["up_close"] == 1].groupby(["cand", "date"]).head(1).set_index(["cand", "date"])[TARGET]
    return out.reset_index()


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return round(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), 2) if len(x) >= 20 and x.std() > 0 else float("nan")


def cmd_report(freq: str) -> None:
    files = sorted((CACHE / freq).glob("*.parquet"))
    d = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    d = d[d[TARGET].notna()]
    for mix, (a, b) in MIXES.items():
        if {a, b} <= set(d["cand"]):
            x = d[d["cand"] == a].merge(d[d["cand"] == b][["date", "code", "prob"]], on=["date", "code"],
                                        suffixes=("", "_b"))
            d = pd.concat([d, x.assign(prob=(x["prob"] + x["prob_b"]) / 2, cand=mix).drop(columns="prob_b")],
                          ignore_index=True)
    daily = _daily(d)
    cands = list(dict.fromkeys(["base"] + sorted(daily["cand"].unique(), key=lambda c: list(CANDS) .index(c)
                                                 if c in CANDS else 99)))
    first = daily["date"].min()
    segs = {f"선택 {first:%Y-%m}~2026-06": (str(first.date()), SELECT_END),
            "2026-07~ (예전 연구에서 본 구간)": (SELECT_END, "2100-01-01"),
            "10/22~ (워크포워드, 실전 모델 아님)": ("2026-10-22", "2100-01-01")}
    if first < pd.Timestamp("2025-07-01"):
        segs = {f"과거 {first:%Y-%m}~2025-06": (str(first.date()), "2025-07-01"),
                "선택 2025-07~2026-06": ("2025-07-01", SELECT_END), **dict(list(segs.items())[1:])}
    rows = []
    for seg, (s, e) in segs.items():
        dd = daily[(daily["date"] >= s) & (daily["date"] < e)]
        ds = d[(d["date"] >= s) & (d["date"] < e)]
        b = dd[dd["cand"] == "base"].set_index("date")
        for c in cands:
            x = dd[dd["cand"] == c].set_index("date")
            common = x.index.intersection(b.index)
            if not len(common):
                continue
            x, bb = x.loc[common], b.loc[common]
            r = {"구간": seg, "후보": c, "일수": len(common)}
            for k in ["P@10", "hit1", "lu1"]:
                r[k] = round(x[k].mean(), 4)
            if c != "base":
                diff = x["P@10"] - bb["P@10"]
                mon = diff.groupby(diff.index.to_period("M")).mean()
                r.update({"ΔP@10": round(diff.mean(), 4), "t 일별": _t(diff),
                          "t 월평균": round(mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon))), 2) if len(mon) > 2 else float("nan"),
                          "Δ>0 월": f"{int((mon > 0).sum())}/{len(mon)}",
                          "Δhit1": round((x["hit1"] - bb["hit1"]).mean(), 4), "Δlu1": round((x["lu1"] - bb["lu1"]).mean(), 4)})
            dc = ds[(ds["cand"] == c) & ds["date"].isin(common)]
            n = int(round(BUDGET_PER_DAY * len(common)))
            top = dc.nlargest(n, "prob")
            p = float(top[TARGET].mean()) if n else float("nan")
            r.update({"예산 픽": n, "예산 정밀도": round(p, 4), "예산 SE": round(np.sqrt(p * (1 - p) / n), 4) if n else float("nan"),
                      "p≥0.15 픽": int((dc["prob"] >= 0.15).sum()),
                      "p≥0.15 정밀도": round(float(dc.loc[dc["prob"] >= 0.15, TARGET].mean()), 4) if (dc["prob"] >= 0.15).any() else float("nan"),
                      "최고 prob 중앙값": round(float(dc.groupby("date")["prob"].max().median()), 4)})
            rows.append(r)
    t = pd.DataFrame(rows)
    config.ensure_dirs()
    tag = "" if freq == "MS" else "_weekly"
    t.to_csv(config.TABLE_DIR / f"g5_recency_y_up{tag}.csv", index=False, encoding="utf-8-sig")
    monthly = daily.assign(month=daily["date"].dt.to_period("M").astype(str)).pivot_table(
        index="month", columns="cand", values="P@10", aggfunc="mean").round(4)
    monthly.to_csv(config.TABLE_DIR / f"g5_recency_monthly_y_up{tag}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    for seg in segs:
        print(f"\n## {seg}")
        print(t[t["구간"] == seg].drop(columns="구간").to_string(index=False))
    print("\n## 월별 P@10")
    print(monthly.to_string())
    sel = t[t["구간"].str.startswith("선택")].set_index("후보")
    late = t[t["구간"].str.startswith("2026-07")].set_index("후보")
    print("\n## 판정 (선택 구간 기준, 규칙은 문서 머리말)")
    noise = abs(sel.loc["base_s", "t 일별"]) if "base_s" in sel.index else float("nan")
    for c in sel.index.drop("base"):
        r = sel.loc[c]
        guard = (r["Δhit1"] > -np.sqrt(sel.loc["base", "hit1"] * (1 - sel.loc["base", "hit1"]) / r["일수"])
                 and r["예산 정밀도"] >= sel.loc["base", "예산 정밀도"] - sel.loc["base", "예산 SE"]
                 and (c not in late.index or late.loc[c, "ΔP@10"] >= 0))
        months_ok = int(str(r["Δ>0 월"]).split("/")[0]) >= 9
        verdict = ("병행 기록 후보" if r["t 일별"] >= 2.7 and r["t 월평균"] >= 2 and months_ok and guard else
                   "탐색 단계" if r["t 일별"] >= 2 and guard and r["t 일별"] > noise + 1 else "아님")
        print(f"{c:<10} ΔP@10 {r['ΔP@10']:+.4f}  t {r['t 일별']:+.2f}  월 t {r['t 월평균']:+.2f}  Δ>0 {r['Δ>0 월']}  "
              f"보조조건 {'통과' if guard else '실패'}  → {verdict}")
    print(f"(시드 잡음 base_s |t| = {noise})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--only", default="")
    r.add_argument("--freq", default="MS")
    r.add_argument("--start", default=START)
    r.add_argument("--end", default=None, help="이 날짜 전까지만 예측(과거 구간 확인용)")
    sub.add_parser("check")
    p = sub.add_parser("report")
    p.add_argument("--freq", default="MS")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run([x for x in a.only.split(",") if x] or None, a.freq, a.start, a.end)
    elif a.cmd == "check":
        cmd_check()
    else:
        cmd_report(a.freq)


if __name__ == "__main__":
    main()
