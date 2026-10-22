"""상한가 예측을 종목 구간별로 다르게 다루면 나아지는가: 시총 300억 미만 / 우선주 / 나머지.

    python scripts/segment_model_study.py run      # 구간별 모델 워크포워드 (분기별 저장, 끊겨도 이어서)
    python scripts/segment_model_study.py report   # 방법별 P@k 비교표

기준(G)은 evolve 가 만든 운영 v5 의 분기별 워크포워드 예측(ten_y_up_<v5 key>.parquet)을 그대로 쓴다.
비교하는 방법 (모두 예측일 D 장 마감까지 아는 정보만 쓴다):
  S1 구간 전용 모델: 구간마다 v5 설정으로 따로 학습해 그 구간 종목만 예측, 확률을 합쳐 전체 순위
  S2 혼합: (G + S1) / 2
  S3 구간별 재보정: 직전 8분기 G 의 표본외 예측(일별 상위 30위 안)에서 구간별 '실제 적중 / 예측 확률 합' 배수를
     구해 G 확률에 곱한다(0.5~2배로 제한)
  Q  구간별 할당: 직전 8분기 실제 상한가의 구간 비중대로 상위 10개 자리를 나눈다
  R  규칙 제외: 최근 7거래일 안에 가격제한 없는 날(상장 첫날 제외)이 있었던 종목(정리매매 등)을 뺀다
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
CACHE = config.OUTPUT_DIR / "experiments" / "segment_models"
SMALL_CAP = 3e10          # 300억 원
SEGS = ("pref", "small", "rest")
SEG_KO = {"pref": "우선주", "small": "300억 미만", "rest": "나머지"}
START, HOLD = "2019-01-01", "2024-01-01"


def _cfg() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def _v5_key() -> str:
    st = json.loads((STATE / "state_y_up.json").read_text(encoding="utf-8"))
    return st["production"]["key"]


def segment_of(df: pd.DataFrame) -> np.ndarray:
    """예측일 D 의 구간. 시총은 D 종가 기준(log_mcap = log(시가총액 원))."""
    mcap = np.exp(df["log_mcap"].to_numpy(dtype=float))
    pref = df["is_pref"].to_numpy() == 1
    return np.where(pref, "pref", np.where(np.nan_to_num(mcap, nan=np.inf) < SMALL_CAP, "small", "rest"))


def cmd_run(start: str, quarters: int | None) -> None:
    from krxlimit.__main__ import _features
    from krxlimit.model import fit, predict_proba, select_features, universe_mask
    cfg = _cfg()
    feats = select_features(only_groups=cfg["groups"])
    f = _features()
    seg = segment_of(f)
    base = f["y_up"].notna().to_numpy() & universe_mask(f, tuple(cfg["exclude"]))
    dates, tidx, y = f["date"].to_numpy(), f["tidx"].to_numpy(), f["y_up"].to_numpy()
    bounds = list(pd.date_range(start, f["date"].max(), freq="QS")) + [f["date"].max() + pd.Timedelta(days=1)]
    pairs = list(zip(bounds[:-1], bounds[1:]))[: quarters or None]
    CACHE.mkdir(parents=True, exist_ok=True)
    for ps, pe in pairs:
        test_q = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
        t0 = tidx[test_q].min()
        train_q = base & (tidx < t0)  # t0-1 행의 라벨은 t0 장 마감에 확정 → 사용 가능 (model.walk_forward 와 같은 규칙)
        if cfg["train_years"]:
            train_q &= dates >= np.datetime64(ps - pd.DateOffset(years=cfg["train_years"]))
        for s in SEGS:
            path = CACHE / f"{s}_{ps:%Y%m%d}.parquet"
            if path.exists():
                continue
            tr, te = train_q & (seg == s), test_q & (seg == s)
            n_pos = int(np.nansum(y[tr]))
            t = time.time()
            out = f.loc[te, ["date", "code"]].copy()
            if n_pos < 20 or not te.any():
                out["p_seg"] = np.nan
            else:
                model = fit(f, "y_up", feats, cfg["kind"], cfg["params"] or None, cfg["neg_rate"], mask=tr,
                            hard_neg=cfg.get("hard_neg", False))
                out["p_seg"] = predict_proba(model, f.loc[te, feats], cfg["neg_rate"])
            out["seg"] = s
            out.to_parquet(path, index=False)
            print(f"{ps:%Y-%m} {SEG_KO[s]}: 학습 {int(tr.sum()):,}행(양성 {n_pos:,}) → 예측 {int(te.sum()):,}행, "
                  f"{time.time() - t:.0f}초", flush=True)


def _nolimit7() -> pd.DataFrame:
    pan = pd.read_parquet(config.DATA_DIR / "processed" / "panel.parquet",
                          columns=["date", "code", "tidx", "limit_valid", "traded", "is_first_day"])
    pan["code"] = pan["code"].astype(str)
    pan = pan.sort_values(["code", "tidx"])
    nolim = (pan["traded"].astype(bool) & ~pan["limit_valid"].astype(bool) & ~pan["is_first_day"].astype(bool))
    pan["nolimit7"] = nolim.astype(int).groupby(pan["code"]).transform(lambda s: s.rolling(7, min_periods=1).max())
    return pan[["date", "code", "nolimit7"]]


def _topk(d: pd.DataFrame, col: str, k: int, mask: np.ndarray | None = None) -> pd.Series:
    x = d if mask is None else d[mask]
    x = x.sort_values(["date", col], ascending=[True, False], kind="mergesort")
    return x.groupby("date", sort=True).head(k).groupby("date")["y_up"].mean()


def _quota_top10(d: pd.DataFrame, q_bounds) -> pd.Series:
    """직전 8분기 실제 상한가의 구간 비중대로 10자리를 나눠 G 확률 순으로 채운다(모자라면 전체 순위로 채움)."""
    res = {}
    for ps, pe in q_bounds:
        hist = d[(d["date"] < ps) & (d["date"] >= ps - pd.DateOffset(years=2))]
        share = hist.groupby("seg")["y_up"].sum()
        share = share / share.sum() if share.sum() > 0 else pd.Series({s: 1 / 3 for s in SEGS})
        alloc = {s: int(round(10 * share.get(s, 0))) for s in SEGS}
        cur = d[(d["date"] >= ps) & (d["date"] < pe)]
        for dt, g in cur.groupby("date"):
            g = g.sort_values("p_G", ascending=False, kind="mergesort")
            pick = []
            for s in SEGS:
                pick += list(g[g["seg"] == s].index[: alloc[s]])
            if len(pick) < 10:
                pick += [i for i in g.index if i not in set(pick)][: 10 - len(pick)]
            res[dt] = g.loc[pick[:10], "y_up"].mean()
    return pd.Series(res).sort_index()


def cmd_report() -> None:
    key = _v5_key()
    g = pd.read_parquet(STATE / f"ten_y_up_{key}.parquet", columns=["date", "code", "y_up", "prob"])
    g = g[g["y_up"].notna()].rename(columns={"prob": "p_G"})
    g["code"] = g["code"].astype(str)
    feats = pd.read_parquet(config.DATA_DIR / "processed" / "features.parquet",
                            columns=["date", "code", "log_mcap", "is_pref"])
    feats["code"] = feats["code"].astype(str)
    feats["seg"] = segment_of(feats)
    g = g.merge(feats[["date", "code", "seg"]], on=["date", "code"], how="left")
    parts = [pd.read_parquet(p) for p in sorted(CACHE.glob("*_*.parquet"))]
    s1 = pd.concat(parts, ignore_index=True)[["date", "code", "p_seg"]]
    s1["code"] = s1["code"].astype(str)
    d = g.merge(s1, on=["date", "code"], how="left").merge(_nolimit7(), on=["date", "code"], how="left")
    covered = d.loc[d["date"] >= START, "p_seg"].notna().mean()
    last_q = pd.Timestamp(max(p.stem.split("_")[1] for p in CACHE.glob("*_*.parquet")))
    end = last_q + pd.offsets.QuarterBegin(startingMonth=1)
    print(f"구간 모델 예측 범위: {START} ~ {end.date()} (표본외 행 중 구간 예측 있음 {covered:.1%})")
    d["p_S1"] = d["p_seg"].fillna(d["p_G"])
    d["p_S2"] = 0.5 * d["p_G"] + 0.5 * d["p_S1"]

    # S3: 구간별 재보정 배수 (직전 8분기, 일별 상위 30위 안의 G 표본외 예측으로 계산)
    d["rk_G"] = d.groupby("date")["p_G"].rank(ascending=False, method="first")
    q_bounds = list(zip(pd.date_range(START, end, freq="QS")[:-1], pd.date_range(START, end, freq="QS")[1:]))
    d["mult"] = np.nan
    mult_log = []
    for ps, pe in q_bounds:
        hist = d[(d["date"] < ps) & (d["date"] >= ps - pd.DateOffset(years=2)) & (d["rk_G"] <= 30)]
        m = (hist.groupby("seg")["y_up"].sum() / hist.groupby("seg")["p_G"].sum()).clip(0.5, 2.0)
        sel = (d["date"] >= ps) & (d["date"] < pe)
        d.loc[sel, "mult"] = d.loc[sel, "seg"].map(m).fillna(1.0)
        mult_log.append({"분기": f"{ps:%Y-%m}", **{SEG_KO[s]: round(float(m.get(s, np.nan)), 2) for s in SEGS}})
    d["p_S3"] = d["p_G"] * d["mult"]

    d = d[(d["date"] >= START) & (d["date"] < end)].copy()
    keep_r = d["nolimit7"].fillna(0).to_numpy() == 0
    days = pd.Index(sorted(d["date"].unique()))
    methods = {"G 운영 v5": ("p_G", None), "S1 구간 전용 모델": ("p_S1", None), "S2 혼합 (G+S1)/2": ("p_S2", None),
               "S3 구간별 재보정": ("p_S3", None), "R 정리매매 등 제외": ("p_G", keep_r),
               "S3 + R": ("p_S3", keep_r), "S2 + R": ("p_S2", keep_r)}
    daily = {name: _topk(d, col, 10, m).reindex(days).fillna(0) for name, (col, m) in methods.items()}
    daily["Q 구간별 할당"] = _quota_top10(d, q_bounds).reindex(days).fillna(0)
    rows = []
    for period, sel in [("개발 2019~2023", days < HOLD), ("보류 2024~", days >= HOLD), ("전체", days >= START)]:
        base = daily["G 운영 v5"][sel]
        for name, ser in daily.items():
            x = ser[sel]
            diff = x - base
            sd = diff.std(ddof=1)
            rows.append({"기간": period, "방법": name, "일수": int(sel.sum()), "P@10": round(x.mean(), 4),
                         "ΔP@10": round(diff.mean(), 4),
                         "t": round(diff.mean() / (sd / np.sqrt(len(diff))), 2) if sd > 0 else 0.0})
    table = pd.DataFrame(rows)
    # 구간 안 순위 품질: 같은 구간 안에서 상위 3개 적중률 (G vs S1)
    inner = []
    for s in SEGS:
        m = (d["seg"] == s).to_numpy()
        for period, lo, hi in [("개발", START, HOLD), ("보류", HOLD, "2100-01-01")]:
            pm = m & (d["date"] >= lo).to_numpy() & (d["date"] < hi).to_numpy()
            a, b = _topk(d, "p_G", 3, pm), _topk(d, "p_S1", 3, pm)
            diff = (b - a).dropna()
            inner.append({"구간": SEG_KO[s], "기간": period, "일수": len(diff), "G 구간내 P@3": round(a.mean(), 4),
                          "S1 구간내 P@3": round(b.mean(), 4), "차이": round(diff.mean(), 4),
                          "t": round(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))), 2)})
    inner = pd.DataFrame(inner)
    # 상위 10개 구성
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
    table.to_csv(config.TABLE_DIR / "g5_segment_models_y_up.csv", index=False, encoding="utf-8-sig")
    inner.to_csv(config.TABLE_DIR / "g5_segment_models_inner_y_up.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(config.TABLE_DIR / "g5_segment_models_composition_y_up.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mult_log).to_csv(config.TABLE_DIR / "g5_segment_models_calibration_y_up.csv", index=False,
                                  encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(table.to_string(index=False)); print(); print(inner.to_string(index=False)); print()
    print(comp.to_string(index=False)); print(); print(pd.DataFrame(mult_log).tail(8).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", default=START)
    r.add_argument("--quarters", type=int, default=None, help="앞에서부터 이만큼 분기만 (시험용)")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.start, a.quarters)
    else:
        cmd_report()


if __name__ == "__main__":
    main()
