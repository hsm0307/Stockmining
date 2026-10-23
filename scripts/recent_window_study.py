"""최근 데이터 위주 학습(학습 창 축소) 비교: 상한가 v4 설정으로 학습 기간만 바꾼다.

    python scripts/recent_window_study.py train --months 6      # 최근 6개월만 학습한 y_up_recent6m 저장
    python scripts/recent_window_study.py backtest              # 2025-01~ 분기별 워크포워드 비교표

backtest 의 '전체 누적'은 evolve 가 만든 v4 10년 기준선(같은 분기 재학습)을 그대로 쓰고,
12개월·6개월 창만 새로 학습한다. 분기별 예측은 outputs/experiments/recent_window/ 에 저장해
중간에 끊겨도 이어서 돌린다. 데이터가 바뀌면 그 폴더를 지우고 다시 돌린다.
비교용 실험이며 운영 버전 승격과는 무관하다.
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
from krxlimit.__main__ import _features  # noqa: E402
from krxlimit.model import fit, predict_proba, select_features, train_final, universe_mask  # noqa: E402

CFG_PATH = ROOT / "configs" / "y_up_v4.json"
CACHE = config.OUTPUT_DIR / "experiments" / "recent_window"
WINDOWS = (12, 6)


def _cfg():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    return cfg, select_features(only_groups=cfg["groups"])


def cmd_train(months: int) -> None:
    f = _features()
    cfg, feats = _cfg()
    sub = f[f["date"] >= f["date"].max() - pd.DateOffset(months=months)].reset_index(drop=True)
    print(f"학습 구간: {sub['date'].min().date()} ~ {sub['date'].max().date()}, "
          f"행 {len(sub):,}, 양성 {int(sub['y_up'].sum()):,}")
    train_final(sub, "y_up", kind=cfg["kind"], params=cfg["params"] or None, neg_rate=cfg["neg_rate"],
                exclude=tuple(cfg["exclude"]), name=f"y_up_recent{months}m", features=feats,
                train_years=None, hard_neg=cfg.get("hard_neg", False))


def _baseline() -> pd.DataFrame:
    files = sorted((config.OUTPUT_DIR / "evolve").glob("*/ten_y_up_*.parquet"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("v4 10년 기준선이 없습니다. evolve 를 먼저 돌리세요.")
    print(f"전체 누적 기준선: {files[-1].relative_to(ROOT)}")
    return pd.read_parquet(files[-1], columns=["date", "code", "y_up", "prob"])


def cmd_backtest(start: str) -> None:
    f = _features()
    cfg, feats = _cfg()
    base = f["y_up"].notna().to_numpy() & universe_mask(f, tuple(cfg["exclude"]), "y_up")
    dates, tidx = f["date"].to_numpy(), f["tidx"].to_numpy()
    bounds = list(pd.date_range(start, f["date"].max(), freq="QS")) + [f["date"].max() + pd.Timedelta(days=1)]
    CACHE.mkdir(parents=True, exist_ok=True)
    parts = []
    for months in WINDOWS:
        for ps, pe in zip(bounds[:-1], bounds[1:]):
            path = CACHE / f"{months}m_{ps:%Y%m%d}.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path))
                continue
            test_m = base & (dates >= np.datetime64(ps)) & (dates < np.datetime64(pe))
            t0 = tidx[test_m].min()
            train_m = base & (tidx < t0) & (dates >= np.datetime64(ps - pd.DateOffset(months=months)))
            t = time.time()
            model = fit(f, "y_up", feats, cfg["kind"], cfg["params"] or None, cfg["neg_rate"], mask=train_m,
                        hard_neg=cfg.get("hard_neg", False))
            pr = f.loc[test_m, ["date", "code", "y_up"]].copy()
            pr["prob"] = predict_proba(model, f.loc[test_m, feats], cfg["neg_rate"])
            pr["window"] = f"{months}m"
            pr.to_parquet(path, index=False)
            parts.append(pr)
            print(f"{months}개월 창 {ps:%Y-%m}: 학습 {int(train_m.sum()):,}행 → 예측 {len(pr):,}행, {time.time() - t:.0f}초",
                  flush=True)
    allh = _baseline()
    allh = allh[(allh["date"] >= start) & allh["y_up"].notna()].assign(window="all")
    d = pd.concat([allh, *parts], ignore_index=True)
    daily = (d.sort_values("prob", ascending=False).groupby(["window", "date"]).head(10)
              .groupby(["window", "date"])["y_up"].mean().unstack(0).dropna())
    rows = []
    for name, m in [(f"{start[:7]}~ 전체", slice(None)), ("2026년", daily.index >= "2026-01-01"),
                    ("2026-07 이후", daily.index >= "2026-07-01")]:
        x = daily.loc[m]
        r = {"구간": name, "일수": len(x), **{f"P@10 {w}": round(x[w].mean(), 4) for w in ["all", "12m", "6m"]}}
        for w in ["12m", "6m"]:
            diff = x[w] - x["all"]
            r[f"ΔP@10 {w}−all"] = round(diff.mean(), 4)
            r[f"t {w}"] = round(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))), 2)
        rows.append(r)
    table = pd.DataFrame(rows)
    quarterly = daily.groupby(daily.index.to_period("Q")).mean().round(4).reset_index()
    quarterly["date"] = quarterly["date"].astype(str)
    quarterly = quarterly.rename(columns={"date": "분기", "all": "전체 누적", "12m": "최근 12개월", "6m": "최근 6개월"})
    config.ensure_dirs()
    table.to_csv(config.TABLE_DIR / "g5_recent_window_y_up.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(config.TABLE_DIR / "g5_recent_window_quarterly_y_up.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(quarterly.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--months", type=int, default=6)
    b = sub.add_parser("backtest")
    b.add_argument("--start", default="2025-01-01")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a.months)
    else:
        cmd_backtest(a.start)


if __name__ == "__main__":
    main()
