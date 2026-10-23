"""목표 지표(고확신 추천 적중률)의 백테스트 기준 성적표.

    python scripts/highconf_backtest.py            # 운영 상한가 모델의 10년 분기별 워크포워드 예측으로 계산

고확신 추천 = 예측 확률이 기준(15%, 참고 20%) 이상인 종목. 기간·연도·월별로 추천 수, 적중률, 시가 매수 가능 적중률,
하루 평균 상한가 수를 낸다. 확률 크기는 분기마다 다시 학습한 모델에 따라 크게 달라지므로(추천이 0개인 분기도 있음)
순위 기준(매일 1위, 2위까지) 적중률도 함께 낸다. 예측은 evolve 가 만든 운영 버전 캐시(ten_y_up_<key>.parquet)를 쓴다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krxlimit import config  # noqa: E402
from krxlimit.live import HC_THRESHOLD, HC_THRESHOLD_STRICT, HC_TARGET  # noqa: E402

STATE = config.OUTPUT_DIR / "evolve" / "v2-haltlabel"


def load(target: str = "y_up") -> tuple[pd.DataFrame, str]:
    st = json.loads((STATE / f"state_{target}.json").read_text(encoding="utf-8"))
    prod = st["production"]
    r = pd.read_parquet(STATE / f"ten_{target}_{prod['key']}.parquet",
                        columns=["date", "code", target, "prob", "next_open_ret"])
    r = r[r[target].notna()].copy()
    pan = pd.read_parquet(config.DATA_DIR / "processed" / "panel.parquet", columns=["date", "code", "tidx", "open", "upper"])
    pan["code"] = pan["code"].astype(str)
    r["code"] = r["code"].astype(str)
    r = r.merge(pan[["date", "code", "tidx"]], on=["date", "code"], how="left")
    nxt = pan.assign(tidx=pan["tidx"] - 1).rename(columns={"open": "n_open", "upper": "n_upper"})[["code", "tidx", "n_open", "n_upper"]]
    r = r.merge(nxt, on=["code", "tidx"], how="left")
    r["tradable"] = (r[target] == 1) & (r["n_open"] < r["n_upper"])
    return r, prod["name"]


def summarize(r: pd.DataFrame, target: str, group: pd.Series) -> pd.DataFrame:
    days = r.groupby(group)["date"].nunique()
    events = r.groupby(group)[target].sum() / days
    out = {"거래일": days, "하루 평균 상한가": events.round(2)}
    rk = r.groupby("date")["prob"].rank(ascending=False, method="first")
    for k in (1, 2):
        top = r[rk <= k]
        out[f"매일 {k}위까지 적중률"] = top.groupby(group.loc[top.index])[target].mean().reindex(days.index).round(3)
    for th, tag in [(HC_THRESHOLD[target], "기준"), (HC_THRESHOLD_STRICT, "참고")]:
        s = r[r["prob"] >= th]
        g = s.groupby(group.loc[s.index])
        n = g.size().reindex(days.index).fillna(0)
        out[f"{tag}{th:.0%} 추천/일"] = (n / days).round(2)
        out[f"{tag}{th:.0%} 추천 있는 날"] = (g["date"].nunique().reindex(days.index).fillna(0) / days).round(2)
        out[f"{tag}{th:.0%} 적중률"] = (g[target].mean().reindex(days.index)).round(3)
        out[f"{tag}{th:.0%} 시가매수가능 적중률"] = (g["tradable"].mean().reindex(days.index)).round(3)
    return pd.DataFrame(out)


def main(target: str = "y_up") -> None:
    r, name = load(target)
    r = r[r["date"] >= "2019-01-01"]
    period = pd.Series(np.select([r["date"] < "2024-01-01", r["date"] < "2026-01-01"],
                                 ["2019~2023", "2024~2025"], "2026"), index=r.index, name="기간")
    by_period = summarize(r, target, period)
    by_year = summarize(r, target, r["date"].dt.year.rename("연도"))
    by_month = summarize(r[r["date"] >= "2025-01-01"], target, r.loc[r["date"] >= "2025-01-01", "date"].dt.to_period("M").astype(str).rename("월"))
    config.ensure_dirs()
    by_period.to_csv(config.TABLE_DIR / f"g5_highconf_{target}.csv", encoding="utf-8-sig")
    by_year.to_csv(config.TABLE_DIR / f"g5_highconf_yearly_{target}.csv", encoding="utf-8-sig")
    by_month.to_csv(config.TABLE_DIR / f"g5_highconf_monthly_{target}.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(f"모델 {name}, 목표 적중률 {HC_TARGET:.0%} (확률 {HC_THRESHOLD[target]:.0%} 이상)")
    for t in (by_period, by_year, by_month.tail(10)):
        print(t.to_string()); print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y_up")
