"""KOSPI/KOSDAQ 일별 시세 수집기 (이어받기 지원).

data/raw/{MARKET}/{YYYY}/{YYYYMMDD}.parquet 로 하루치 원본을 저장한다.
휴장일은 data/raw/_calendar.json 에 기록해 다시 호출하지 않는다.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import config
from .krx_api import KRXClient


def raw_path(market: str, ds: str) -> Path:
    return config.RAW_DIR / market / ds[:4] / f"{ds}.parquet"


def _calendar_path() -> Path:
    return config.RAW_DIR / "_calendar.json"


def _load_calendar() -> dict[str, str]:
    if _calendar_path().exists():
        return json.loads(_calendar_path().read_text())
    return {}


def _save_calendar(cal: dict[str, str]) -> None:
    _calendar_path().parent.mkdir(parents=True, exist_ok=True)
    _calendar_path().write_text(json.dumps(dict(sorted(cal.items())), indent=0))


def collect(start: str, end: str | None = None, markets=("KOSPI", "KOSDAQ"),
            client: KRXClient | None = None, force: bool = False, verbose: bool = True) -> int:
    """start~end (YYYYMMDD) 평일을 순회하며 아직 없는 날짜만 받아온다. 반환값: API 호출 수."""
    config.ensure_dirs()
    client = client or KRXClient()
    end = end or date.today().strftime("%Y%m%d")
    cal = _load_calendar()
    # 최근 3일 이내 날짜는 데이터가 아직 안 올라왔을 수 있으므로 휴장으로 확정하지 않는다.
    settle = (date.today() - timedelta(days=3)).strftime("%Y%m%d")

    days = [d.strftime("%Y%m%d") for d in pd.bdate_range(start, end)]
    n_calls = 0
    for k, ds in enumerate(days):
        if cal.get(ds) == "closed" and not force:
            continue
        todo = [m for m in markets if force or not raw_path(m, ds).exists()]
        if not todo:
            continue
        for m in todo:
            df = client.daily_trades(m, ds)
            n_calls += 1
            if df.empty:
                if m == todo[0]:  # 첫 시장이 비어 있으면 휴장일(또는 미반영)로 보고 나머지 시장은 건너뜀
                    if ds <= settle:
                        cal[ds] = "closed"
                    break
                print(f"[경고] {ds} {m} 데이터가 비어 있습니다.")
                continue
            p = raw_path(m, ds)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            cal[ds] = "open"
        if verbose and (n_calls % 50 == 0 or k == len(days) - 1):
            print(f"  {ds} 까지 진행 (API 호출 {n_calls}건)")
            _save_calendar(cal)
    _save_calendar(cal)
    if verbose:
        print(f"수집 완료: API 호출 {n_calls}건")
    return n_calls


def load_raw(markets=("KOSPI", "KOSDAQ"), start: str | None = None) -> pd.DataFrame:
    frames = []
    for m in markets:
        for p in sorted((config.RAW_DIR / m).glob("*/*.parquet")):
            if start and p.stem < start:
                continue
            df = pd.read_parquet(p)
            df["MARKET"] = m
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{config.RAW_DIR} 에 원본 데이터가 없습니다. 먼저 `python -m krxlimit collect` 를 실행하세요.")
    return pd.concat(frames, ignore_index=True)


def check(bas_dd: str | None = None) -> None:
    """API 키/서비스 승인 여부를 한 번 호출해 확인한다."""
    client = KRXClient()
    for d in pd.bdate_range(end=bas_dd or date.today().strftime("%Y%m%d"), periods=10)[::-1]:
        ds = d.strftime("%Y%m%d")
        ok = True
        for m in config.ENDPOINTS:
            df = client.daily_trades(m, ds)
            print(f"{ds} {m}: {len(df)}종목")
            ok &= not df.empty
        if ok:
            print(df.head(3).to_string())
            return
    print("최근 10 영업일 모두 데이터가 비어 있습니다.")
