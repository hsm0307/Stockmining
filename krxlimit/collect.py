"""KOSPI/KOSDAQ 일별 시세 수집기 (이어받기 지원).

data/raw/{MARKET}/{YYYY}/{YYYYMMDD}.parquet 로 하루치 원본을 저장한다.
휴장일은 data/raw/_calendar.json 에 기록해 다시 호출하지 않는다.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            client: KRXClient | None = None, force: bool = False, verbose: bool = True,
            workers: int = 6) -> int:
    """start~end (YYYYMMDD) 평일 중 아직 없는 날짜만 받아온다(날짜 단위 병렬). 반환값: API 호출 수.

    KRX 응답이 호출당 약 2초라 순차 수집은 10년치에 3시간가량 걸린다 → 기본 6개 스레드로 병렬 수집.
    """
    config.ensure_dirs()
    end = end or date.today().strftime("%Y%m%d")
    cal = _load_calendar()
    # 최근 3일 이내 날짜는 데이터가 아직 안 올라왔을 수 있으므로 휴장으로 확정하지 않는다.
    settle = (date.today() - timedelta(days=3)).strftime("%Y%m%d")
    days = [d.strftime("%Y%m%d") for d in pd.bdate_range(start, end)]
    pending = [ds for ds in days if (force or cal.get(ds) != "closed")
               and any(force or not raw_path(m, ds).exists() for m in markets)]
    if not pending:
        if verbose:
            print("새로 받을 날짜가 없습니다.")
        return 0

    if client is None:
        KRXClient()  # 키가 없으면 스레드를 띄우기 전에 바로 오류
    local = threading.local()

    def get_client():
        if client is not None:
            return client
        if not hasattr(local, "c"):
            local.c = KRXClient()
        return local.c

    def fetch_day(ds: str):
        c = get_client()
        todo = [m for m in markets if force or not raw_path(m, ds).exists()]
        status, calls = None, 0
        for m in todo:
            df = c.daily_trades(m, ds)
            calls += 1
            if df.empty:
                if m == todo[0]:  # 첫 시장이 비어 있으면 휴장일(또는 미반영)로 보고 나머지 시장은 건너뜀
                    status = "closed" if ds <= settle else None
                    break
                print(f"[경고] {ds} {m} 데이터가 비어 있습니다.")
                continue
            p = raw_path(m, ds)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            status = "open"
        return ds, status, calls

    n_calls, done = 0, 0
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    futs = [ex.submit(fetch_day, ds) for ds in pending]
    try:
        for fut in as_completed(futs):
            ds, status, calls = fut.result()
            n_calls += calls
            done += 1
            if status:
                cal[ds] = status
            if verbose and (done % 100 == 0 or done == len(pending)):
                print(f"  {done}/{len(pending)}일 처리 (API 호출 {n_calls}건)", flush=True)
                _save_calendar(cal)
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)
        _save_calendar(cal)
        raise
    ex.shutdown()
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
