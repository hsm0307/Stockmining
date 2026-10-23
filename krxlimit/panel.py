"""원본 일별 시세 → 분석용 패널(종목×일자) 생성.

기준가격 = 종가 − 대비(CMPPREVDD_PRC)를 역산하고 close/base를 연결한 ri를 만든다.
ri는 기준가격을 연결한 가격수익률 대용치이며, 기업행위가 검증된 수정주가나
현금배당·권리대금 등을 포함한 총수익률이 아니다. 대비 결측은 0으로 가정하지 않는다.
수익률을 사용할 때는 EventContext 등에서 해당 구간의 관측과 연결 가능 여부를 확인한다.
상·하한가 라벨은 역산 기준가격에 보통 주식의 호가단위·가격제한 규칙을 적용한 결과로,
특수 거래 상태까지 확인한 공식 사건 라벨이 아니다.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import config
from .collect import load_raw
from .ticks import limit_prices

RENAME = {
    "BAS_DD": "date", "ISU_CD": "code", "ISU_NM": "name", "MKT_NM": "mkt_nm",
    "SECT_TP_NM": "sect", "TDD_CLSPRC": "close", "CMPPREVDD_PRC": "chg", "FLUC_RT": "fluc",
    "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "ACC_TRDVOL": "volume",
    "ACC_TRDVAL": "value", "MKTCAP": "mktcap", "LIST_SHRS": "shares", "MARKET": "market",
}
NUM_COLS = ["close", "chg", "fluc", "open", "high", "low", "volume", "value", "mktcap", "shares"]

# "exact": 계산한 상한가와 종가가 정확히 일치할 때만 인정 (기본)
# "loose": exact 또는 (등락률 ≥ 29.5% & 종가 = 고가) 도 인정 → 규정 계산이 어긋날 때 비교용
LIMIT_RULE = os.environ.get("KRX_LIMIT_RULE", "exact")
LIQ_DAYS = 7    # 정리매매 기간(그 종목의 거래일 수)

_PREF_NAME = re.compile(r"(?:\d?우[A-Z]?|우\(전환\))$")


def _to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.replace({"": None, "-": None, "None": None, "nan": None}), errors="coerce")


def _normalize_code(code: pd.Series) -> pd.Series:
    code = code.astype(str).str.strip()
    std = code.str.len().eq(12) & code.str.startswith("KR")  # 표준코드(KR7005930003) → 단축코드
    return code.where(~std, code.str.slice(3, 9))


def _run_length(flag: pd.Series, key: pd.Series) -> pd.Series:
    """key(종목) 안에서 flag 가 연속으로 True 인 길이 (현재 행 포함). flag 는 정렬돼 있어야 한다."""
    f = flag.to_numpy(dtype=bool)
    k = key.to_numpy()
    brk = np.r_[True, (k[1:] != k[:-1]) | (f[1:] != f[:-1])]
    run_id = np.cumsum(brk)
    out = pd.Series(f.astype(np.int32), index=flag.index).groupby(run_id).cumsum()
    return out.where(flag, 0).astype(np.int32)


def build_panel(raw: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    raw = load_raw() if raw is None else raw
    df = raw.rename(columns=RENAME)[list(RENAME.values())].copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df["code"] = _normalize_code(df["code"])
    for c in NUM_COLS:
        df[c] = _to_num(df[c])
    df["sect"] = df["sect"].fillna("").astype(str)
    df["name"] = df["name"].astype(str)
    df = (df.dropna(subset=["close"])
            .drop_duplicates(["date", "code"], keep="last")
            .sort_values(["code", "date"], kind="mergesort")
            .reset_index(drop=True))

    # ---- 기준가, 거래 여부, 상장 첫날
    # Unknown reference changes must remain unknown: imputing zero fabricates a
    # flat daily link and can turn an unavailable holding return into 0%.
    df["base"] = df["close"] - df["chg"]
    df["traded"] = (df["volume"].fillna(0) > 0) & (df["open"].fillna(0) > 0)
    first_date = df.groupby("code", sort=False)["date"].transform("min")
    df["is_first_day"] = (df["date"] == first_date) & (first_date > df["date"].min())

    ok_base = (df["base"] > 0) & ~df["is_first_day"]
    df["ret"] = np.where(ok_base, df["close"] / df["base"].where(ok_base) - 1, np.nan).astype("float32")

    # ---- 상·하한가
    df["is_kosdaq"] = df["market"].eq("KOSDAQ")
    upper = np.full(len(df), np.nan)
    lower = np.full(len(df), np.nan)
    m = ok_base.to_numpy()
    up, lo = limit_prices(df.loc[m, "base"].to_numpy(), df.loc[m, "date"].to_numpy(),
                          df.loc[m, "is_kosdaq"].to_numpy())
    upper[m], lower[m] = up, lo
    df["upper"], df["lower"] = upper, lower

    # 정리매매·재상장 등 가격제한이 없는 날은 제외. 종가가 ±30% 안이어도 장중 고가·저가가 계산된 가격제한 밖이면
    # 그날은 가격제한 없이 거래된 것이다(제한이 있으면 그 범위 밖 체결은 불가능) → 우연히 종가가 계산상 상·하한가와
    # 같아도 상·하한가로 세지 않는다.
    in_range = (df["high"] <= df["upper"]) & (df["low"] >= df["lower"])
    point_valid = (df["traded"] & ok_base & (df["ret"].abs() <= 0.305) & (df["upper"] > df["base"]) & in_range)
    # 정리매매(7거래일) 중 우연히 범위 안에서 움직인 날도 가격제한이 없다. 가격제한 없이 거래된 날부터 그 종목의
    # 거래일로 LIQ_DAYS 일(그날 포함, 사이의 거래정지일도 포함)을 '가격제한 없는 기간'(nolimit_window)으로 보고 뺀다.
    # 그날까지의 정보만 쓰므로 학습과 실전의 판정이 같고, 가처분으로 멈췄다가 재개된 정리매매도 잡는다.
    # 10년 자료에서 이 기간에 빠지는 상·하한가는 모두 정리매매 중 종가가 우연히 계산상 한가와 같았던 기록이었다.
    traded_n = df["traded"].astype(np.int32).groupby(df["code"], sort=False).cumsum()
    start = traded_n.where(df["traded"] & ok_base & ~point_valid).groupby(df["code"], sort=False).ffill()
    df["nolimit_window"] = (traded_n - start) <= LIQ_DAYS - 1     # 가격제한 없는 날이 없었으면 NaN → False
    df["limit_valid"] = point_valid & ~df["nolimit_window"]
    v = df["limit_valid"]
    df["up_close"] = v & (df["close"] == df["upper"])
    df["down_close"] = v & (df["close"] == df["lower"])
    if LIMIT_RULE == "loose":
        df["up_close"] |= v & (df["ret"] >= 0.295) & (df["close"] == df["high"])
        df["down_close"] |= v & (df["ret"] <= -0.295) & (df["close"] == df["low"])
    df["up_touch"] = v & ((df["high"] >= df["upper"]) | df["up_close"])
    df["down_touch"] = v & ((df["low"] <= df["lower"]) | df["down_close"])
    df["up_jeom"] = df["up_close"] & (df["open"] == df["close"]) & (df["low"] == df["close"])
    df["down_jeom"] = df["down_close"] & (df["open"] == df["close"]) & (df["high"] == df["close"])
    df["up_fail"] = df["up_touch"] & ~df["up_close"]      # 장중 상한가 터치 후 이탈
    df["down_fail"] = df["down_touch"] & ~df["down_close"]  # 장중 하한가 터치 후 회복

    # ---- 종목 유형
    numeric_code = df["code"].str.fullmatch(r"\d{6}")
    df["is_pref"] = np.where(numeric_code, ~df["code"].str.endswith("0"),
                             df["name"].str.contains(_PREF_NAME))
    df["is_spac"] = df["name"].str.contains("스팩|SPAC", regex=True)
    df["is_reit"] = df["name"].str.contains("(?<![메블])리츠|REIT", regex=True)  # 메리츠·블리츠웨이 등 제외
    df["is_mgmt"] = df["sect"].str.contains("관리")
    df["is_caution"] = df["sect"].str.contains("투자주의|환기")

    # ---- 기준가격 연결 지수, 연속 상/하한가
    # ri has an arbitrary level across an unknown link. Keep ret missing so the
    # event-return path checks reject intervals spanning that link.
    df["ri"] = (1 + df["ret"].fillna(0).astype("float64")).groupby(df["code"], sort=False).cumprod()
    tr = df["traded"]
    sub = df.loc[tr, ["code", "up_close", "down_close"]]
    df["up_streak"] = 0
    df["down_streak"] = 0
    df.loc[tr, "up_streak"] = _run_length(sub["up_close"], sub["code"]).to_numpy()
    df.loc[tr, "down_streak"] = _run_length(sub["down_close"], sub["code"]).to_numpy()

    # ---- 전체 거래일 인덱스
    days = np.sort(df["date"].unique())
    df["tidx"] = np.searchsorted(days, df["date"].to_numpy()).astype(np.int32)

    for c in ["ret", "fluc"]:
        df[c] = df[c].astype("float32")
    for c in ["code", "market", "mkt_nm", "sect"]:
        df[c] = df[c].astype("category")

    if save:
        config.ensure_dirs()
        df.to_parquet(config.PANEL_PATH, index=False)
        print(f"패널 저장: {config.PANEL_PATH} ({len(df):,}행, {df['code'].nunique():,}종목, "
              f"{df['date'].min():%Y-%m-%d}~{df['date'].max():%Y-%m-%d})")
    return df


def load_panel() -> pd.DataFrame:
    if not config.PANEL_PATH.exists():
        raise FileNotFoundError("패널이 없습니다. 먼저 `python -m krxlimit build` 를 실행하세요.")
    return pd.read_parquet(config.PANEL_PATH)


def validate_limits(df: pd.DataFrame) -> pd.DataFrame:
    """규정 계산(exact)과 등락률 기반 판정의 일치도 점검표. build 후 반드시 확인할 것."""
    v = df[df["limit_valid"]]
    near_up = (v["ret"] >= 0.295) & (v["close"] == v["high"])
    near_dn = (v["ret"] <= -0.295) & (v["close"] == v["low"])
    rows = [
        ("상한가(규정계산)", int(v["up_close"].sum())),
        ("등락률≥29.5% & 종가=고가", int(near_up.sum())),
        ("상한가: 둘 다 해당", int((v["up_close"] & near_up).sum())),
        ("상한가: 등락률 기준만 해당(누락 의심)", int((~v["up_close"] & near_up).sum())),
        ("상한가: 규정계산만 해당(등락률<29.5%)", int((v["up_close"] & ~near_up).sum())),
        ("하한가(규정계산)", int(v["down_close"].sum())),
        ("등락률≤-29.5% & 종가=저가", int(near_dn.sum())),
        ("하한가: 등락률 기준만 해당(누락 의심)", int((~v["down_close"] & near_dn).sum())),
        ("하한가: 규정계산만 해당(등락률>-29.5%)", int((v["down_close"] & ~near_dn).sum())),
        ("|등락률|>30.5% (가격제한 없음, 제외)", int((df["traded"] & (df["ret"].abs() > 0.305)).sum())),
    ]
    out = pd.DataFrame(rows, columns=["항목", "건수"])
    miss = v[~v["up_close"] & near_up]
    if len(miss):
        print("누락 의심 예시 (상한가):")
        print(miss[["date", "code", "name", "market", "base", "close", "high", "upper", "fluc"]].head(10).to_string())
    return out


def to_wide(df: pd.DataFrame, col: str, days: np.ndarray | None = None) -> pd.DataFrame:
    """long 패널 → (일자 × 종목) 행렬."""
    w = df.pivot(index="date", columns="code", values=col)
    w.columns = w.columns.astype(str)
    if days is not None:
        w = w.reindex(days)
    return w

