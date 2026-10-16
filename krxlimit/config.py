"""프로젝트 전역 설정 (경로, 기간, 규정 상수).

값은 환경변수(.env)로 덮어쓸 수 있다. API 키는 절대 코드에 하드코딩하지 말 것.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """python-dotenv 없이 .env 파일의 KEY=VALUE 를 환경변수로 읽는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- 경로
DATA_DIR = Path(os.environ.get("KRX_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = Path(os.environ.get("KRX_OUTPUT_DIR", ROOT / "outputs"))
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = Path(os.environ.get("KRX_MODEL_DIR", ROOT / "models"))
PRED_DIR = Path(os.environ.get("KRX_PRED_DIR", ROOT / "predictions"))

PANEL_PATH = PROCESSED_DIR / "panel.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# ---------------------------------------------------------------- KRX Open API
API_KEY = os.environ.get("KRX_API_KEY", "")
API_BASE = "https://data-dbg.krx.co.kr/svc/apis"
ENDPOINTS = {
    "KOSPI": "sto/stk_bydd_trd",   # 유가증권 일별매매정보
    "KOSDAQ": "sto/ksq_bydd_trd",  # 코스닥 일별매매정보
}
API_SLEEP_SEC = float(os.environ.get("KRX_API_SLEEP", "0.15"))

# ---------------------------------------------------------------- 분석 기간
# 2015-06-15 가격제한폭이 ±15% → ±30% 로 확대되었으므로 2016년부터 쓰면 한 가지 규정만 다루게 된다.
DEFAULT_START = os.environ.get("KRX_START", "20160101")

# ---------------------------------------------------------------- 시장 규정
LIMIT_30_FROM = "2015-06-15"      # 이 날부터 가격제한폭 30%
TICK_UNIFIED_FROM = "2023-01-25"  # 이 날부터 유가/코스닥 호가가격단위 통일

# 사후 수익률 측정 구간(거래일): 익일, 1주, 1달, 3달, 반년, 1년
HORIZONS = {"1d": 1, "1w": 5, "1m": 20, "3m": 60, "6m": 120, "1y": 250}

# 왕복 거래비용 가정 (매도세 약 0.15~0.25% + 수수료/슬리피지). 순수익률 계산에만 사용.
ROUND_TRIP_COST = float(os.environ.get("KRX_ROUND_TRIP_COST", "0.0025"))

RANDOM_SEED = 42


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
