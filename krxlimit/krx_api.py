"""KRX Open API (openapi.krx.co.kr) 클라이언트.

- 서비스별로 KRX Open API 사이트에서 '이용신청' 승인을 받아야 호출된다.
  (유가증권 일별매매정보, 코스닥 일별매매정보)
- 요청: GET {API_BASE}/{endpoint}?basDd=YYYYMMDD, 헤더 AUTH_KEY
- 응답: {"OutBlock_1": [{BAS_DD, ISU_CD, ISU_NM, MKT_NM, SECT_TP_NM, TDD_CLSPRC,
         CMPPREVDD_PRC, FLUC_RT, TDD_OPNPRC, TDD_HGPRC, TDD_LWPRC, ACC_TRDVOL,
         ACC_TRDVAL, MKTCAP, LIST_SHRS}, ...]}
- 일 호출 한도(약 10,000건)가 있으므로 수집기는 날짜별 파일 캐시를 두고 이어받기를 지원한다.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from . import config


class KRXAPIError(RuntimeError):
    pass


class KRXClient:
    def __init__(self, api_key: str | None = None, sleep_sec: float | None = None,
                 timeout: float = 30.0, max_retries: int = 4):
        self.api_key = api_key or config.API_KEY
        if not self.api_key:
            raise KRXAPIError(
                "KRX_API_KEY 가 없습니다. 프로젝트 루트의 .env 파일에 KRX_API_KEY=발급키 를 적어 주세요."
            )
        self.sleep_sec = config.API_SLEEP_SEC if sleep_sec is None else sleep_sec
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"AUTH_KEY": self.api_key})

    def get(self, endpoint: str, bas_dd: str) -> list[dict]:
        url = f"{config.API_BASE}/{endpoint}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params={"basDd": bas_dd}, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    raise KRXAPIError(
                        f"{endpoint} 인증 실패(HTTP {resp.status_code}). API 키와 해당 서비스 이용신청 승인 여부를 확인하세요. "
                        f"응답: {resp.text[:200]}"
                    )
                resp.raise_for_status()
                payload = resp.json()
                if "OutBlock_1" not in payload:
                    # 에러 응답 예: {"respCode": "401", "respMsg": "Unauthorized API Call"}
                    raise KRXAPIError(f"{endpoint} {bas_dd} 비정상 응답: {str(payload)[:300]}")
                time.sleep(self.sleep_sec)
                return payload["OutBlock_1"] or []
            except KRXAPIError:
                raise
            except (requests.RequestException, ValueError) as e:  # 네트워크 오류/JSON 파싱 오류는 재시도
                last_err = e
                time.sleep(2 ** (attempt + 1))
        raise KRXAPIError(f"{endpoint} {bas_dd} 요청 실패: {last_err}")

    def daily_trades(self, market: str, bas_dd: str) -> pd.DataFrame:
        """해당 일자 시장 전 종목 일별 시세 (원본 문자열 그대로)."""
        rows = self.get(config.ENDPOINTS[market], bas_dd)
        return pd.DataFrame(rows, dtype=str)
