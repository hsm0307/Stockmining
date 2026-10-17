import numpy as np
import pandas as pd

from krxlimit.panel import build_panel


def _row(d, code, close, chg, o, h, l, vol, market="KOSDAQ", name="가나다"):
    return {"BAS_DD": d, "ISU_CD": code, "ISU_NM": name, "MKT_NM": market, "SECT_TP_NM": "", "TDD_CLSPRC": str(close),
            "CMPPREVDD_PRC": str(chg), "FLUC_RT": "0", "TDD_OPNPRC": str(o), "TDD_HGPRC": str(h), "TDD_LWPRC": str(l),
            "ACC_TRDVOL": str(vol), "ACC_TRDVAL": str(vol * close), "MKTCAP": str(close * 1000),
            "LIST_SHRS": "1000", "MARKET": market}


def test_limit_flags_and_streaks():
    rows = [
        _row("20240102", "111110", 10_000, 0, 10_000, 10_100, 9_900, 100),
        _row("20240103", "111110", 13_000, 3_000, 13_000, 13_000, 13_000, 5),        # 점상
        _row("20240104", "111110", 16_900, 3_900, 15_000, 16_900, 14_500, 900),      # 일반 상한가 (2연상)
        _row("20240105", "111110", 16_900, 0, 0, 0, 0, 0),                          # 거래정지
        _row("20240108", "111110", 21_950, 5_050, 21_950, 21_950, 21_000, 50),       # 3연상 (정지일 건너뜀)
        _row("20240109", "111110", 20_000, -1_950, 28_500, 28_500, 19_000, 999),     # 상한 터치 후 이탈
        # 2:1 액면분할: 기준가 10,000 (전일 종가 20,000) → 수익률은 기준가 대비
        _row("20240110", "111110", 10_500, 500, 10_000, 10_600, 9_900, 100),
        # 초저가: 가격제한폭이 0원 → 상/하한가 판정 제외
        _row("20240102", "222220", 2, 0, 2, 2, 2, 100),
        _row("20240103", "222220", 2, 0, 2, 2, 2, 100),
    ]
    p = build_panel(pd.DataFrame(rows), save=False)
    a = p[p["code"] == "111110"].set_index("date")
    assert a["up_close"].tolist() == [False, True, True, False, True, False, False]
    assert a["up_jeom"].tolist() == [False, True, False, False, False, False, False]
    assert a["up_fail"].tolist() == [False, False, False, False, False, True, False]
    assert a["up_streak"].tolist() == [0, 1, 2, 0, 3, 0, 0]
    assert abs(a["ret"].iloc[-1] - 0.05) < 1e-6          # 분할일 수익률 = 10,500 / 10,000 - 1
    assert abs(a["ri"].iloc[-1] / a["ri"].iloc[-2] - 1.05) < 1e-6
    b = p[p["code"] == "222220"]
    assert not b["limit_valid"].any() and not b["up_close"].any() and not b["down_close"].any()


def test_next_day_halt_counts_as_miss():
    """다음 날 거래정지면 라벨은 NaN 이 아니라 0(실패), 데이터 마지막 날만 NaN."""
    from krxlimit.features import build_features
    rows = [
        _row("20240102", "111110", 10_000, 0, 10_000, 10_100, 9_900, 100),
        _row("20240103", "111110", 13_000, 3_000, 13_000, 13_000, 13_000, 5),   # 상한가
        _row("20240104", "111110", 13_000, 0, 0, 0, 0, 0),                     # 거래정지
        _row("20240105", "111110", 16_900, 3_900, 15_000, 16_900, 14_500, 900),  # 상한가
        _row("20240102", "222220", 5_000, 0, 5_000, 5_100, 4_900, 100),
        _row("20240103", "222220", 5_000, 0, 5_000, 5_100, 4_900, 100),
        _row("20240104", "222220", 5_000, 0, 5_000, 5_100, 4_900, 100),
        _row("20240105", "222220", 5_000, 0, 5_000, 5_100, 4_900, 100),
    ]
    p = build_panel(pd.DataFrame(rows), save=False)
    f = build_features(p, save=False, peer=False).set_index(["code", "date"])
    a = f.loc["111110"]
    assert a.loc["2024-01-03", "y_up"] == 0          # 다음 날(01-04) 거래정지 → 실패
    assert a.loc["2024-01-02", "y_up"] == 1
    assert np.isnan(a.loc["2024-01-05", "y_up"])      # 마지막 날 → 결과 모름
    assert a.loc["2024-01-03", "next_ret"] == 0       # 정지일 수익률 0


def test_reit_detection_excludes_meritz():
    names = ["메리츠금융지주", "메리츠화재", "블리츠웨이엔터테인먼트", "신한알파리츠", "이리츠코크렙", "SK리츠"]
    rows = [_row("20240102", f"{100000 + i * 10:06d}", 5_000, 0, 5_000, 5_100, 4_900, 100, name=n) for i, n in enumerate(names)]
    p = build_panel(pd.DataFrame(rows), save=False).set_index("name")
    assert p.loc[["신한알파리츠", "이리츠코크렙", "SK리츠"], "is_reit"].all()
    assert not p.loc[["메리츠금융지주", "메리츠화재", "블리츠웨이엔터테인먼트"], "is_reit"].any()


def test_peak_analysis_censoring():
    """최고점 분석: 극점 창이 데이터 끝에 잘린 구간은 빼고, 1년 창이 끝을 넘으면 상장폐지도 NaN,
    첫날에 시작한 구간은 연상 전 가격이 없어 비교 지표가 NaN 이어야 한다."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from synthetic import make_raw
    from krxlimit.analysis_peak import PEAK_WIN, streak_extremes
    from krxlimit.events import EventContext
    panel = build_panel(make_raw(n_stocks=60, start="2020-01-01", end="2022-12-31", seed=3), save=False)
    ctx = EventContext(panel)
    for side in ("up", "down"):
        df = streak_extremes(panel, ctx, side)
        if df.empty:
            continue
        end_ti = np.searchsorted(ctx.days, df["종료일"].to_numpy())
        assert (end_ti + PEAK_WIN < ctx.T).all()
        censored = end_ti + df["극점도달(종료후일)"].to_numpy() + 250 >= ctx.T
        assert df.loc[censored, "상장폐지"].isna().all()
        assert df.loc[censored, "극점후1년_거래정지"].isna().all()
        col = "1년후_연상전보다낮음" if side == "up" else "1년내하락전가격회복"
        first = df["시작일"] == ctx.days[0]
        assert df.loc[first, col].isna().all()
        assert df.loc[~first & ~censored, col].notna().all()
