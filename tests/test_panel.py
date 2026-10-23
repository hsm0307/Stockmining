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
        assert df.loc[censored, "목표일전관측소실"].isna().all()
        assert df.loc[censored, "극점후1년_정규장미거래"].isna().all()
        col = "1년후_연상전보다낮음" if side == "up" else "1년내하락전가격회복"
        first = df["시작일"] == ctx.days[0]
        assert df.loc[first, col].isna().all()
        starts = np.searchsorted(ctx.days, df["시작일"].to_numpy())
        codes = np.searchsorted(ctx.W.codes, df["code"].astype(str).to_numpy())
        pre_observed = ctx.observed_path_mask(starts, codes, [-1])[:, 0]
        assert df.loc[~pre_observed, col].isna().all()
        assert df.loc[pre_observed & df["1년수익률관측"], col].notna().all()



def test_nolimit_flag_excludes_liquidation_trading():
    """가격제한 없이 거래된 날(정리매매 등)이 최근 7거래일 안에 있으면 is_nolimit=1 → 예측 대상에서 빠진다."""
    from krxlimit.features import build_features
    from krxlimit.panel import LIQ_DAYS as NOLIMIT_DAYS
    from krxlimit.model import universe, universe_mask
    days = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for k, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        rows.append(_row(ds, "333330", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="보통"))
        if k == 20:     # 정리매매 첫날: 기준가 10,000 → 5,000 (-50%, 가격제한 밖)
            rows.append(_row(ds, "444440", 5_000, -5_000, 6_000, 6_000, 4_800, 900, name="정리"))
        elif k > 20:    # 이후 거래는 제한폭 안에서 보합
            rows.append(_row(ds, "444440", 5_000, 0, 5_000, 5_050, 4_950, 300, name="정리"))
        else:
            rows.append(_row(ds, "444440", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="정리"))
    p = build_panel(pd.DataFrame(rows), save=False)
    f = build_features(p, save=False, peer=False)
    a = f[f["code"] == "444440"].set_index("date")["is_nolimit"]
    assert a.loc[days[19]] == 0
    assert (a.loc[days[20]:days[20 + NOLIMIT_DAYS - 1]] == 1).all()
    assert a.loc[days[20 + NOLIMIT_DAYS]] == 0
    assert (f.loc[f["code"] == "333330", "is_nolimit"] == 0).all()
    kept = universe(f, target="y_up")
    assert not ((kept["code"] == "444440") & (kept["date"] == days[20])).any()
    assert universe_mask(f, target="y_up").sum() == len(f) - NOLIMIT_DAYS
    # 하한가 예측에도 적용한다. 타깃을 주지 않으면(분석용) 적용 안 함
    assert universe_mask(f, target="y_down").sum() == len(f) - NOLIMIT_DAYS
    assert universe_mask(f).all()
    # 명시적으로 exclude 해도 두 번 빼지 않는다
    assert universe_mask(f, ("nolimit",), target="y_down").sum() == len(f) - NOLIMIT_DAYS
    # 표시 열이 없는 옛 특징 파일에서도 동작해야 한다
    assert universe_mask(f.drop(columns=["is_nolimit"]), target="y_up").all()


def test_intraday_out_of_range_day_is_not_a_limit_event():
    """종가가 우연히 계산상 하한가와 같아도 장중 저가가 가격제한 밖이면(정리매매) 하한가가 아니다.
    장중 범위 안에서 하한가로 마감한 날은 그대로 하한가."""
    from krxlimit.features import build_features
    from krxlimit.model import universe_mask
    days = pd.bdate_range("2024-01-02", periods=12)
    rows = []
    for k, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        if k == 8:   # 기준가 10,000, 하한가 7,000. 555550 은 장중 4,000 까지(제한 밖), 666660 은 범위 안
            rows.append(_row(ds, "555550", 7_000, -3_000, 9_000, 9_500, 4_000, 900, name="정리"))
            rows.append(_row(ds, "666660", 7_000, -3_000, 9_000, 9_500, 7_000, 900, name="보통"))
        elif k > 8:
            rows.append(_row(ds, "555550", 7_000, 0, 7_000, 7_050, 6_950, 300, name="정리"))
            rows.append(_row(ds, "666660", 7_000, 0, 7_000, 7_050, 6_950, 300, name="보통"))
        else:
            rows.append(_row(ds, "555550", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="정리"))
            rows.append(_row(ds, "666660", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="보통"))
    p = build_panel(pd.DataFrame(rows), save=False).set_index(["code", "date"])
    assert not p.loc[("555550", days[8]), "limit_valid"] and not p.loc[("555550", days[8]), "down_close"]
    assert p.loc[("666660", days[8]), "limit_valid"] and p.loc[("666660", days[8]), "down_close"]
    f = build_features(p.reset_index(), save=False, peer=False).set_index(["code", "date"])
    assert f.loc[("555550", days[7]), "y_down"] == 0 and f.loc[("666660", days[7]), "y_down"] == 1
    assert f.loc[("555550", days[8]), "is_nolimit"] == 1 and f.loc[("666660", days[8]), "is_nolimit"] == 0
    m = pd.Series(universe_mask(f.reset_index(), target="y_down"), index=f.index)
    assert not m.loc[("555550", days[8])] and m.loc[("666660", days[8])]


def test_liquidation_window_has_no_limit_events():
    """정리매매: 가격제한 없는 날부터 그 종목의 거래일로 7일 동안은 범위 안에서 계산상 하한가로 끝나도 하한가가 아니다.
    미래(상장폐지 여부)를 보지 않으므로 상장폐지된 종목과 계속 거래되는 종목의 판정이 같고, 기간이 지나면 다시 판정한다."""
    days = pd.bdate_range("2024-01-02", periods=20)
    rows = []
    for k, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        rows.append(_row(ds, "888880", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="보통"))
        for code in ("777770", "999990"):
            if code == "777770" and k > 11:      # 777770 은 정리매매 마지막 날(k=11) 뒤 상장폐지
                continue
            if k < 5:
                rows.append(_row(ds, code, 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name=code))
            elif k == 5:                         # 정리매매 첫날 -80%
                rows.append(_row(ds, code, 2_000, -8_000, 2_500, 2_500, 1_900, 900, name=code))
            elif k == 6:                         # 범위 안에서 계산상 하한가(2,000 → 1,400)로 마감
                rows.append(_row(ds, code, 1_400, -600, 1_900, 1_950, 1_400, 900, name=code))
            else:
                rows.append(_row(ds, code, 1_400, 0, 1_400, 1_420, 1_390, 300, name=code))
    p = build_panel(pd.DataFrame(rows), save=False).set_index(["code", "date"])
    assert not p.loc[("777770", days[5]), "limit_valid"]
    assert not p.loc[("777770", days[6]), "down_close"] and not p.loc["777770", "limit_valid"].iloc[5:].any()
    assert p.loc[("777770", days[4]), "limit_valid"]
    assert not p.loc[("999990", days[6]), "down_close"]      # 판정은 그날까지의 자료로만: 계속 거래되는 종목도 같다
    assert not p.loc["999990", "limit_valid"].iloc[5:12].any() and p.loc["999990", "limit_valid"].iloc[12:].all()


def test_liquidation_split_by_halt_counts_traded_days():
    """가처분 등으로 정리매매가 중간에 멈췄다가 재개되면 거래일로 세어 7일을 채운다(거래정지일은 세지 않음)."""
    from krxlimit.features import build_features
    days = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    traded_after = []
    for k, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        rows.append(_row(ds, "888880", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="보통"))
        if k < 5:
            rows.append(_row(ds, "121210", 10_000, 0, 10_000, 10_100, 9_900, 100 + k, name="정리"))
        elif k == 5:     # 정리매매 첫날 -80%
            rows.append(_row(ds, "121210", 2_000, -8_000, 2_500, 2_500, 1_900, 900, name="정리"))
        elif k in (6, 7) or 20 <= k:      # 3거래일 뒤 멈췄다가 k=20 부터 재개
            rows.append(_row(ds, "121210", 2_000, 0, 2_000, 2_010, 1_990, 300, name="정리"))
            traded_after.append(k)
        else:            # 거래정지
            rows.append(_row(ds, "121210", 2_000, 0, 0, 0, 0, 0, name="정리"))
    p = build_panel(pd.DataFrame(rows), save=False).set_index(["code", "date"])
    a = p.loc["121210"]
    window = [5, 6, 7, 20, 21, 22, 23]                       # 거래일로 7일
    assert a["nolimit_window"].iloc[window].all() and a["nolimit_window"].iloc[8:20].all()
    assert not a["nolimit_window"].iloc[24:].any() and a["limit_valid"].iloc[24:].all()
    f = build_features(p.reset_index(), save=False, peer=False).set_index(["code", "date"]).loc["121210"]
    assert (f["is_nolimit"].reindex(days[[6, 7, 20, 23]]) == 1).all()
    assert (f.loc[f.index >= days[24], "is_nolimit"] == 0).all()

