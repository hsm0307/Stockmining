"""합성 데이터로 수집→패널→특징→분석→백테스트→학습→예측→채점→보고서 전 과정을 실행해 본다."""
import numpy as np
import pandas as pd
import pytest

from krxlimit import config
from synthetic import make_raw


class FakeClient:
    """KRX Open API 대신 합성 데이터를 날짜·시장별로 돌려주는 가짜 클라이언트."""

    def __init__(self, raw):
        self.by_key = {k: g.drop(columns="MARKET") for k, g in raw.groupby(["BAS_DD", "MARKET"])}
        self.calls = 0

    def daily_trades(self, market, ds):
        self.calls += 1
        return self.by_key.get((ds, market), pd.DataFrame()).reset_index(drop=True)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("krx")
    mp = pytest.MonkeyPatch()
    for name, sub in [("DATA_DIR", "data"), ("RAW_DIR", "data/raw"), ("PROCESSED_DIR", "data/processed"),
                      ("OUTPUT_DIR", "outputs"), ("FIG_DIR", "outputs/figures"), ("TABLE_DIR", "outputs/tables"),
                      ("MODEL_DIR", "models"), ("PRED_DIR", "predictions")]:
        mp.setattr(config, name, root / sub)
    mp.setattr(config, "PANEL_PATH", root / "data/processed/panel.parquet")
    mp.setattr(config, "FEATURES_PATH", root / "data/processed/features.parquet")
    raw = make_raw(n_stocks=70, start="2020-01-01", end="2023-06-30", seed=1)
    yield root, raw
    mp.undo()


def test_end_to_end(env):
    root, raw = env
    from krxlimit import analysis_counts, analysis_post, analysis_pre, analysis_streak
    from krxlimit.collect import collect
    from krxlimit.events import EventContext
    from krxlimit.features import build_features
    from krxlimit.live import evaluate_live, predict
    from krxlimit.model import run_backtest, train_final
    from krxlimit.panel import build_panel, validate_limits
    from krxlimit.report import build_report

    client = FakeClient(raw)
    collect("20200101", "20230630", client=client, verbose=False)
    n_first = client.calls
    collect("20200101", "20230630", client=client, verbose=False)  # 이어받기: 다시 호출하지 않아야 함
    assert client.calls == n_first

    p = build_panel()
    v = validate_limits(p).set_index("항목")["건수"]
    assert v["상한가: 등락률 기준만 해당(누락 의심)"] == 0
    assert v["하한가: 등락률 기준만 해당(누락 의심)"] == 0
    assert p["up_close"].sum() > 100 and p["is_first_day"].sum() == 3

    f = build_features(p)
    ctx = EventContext(p)
    analysis_counts.run(p)
    analysis_pre.run(p, ctx)
    analysis_post.run(p, ctx)
    analysis_streak.run(p, f, ctx)

    res = run_backtest(f, "y_up", start="2022-01-01", freq="QS")
    s = res["model"]["summary"]
    assert 0 <= s["P@10"] <= 1 and s["n"] > 0
    assert len(res["preds"]) > 0 and res["preds"]["date"].min() >= pd.Timestamp("2022-01-01")

    train_final(f[f["date"] <= "2023-05-31"], "y_up")
    train_final(f[f["date"] <= "2023-05-31"], "y_down", kind="logit", name="y_down_v2")
    days = np.sort(p["date"].unique())
    for d in days[-10:-1]:
        out = predict(p, "y_up", date=str(pd.Timestamp(d).date()), verbose=False)
        assert out["prob"].between(0, 1).all()
        predict(p, "y_down_v2", date=str(pd.Timestamp(d).date()), verbose=False)
    sb = evaluate_live(p, "y_up", verbose=False)
    assert len(sb) == 9
    assert len(evaluate_live(p, "y_down_v2", verbose=False)) == 9

    # 실시간 예측(최근 구간만 계산)과 전체 계산의 특징값이 같아야 함 (train/serve 불일치 방지)
    from krxlimit.features import ALL_FEATURES, LIVE_LOOKBACK
    last = days[-1]
    sub = p[p["date"] >= days[-LIVE_LOOKBACK]]
    a = f[f["date"] == last].set_index("code")[ALL_FEATURES].sort_index()
    b = build_features(sub, save=False, keep_from=str(pd.Timestamp(last).date())).set_index("code")[ALL_FEATURES].sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=1e-5, atol=1e-6)

    text = build_report()
    assert "목표 5" in text
    assert (root / "outputs/figures/g3_paths.png").exists()


def test_improve_search(env, monkeypatch):
    """개선 탐색이 모든 모델 종류(lgbm/bag/rank/ens)로 끝까지 돌고 새 버전을 저장하는지."""
    root, raw = env
    from krxlimit import improve as imp
    from krxlimit.features import load_features
    from krxlimit.live import predict
    from krxlimit.panel import load_panel

    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(imp, "MIN_T", -99.0)  # 합성 데이터에서도 채택이 일어나도록
    monkeypatch.setattr(imp, "NEW_GROUPS", ["rel_strength", "pref_link"])
    monkeypatch.setattr(imp, "PARAM_CANDIDATES", {"leaves15": {"params": {"num_leaves": 15}}})
    monkeypatch.setattr(imp, "WINDOW_CANDIDATES", {"recent1y": {"train_years": 1}})
    f = load_features()
    out = imp.improve(f, "y_up", freq="QS", promote=True, dev=("2022-01-01", "2022-10-01"), hold_start="2022-10-01")
    assert len(out["dev_board"]) >= 6
    if out["better"]:
        name = out["promoted"]
        assert (root / "configs" / f"{name}.json").exists()
        p = load_panel()
        res = predict(p, name, date=str(p["date"].max().date()), verbose=False)
        assert res["prob"].between(0, 1).all()
