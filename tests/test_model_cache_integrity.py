"""같은 크기의 데이터 정정과 부분 분기 변경도 캐시 없는 재학습 결과와 같아야 한다."""
import numpy as np
import pandas as pd
import pytest

from krxlimit import model


@pytest.fixture
def data():
    days = pd.bdate_range("2018-10-01", "2019-03-29")
    n = len(days) * 4
    row = np.arange(n)
    return pd.DataFrame({
        "date": np.repeat(days, 4), "tidx": np.repeat(np.arange(len(days)), 4),
        "next_tidx": np.repeat(np.arange(len(days)), 4) + 1,
        "code": np.tile(["000001", "000002", "000003", "000004"], len(days)),
        "name": "synthetic", "market": "KOSPI", "y_up": (row % 7 == 0).astype(float),
        "x": np.sin(row / 13) + (row % 7 == 0) * 0.5,
        "next_ret": row % 7 / 100, "next_open_ret": row % 3 / 100,
        "up_close": 0.0, "up_streak": 0.0, "down_close": 0.0,
        "down_streak": 0.0, "up_touch": 0.0, "ret1": row % 5 / 100, "is_pref": 0,
    })


@pytest.fixture
def count_fits(monkeypatch):
    real_fit = model.fit
    calls = []

    def fit(*args, **kwargs):
        calls.append(True)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(model, "fit", fit)
    return calls


def run(data, cache=None, end="2019-03-01", **kwargs):
    return model.walk_forward(data, start="2019-01-01", end=end, kind="logit",
                              features=["x"], neg_rate=1, verbose=False, cache=cache, **kwargs)


def test_unchanged_input_reuses_cache(data, tmp_path, count_fits):
    first = run(data, tmp_path / "cache")
    n = len(count_fits)
    repeated = run(data, tmp_path / "cache")
    assert len(count_fits) == n
    pd.testing.assert_frame_equal(first, repeated)


@pytest.mark.parametrize("first_end,second_end", [
    ("2019-01-15", "2019-03-01"), ("2019-03-01", "2019-01-15"),
])
def test_changed_end_matches_uncached(data, tmp_path, first_end, second_end, count_fits):
    run(data, tmp_path / "cache", end=first_end)
    before = len(count_fits)
    cached = run(data, tmp_path / "cache", end=second_end)
    assert len(count_fits) > before
    expected = run(data, end=second_end)
    pd.testing.assert_frame_equal(cached, expected)
    assert cached["date"].max() < pd.Timestamp(second_end)


@pytest.mark.parametrize("column", [
    "x", "y_up", "next_ret", "next_open_ret", "next_tidx", "name", "code", "market", "is_pref",
])
def test_same_size_corrections_match_uncached(data, tmp_path, column, count_fits):
    original = run(data, tmp_path / "cache", exclude=("pref",))
    before = len(count_fits)
    changed = data.copy()
    test_row = changed.index[changed["date"] >= "2019-01-01"][0]
    if column == "x":
        changed.loc[:20, "x"] += 8  # 학습값만 바뀌어도 확률이 달라진다.
    elif column == "y_up":
        changed.loc[[0, 1], column] = changed.loc[[1, 0], column].to_numpy()
        assert changed[column].sum() == data[column].sum()
    elif column in ("name", "code", "market"):
        changed.loc[test_row, column] = "corrected"
    else:
        changed.loc[test_row, column] += 1
    assert len(changed) == len(data) and changed["date"].max() == data["date"].max()
    cached = run(changed, tmp_path / "cache", exclude=("pref",))
    assert len(count_fits) > before
    expected = run(changed, exclude=("pref",))
    pd.testing.assert_frame_equal(cached, expected)
    if column == "x":
        assert not np.allclose(original["prob"], cached["prob"])


def test_source_change_invalidates_cache(data, tmp_path, monkeypatch, count_fits):
    package = tmp_path / "source"
    package.mkdir()
    source = package / "model.py"
    source.write_text("# old calculation\n")
    monkeypatch.setattr(model, "__file__", str(source))
    run(data, tmp_path / "cache")
    before = len(count_fits)
    source.write_text("# revised calculation\n")
    cached = run(data, tmp_path / "cache")
    assert len(count_fits) > before
    pd.testing.assert_frame_equal(cached, run(data))


def test_default_change_invalidates_cache(data, tmp_path, monkeypatch, count_fits):
    run(data, tmp_path / "cache")
    before = len(count_fits)
    monkeypatch.setattr(model.config, "RANDOM_SEED", model.config.RANDOM_SEED + 1)
    cached = run(data, tmp_path / "cache")
    assert len(count_fits) > before
    pd.testing.assert_frame_equal(cached, run(data))


def test_evolve_full_period_cache_checks_the_same_input_signature(data, tmp_path, monkeypatch):
    """상위 전체기간 캐시가 분기 캐시 검증을 우회하거나 legacy 파일을 신뢰하면 안 된다."""
    from krxlimit import evolve

    monkeypatch.setattr(evolve, "STATE_DIR", tmp_path)
    monkeypatch.setattr(evolve, "DEV", ("2019-01-01", "2019-03-01"))
    monkeypatch.setattr(evolve, "select_features", lambda **kwargs: ["x"])
    cfg = {"groups": ["synthetic"], "kind": "logit", "params": {}, "neg_rate": 1,
           "exclude": [], "train_years": None}
    production = {"name": "base", "key": "base-key", "cfg": cfg}
    challenger = {"label": "candidate", "key": "candidate-key", "cfg": cfg}
    calls = []

    def run_cfg(feats, target, name, cfg, start, end, freq, stage, **kwargs):
        calls.append(name)
        preds = run(feats, end=None)
        ev = model.evaluate(preds, target)
        return {"preds": preds, "daily": ev["daily"], "summary": ev["summary"]}

    monkeypatch.setattr(evolve, "run_cfg", run_cfg)
    monkeypatch.setattr(evolve, "ten_year_tables", lambda *args: {
        "hold_gain": -0.1, "hold_t": -1.0,
        "paired": pd.DataFrame({"기간": ["10년 전체"], "비교": ["도전자 − base"],
                                 "ΔP@10": [-0.1], "t": [-1.0]}),
    })
    ev = evolve.Evolver(data, verbose=False)
    for item in (production, challenger):
        pd.DataFrame({"unverified_legacy": [1]}).to_parquet(ev._ten_cache("y_up", item["key"]))

    def validate():
        state = {"target": "y_up", "production": production.copy(), "dev_champion": challenger.copy(),
                 "validations": [], "accepted_since_validation": 1}
        ev.validate(state)

    validate()
    assert len(calls) == 2  # 서명 없는 과거 파일을 읽지 않고 둘 다 계산했다.
    validate()
    assert len(calls) == 2  # 그대로인 데이터는 전체기간 캐시를 재사용한다.
    ev.f = data.copy()
    ev.f.loc[:20, "x"] += 8
    validate()
    assert len(calls) == 4
    expected = run(ev.f, end=None)
    actual = pd.read_parquet(ev._ten_cache("y_up", production["key"]))
    pd.testing.assert_frame_equal(actual, expected)
