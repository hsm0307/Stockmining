"""파일 기록상 사후 예측과 검증 불가 기록이 실전 성적에 섞이지 않아야 한다."""
import json

import pandas as pd
import pytest

from krxlimit import config, live


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(config, "FIG_DIR", tmp_path / "figures")
    figures = []

    def save(fig, name):
        figures.append(name)
        live.plt.close(fig)

    monkeypatch.setattr(live, "save_fig", save)
    folder = config.PRED_DIR / "y_up"
    (folder / "full").mkdir(parents=True)
    # 전역 meta의 학습일은 파일별 검증에 사용하면 안 된다.
    (folder / "meta.json").write_text(json.dumps({"target": "y_up", "trained_through": "2026-01-01"}))
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2 + ["2020-01-03"] * 2),
                          "code": ["000001", "000002"] * 2, "up_close": [False, False, True, False]})
    pred = pd.DataFrame({"rank": [1, 2], "code": ["000001", "000002"], "name": ["a", "b"],
                         "prob": [0.9, 0.1], "기준일": "2020-01-02", "모델학습데이터~": "2019-12-31",
                         "생성시각": "2020-01-03 08:30:00"})
    return folder, panel, pred, figures


def write_prediction(folder, pred):
    pred.to_csv(folder / "full" / "20200102.csv.gz", index=False, encoding="utf-8-sig")


@pytest.mark.parametrize("created", [
    "2020-01-03 08:30:00", "2020-01-03T08:59:59+09:00", "2020-01-02T23:30:00Z",
])
def test_preopen_records_are_scored_using_file_metadata(environment, created):
    folder, panel, pred, _ = environment
    pred["생성시각"] = created
    write_prediction(folder, pred)
    result = live.evaluate_live(panel, verbose=False)
    assert len(result) == 1 and result.iloc[0]["hit@5"] == 1
    assert result.iloc[0]["기록검증"] == "timing_records_passed"


@pytest.mark.parametrize("column,value,reason", [
    ("생성시각", "2020-01-03T09:00:00+09:00", "created_at_or_after_next_0900"),
    ("생성시각", "2026-10-17 10:00:00", "created_at_or_after_next_0900"),
    ("생성시각", "2020-01-02T15:29:59+09:00", "created_before_reference_close"),
    ("모델학습데이터~", "2020-01-02", "training_not_before_reference"),
    ("모델학습데이터~", "2020-12-31", "training_not_before_reference"),
    ("기준일", "2020-01-01", "reference_date_mismatch"),
    ("생성시각", "not-a-time", "invalid_timestamp"),
    ("생성시각", None, "missing_or_inconsistent"),
])
def test_unverified_or_retrospective_files_are_excluded(environment, column, value, reason):
    folder, panel, pred, _ = environment
    pred[column] = value
    write_prediction(folder, pred)
    # 기존에 잘못 생성된 실전 결과도 빈 결과로 갱신되어야 한다.
    (folder / "scoreboard.csv").write_text("old,unverified\n1,1\n")
    result = live.evaluate_live(panel, verbose=False)
    assert result.empty
    assert pd.read_csv(folder / "scoreboard.csv").empty
    audit = pd.read_csv(folder / "live_prediction_audit.csv")
    assert reason in audit.iloc[0]["사유"]


def test_missing_and_mixed_per_row_stamps_are_excluded(environment):
    folder, panel, pred, _ = environment
    write_prediction(folder, pred.drop(columns="모델학습데이터~"))
    assert live.evaluate_live(panel, verbose=False).empty
    pred.loc[1, "생성시각"] = "2026-10-17 10:00:00"
    write_prediction(folder, pred)
    assert live.evaluate_live(panel, verbose=False).empty


def test_replay_never_overwrites_live_outputs(environment):
    folder, panel, pred, figures = environment
    write_prediction(folder, pred)
    live.evaluate_live(panel, verbose=False)
    original_score = (folder / "scoreboard.csv").read_bytes()
    original_hits = (folder / "actual_events_rank.csv").read_bytes()
    pred["생성시각"] = "2026-10-17 10:00:00"
    pred["모델학습데이터~"] = "2020-12-31"
    write_prediction(folder, pred)
    replay = live.evaluate_live(panel, verbose=False, include_retrospective=True)
    assert len(replay) == 1
    assert "training_not_before_reference" in replay.iloc[0]["기록검증"]
    assert (folder / "scoreboard.csv").read_bytes() == original_score
    assert (folder / "actual_events_rank.csv").read_bytes() == original_hits
    assert (folder / "replay_scoreboard.csv").exists()
    assert (folder / "replay_actual_events_rank.csv").exists()
    assert figures == ["live_y_up", "replay_y_up"]


def test_high_confidence_picks_are_tracked(environment):
    """목표 지표: 확률 15% 이상 고확신 추천의 적중·시가 매수 가능 적중을 날마다 기록한다."""
    folder, panel, pred, _ = environment
    panel = panel.assign(open=[100, 100, 120, 100], upper=[130, 130, 130, 130])
    pred = pd.concat([pred, pred.iloc[[1]].assign(rank=3, code="000003", name="c", prob=0.16)], ignore_index=True)
    panel = pd.concat([panel, pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                                            "code": ["000003"] * 2, "up_close": [False, False],
                                            "open": [100, 100], "upper": [130, 130]})], ignore_index=True)
    write_prediction(folder, pred)
    row = live.evaluate_live(panel, verbose=False).iloc[0]
    assert (row["고확신수"], row["고확신적중"], row["고확신매수가능적중"]) == (2, 1, 1)
    assert (row["고확신20수"], row["고확신20적중"]) == (1, 1)
    assert (row["1위적중"], row["2위내적중"]) == (1, 1)
    picks = pd.read_csv(folder / "highconf_picks.csv", dtype={"code": str})
    assert picks["code"].tolist() == ["000001", "000003"] and picks["적중"].tolist() == [True, False]
