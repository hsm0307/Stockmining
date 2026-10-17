"""실전 예측(발표 전 2달) 운영: 매일 예측 저장 → 다음 날 채점 → 누적 성적표.

권장 루틴 (평일 아침 8시 이후, KRX Open API 에 전일 데이터가 올라온 뒤 장 시작 전)
  python -m krxlimit daily
  git add predictions && git commit -m "predict YYYYMMDD" && git push
예측 파일을 장 시작 전에 커밋해 두면 커밋 시각이 '사전에 예측했다'는 증거가 된다.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config
from .features import LIVE_LOOKBACK, build_features
from .model import KS, TARGETS, load_model, predict_proba, universe
from .viz import NEUTRAL, UP_COLOR, date_axis, plt, save_fig

LABEL_COL = {"y_up": "up_close", "y_down": "down_close", "y_uptouch": "up_touch"}
SHOW_COLS = ["rank", "code", "name", "market", "close", "prob", "ret1", "up_close", "up_jeom", "up_streak",
             "down_close", "down_streak", "vr1_20", "turnover", "peer_n_up", "mkt_n_up"]


def _pred_dir(name):
    d = config.PRED_DIR / name
    (d / "full").mkdir(parents=True, exist_ok=True)
    return d


def predict(panel: pd.DataFrame, name: str = "y_up", date: str | None = None, top: int = 20,
            verbose: bool = True) -> pd.DataFrame:
    """name: 모델 이름 (= 타깃 + 접미사, 예: y_up, y_up_v2). 예측은 predictions/{name}/ 에 저장."""
    bundle = load_model(name)
    model, meta = bundle["model"], bundle["meta"]
    target = meta["target"]
    last = pd.Timestamp(date) if date else panel["date"].max()
    if str(last.date()) <= meta["trained_through"]:
        print(f"[주의] 기준일 {last:%Y-%m-%d} 이 모델 학습 데이터 기간(~{meta['trained_through']}) 안에 있습니다. "
              "사후 예측이므로 성능 평가에 쓰면 안 됩니다.")
    days = np.sort(panel.loc[panel["date"] <= last, "date"].unique())
    if len(days) == 0 or days[-1] != last:
        raise ValueError(f"{last:%Y-%m-%d} 데이터가 패널에 없습니다. collect/build 를 먼저 실행하세요.")
    sub = panel[(panel["date"] >= days[max(0, len(days) - LIVE_LOOKBACK)]) & (panel["date"] <= last)]
    f = build_features(sub, save=False, keep_from=str(last.date()))
    f = universe(f[f["date"] == last], meta.get("exclude", ()))
    f["prob"] = predict_proba(model, f[meta["features"]], meta["neg_rate"])
    f = f.sort_values("prob", ascending=False).reset_index(drop=True)
    f["rank"] = np.arange(1, len(f) + 1)
    stamp = {"기준일": f"{last:%Y-%m-%d}", "예측대상": TARGETS[target] + " (다음 거래일)",
             "모델학습데이터~": meta["trained_through"],
             "생성시각": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")}

    d = _pred_dir(name)
    (d / "meta.json").write_text(json.dumps({"model": name, **meta}, ensure_ascii=False, indent=2))
    ds = f"{last:%Y%m%d}"
    top_df = f[SHOW_COLS].head(top).assign(**stamp)
    top_df.to_csv(d / f"{ds}.csv", index=False, encoding="utf-8-sig")
    f[["rank", "code", "name", "prob"]].assign(**stamp).to_csv(d / "full" / f"{ds}.csv.gz", index=False,
                                                               encoding="utf-8-sig")
    if verbose:
        streak = "down_streak" if target == "y_down" else "up_streak"
        show = f[["rank", "code", "name", "market", "close", "prob", "ret1", streak]].head(top).copy()
        show["prob"] = (show["prob"].astype(float) * 100).round(1).astype(str) + "%"
        show["ret1"] = (show["ret1"].astype(float) * 100).round(1)
        print(f"\n[{name}: {stamp['예측대상']}] 기준일 {stamp['기준일']} 상위 {top}개 "
              f"(모델 학습 ~{meta['trained_through']})")
        print(show.to_string(index=False))
        print(f"→ 저장: {d / (ds + '.csv')}")
    return f


def _prediction_timing_errors(pred: pd.DataFrame, reference: pd.Timestamp, following: pd.Timestamp) -> list[str]:
    """파일 기록만 검증한다. 기록의 진위나 실제 거래일별 개장시각을 증명하지 않는다."""
    stamps = {}
    errors = []
    for column in ("기준일", "모델학습데이터~", "생성시각"):
        if column not in pred or pred[column].isna().any() or pred[column].nunique() != 1:
            errors.append(f"missing_or_inconsistent:{column}")
        else:
            try:
                value = pd.Timestamp(pred[column].iloc[0])
                if pd.isna(value):
                    raise ValueError("missing timestamp")
                stamps[column] = value
            except (TypeError, ValueError):
                errors.append(f"invalid_timestamp:{column}")
    if "기준일" in stamps and stamps["기준일"].date() != reference.date():
        errors.append("reference_date_mismatch")
    if "모델학습데이터~" in stamps and stamps["모델학습데이터~"].date() >= reference.date():
        errors.append("training_not_before_reference")
    if "생성시각" in stamps:
        created = stamps["생성시각"]
        # 기존 naive 생성시각은 서울 시간으로 해석한다. 새 기록은 +09:00을 명시한다.
        created = created.tz_localize("Asia/Seoul") if created.tzinfo is None else created.tz_convert("Asia/Seoul")
        earliest = reference.tz_localize("Asia/Seoul") + pd.Timedelta(hours=15, minutes=30)
        deadline = following.tz_localize("Asia/Seoul") + pd.Timedelta(hours=9)
        if created < earliest:
            errors.append("created_before_reference_close")
        if created >= deadline:
            errors.append("created_at_or_after_next_0900")
    return errors


def evaluate_live(panel: pd.DataFrame, name: str = "y_up", verbose: bool = True,
                  include_retrospective: bool = False) -> pd.DataFrame:
    """파일별 시각 검증을 통과한 예측만 실전 집계한다. naive 시각은 Asia/Seoul로 해석한다.

    다음 관측 시장일 09:00 이전 생성만 허용하는 보수적 기준이며, 실제 개장시간이나 파일 진위의 증빙은 아니다.
    include_retrospective=True는 과거 재생 전체를 replay_*에 따로 저장하여 실전 결과와 섞지 않는다.
    """
    d = _pred_dir(name)
    meta_path = d / "meta.json"
    target = json.loads(meta_path.read_text())["target"] if meta_path.exists() else name
    days = np.sort(panel["date"].unique())
    label = LABEL_COL[target]
    rows, hit_rows, audit = [], [], []
    prefix = "replay" if include_retrospective else "live"
    scoreboard_name = "replay_scoreboard.csv" if include_retrospective else "scoreboard.csv"
    hits_name = "replay_actual_events_rank.csv" if include_retrospective else "actual_events_rank.csv"
    mode_label = "과거 재생 성적 (실전 증빙 아님)" if include_retrospective else "파일 기록 기준 실전 성적"
    for fp in sorted((d / "full").glob("*.csv.gz")):
        try:
            ds = pd.Timestamp(datetime.strptime(fp.name[:8], "%Y%m%d"))
        except ValueError:
            audit.append({"파일": fp.name, "실전포함": False, "사유": "invalid_filename_date"})
            continue
        if not np.any(days == np.datetime64(ds)):
            audit.append({"파일": fp.name, "기준일": ds.date(), "실전포함": False,
                          "사유": "reference_date_not_observed"})
            continue
        pos = np.searchsorted(days, ds, side="right")
        if pos >= len(days):
            audit.append({"파일": fp.name, "기준일": ds.date(), "실전포함": False, "사유": "outcome_not_observed"})
            continue  # 아직 결과 없음
        nxt = pd.Timestamp(days[pos])
        pred = pd.read_csv(fp, dtype={"code": str}, encoding="utf-8-sig")
        errors = _prediction_timing_errors(pred, ds, nxt)
        audit.append({"파일": fp.name, "기준일": ds.date(), "결과일": nxt.date(), "실전포함": not errors,
                      "사유": ";".join(errors) if errors else "timing_records_passed"})
        if errors and not include_retrospective:
            continue
        # 다음 날 거래정지·무효·상장폐지 종목도 '실패'로 채점 (백테스트 라벨과 같은 기준)
        act = panel.loc[panel["date"] == nxt, ["code", label]]
        act = act.assign(code=act["code"].astype(str))
        m = pred.merge(act, on="code", how="left")
        m[label] = m[label].fillna(False).astype(bool)
        if m.empty:
            continue
        y = m[label].astype(int).to_numpy()
        row = {"기준일": ds.date(), "결과일": pd.Timestamp(nxt).date(), "종목수": len(m), "실제발생": int(y.sum()),
               "AUC": roc_auc_score(y, m["prob"]) if 0 < y.sum() < len(y) else np.nan,
               "AP": average_precision_score(y, m["prob"]) if y.sum() > 0 else np.nan,
               "기록검증": "timing_records_passed" if not errors else ";".join(errors)}
        for k in KS:
            row[f"hit@{k}"] = int(m.loc[m["rank"] <= k, label].sum())
            row[f"P@{k}"] = row[f"hit@{k}"] / k
        row["실제발생 종목의 최고순위"] = int(m.loc[m[label] == 1, "rank"].min()) if y.sum() else np.nan
        rows.append(row)
        hits = m[m[label] == 1].sort_values("rank")
        for _, h in hits.iterrows():
            hit_rows.append({"기준일": ds.date(), "결과일": pd.Timestamp(nxt).date(), "code": h["code"],
                             "name": h["name"], "예측순위": int(h["rank"]), "예측확률": round(float(h["prob"]), 4)})
    columns = ["기준일", "결과일", "종목수", "실제발생", "AUC", "AP", "기록검증"]
    columns += [col for k in KS for col in (f"hit@{k}", f"P@{k}")] + ["실제발생 종목의 최고순위"]
    sb = pd.DataFrame(rows, columns=columns)
    sb.to_csv(d / scoreboard_name, index=False, encoding="utf-8-sig")
    pd.DataFrame(hit_rows, columns=["기준일", "결과일", "code", "name", "예측순위", "예측확률"]).to_csv(
        d / hits_name, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit, columns=["파일", "기준일", "결과일", "실전포함", "사유"]).to_csv(
        d / f"{prefix}_prediction_audit.csv", index=False, encoding="utf-8-sig")
    if sb.empty:
        (config.FIG_DIR / f"{prefix}_{name}.png").unlink(missing_ok=True)
        if verbose:
            print(f"[{name}] {mode_label}에 포함할 예측이 없습니다. {prefix}_prediction_audit.csv를 확인하세요.")
        return sb

    n = len(sb)
    total = {f"누적P@{k}": sb[f"hit@{k}"].sum() / (k * n) for k in KS}
    base = sb["실제발생"].sum() / sb["종목수"].sum()
    if verbose:
        print(f"\n[{name}: {TARGETS[target]} {mode_label}] {n}거래일, 실제 발생 {int(sb['실제발생'].sum())}건, 기저율 {base:.3%}")
        print("  " + ", ".join(f"{k} {v:.1%} (Lift {v / base:.1f}배)" for k, v in total.items() if base > 0))
        print(sb.tail(10).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    x = pd.to_datetime(sb["결과일"])
    for k, c in zip(KS, ["#d62728", "#ff7f0e", "#2ca02c"]):
        ax[0].plot(x, sb[f"hit@{k}"].cumsum() / (k * np.arange(1, n + 1)), marker="o", ms=3, color=c,
                   label=f"누적 P@{k}")
    ax[0].axhline(base, color=NEUTRAL, ls="--", label="무작위(기저율)")
    ax[0].set_title(f"{mode_label} - {name} ({TARGETS[target]})")
    ax[0].legend()
    ax[1].bar(x, sb["실제발생"], color="#cccccc", label="실제 발생 수")
    ax[1].bar(x, sb["hit@20"], color=UP_COLOR, label="상위20 적중")
    ax[1].set_title("일별 실제 발생 vs 상위 20 적중")
    ax[1].legend()
    date_axis(ax[0])
    date_axis(ax[1])
    save_fig(fig, f"{prefix}_{name}")
    return sb

