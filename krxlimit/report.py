"""분석 결과(표·그림)를 outputs/REPORT.md 하나로 모은다. 발표 자료 초안용."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .viz import read_table


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.1f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def md_table(df: pd.DataFrame | None, head: int | None = None) -> str:
    if df is None or df.empty:
        return "_(결과 없음 — 해당 분석을 먼저 실행하세요)_\n"
    d = df.head(head) if head else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(_fmt(v).replace("|", "/") for v in r.to_numpy()) + " |")
    return "\n".join(lines) + "\n"


def fig(name: str, caption: str = "") -> str:
    path = config.FIG_DIR / f"{name}.png"
    if not path.exists():
        return ""
    return f"\n![{caption or name}](figures/{name}.png)\n"


def build_report() -> str:
    T = read_table
    out = [f"# 한국 주식 상한가·하한가 데이터마이닝 결과\n\n생성: {datetime.now():%Y-%m-%d %H:%M}  \n"
           "데이터: KRX Open API 유가증권/코스닥 일별매매정보\n"]

    out.append("## 0. 데이터와 상한가 판정 점검\n")
    out.append(md_table(T("g1_summary")))
    out.append("\n수익률은 역산 기준가격을 연결한 가격지수 대용치이며 기업행위 현금흐름과 체결을 검증한 총수익률이 아닙니다. "
               "관측 누락·만기 미거래를 결측으로 두며, 관측 소실을 상장폐지로 확정하지 않습니다. "
               "관측 가능한 미래 경로를 선별하므로 전체 사건의 투자성과로 일반화할 수 없습니다.\n")
    out.append("\n규정 계산(기준가·호가단위)과 등락률 기준 판정의 일치도. '누락 의심'이 크면 `KRX_LIMIT_RULE=loose` 로 비교.\n\n")
    out.append(md_table(T("g0_limit_validation")))

    out.append("\n## 1. 상한가·하한가 건수 (목표 1)\n")
    out.append(md_table(T("g1_yearly")))
    out.append(fig("g1_yearly") + fig("g1_monthly") + fig("g1_by_group"))
    out.append("\n### 상한가 다발 종목 (상위 10)\n" + md_table(T("g1_top_stocks_up_close"), 10))
    out.append("\n### 하한가 다발 종목 (상위 10)\n" + md_table(T("g1_top_stocks_down_close"), 10))
    out.append("\n### 하한가가 가장 많았던 날\n" + md_table(T("g1_top_days_n_down"), 10))

    out.append("\n## 2. 상·하한가 직전 20/60일 특이사항 (목표 2)\n")
    out.append(fig("g2_event_study", "사건연구"))
    for side, lbl in [("up", "첫 상한가"), ("down", "첫 하한가")]:
        c = T(f"g2_compare_{side}")
        if c is not None:
            c = c.reindex(c["SMD"].abs().sort_values(ascending=False).index)
        out.append(f"\n### {lbl} vs 대조군 (|SMD| 큰 순, 상위 12)\n" + md_table(c, 12))
    out.append(fig("g2_smd"))
    out.append("\n### 매집 의심 점수\n" + md_table(T("g2_accum_flags_up")) + "\n" + md_table(T("g2_accum_score_dist_up"))
               + "\n" + md_table(T("g2_accum_post_up")) + fig("g2_accum_score"))
    out.append("\n### 직전 패턴 군집 (K-means)\n" + md_table(T("g2_clusters_up")) + fig("g2_clusters"))

    out.append("\n## 3. 상따·하따 이후 수익률 (목표 3)\n")
    out.append("\n평가일은 사건일 D에서 세는 D+h입니다. 익일 시가 진입의 1일 결과는 D+1 시가→D+1 종가입니다. "
               "같은 시가 기준 시장 벤치마크가 없어 그 진입 방식의 시장초과수익은 산출하지 않습니다.\n")
    observation = T("g3_returns_full")
    if observation is not None and "관측기간충족수" in observation:
        cols = ["방향", "이벤트", "진입", "보유기간", "사건수", "관측기간충족수", "n", "수익률미산출수", "관측소실건수"]
        out.append("\n### 표본 유지와 관측 누락\n" + md_table(observation.loc[
            observation["이벤트"].isin(["상한가 마감", "하한가 마감"]) & observation["진입"].eq("당일(제한가 매수)"), cols]))
    for metric in ["평균", "중앙값", "승률", "시장초과평균"]:
        out.append(f"\n### {metric}\n" + md_table(T(f"g3_returns_{metric}")))
    out.append(fig("g3_returns_by_horizon") + fig("g3_paths") + fig("g3_box"))
    bd = T("g3_breakdown")
    if bd is not None:
        sub = bd[(bd["분해기준"] == "연도") & (bd["보유기간"] == "1m")]
        out.append("\n### 연도별 1달 수익률 (상한가/하한가 마감, 당일 진입)\n" + md_table(sub))

    out.append("\n## 4. 연속 상·하한가 (목표 4)\n")
    out.append(md_table(T("g4_continuation_up")) + "\n" + md_table(T("g4_continuation_down")))
    out.append(fig("g4_streaks") + "\n### 상태 전이 확률\n" + md_table(T("g4_transition")) + fig("g4_transition"))
    out.append("\n### 연상 예측 모델 성능 (시간 분할 검증)\n" + md_table(T("g4_model_metrics_up")))
    out.append("\n### 연하 예측 모델 성능\n" + md_table(T("g4_model_metrics_down")))
    out.append("\n### 연상 성공 vs 실패 (첫 상한가 당일 특징, |SMD| 상위 12)\n"
               + md_table(T("g4_first_day_compare_up"), 12))
    out.append("\n### 연속 상한가: 최고점 대비 가격 변화 (연속 일수별 중앙값)\n" + md_table(T("g4_peak_up")))
    out.append("\n### 연속 하한가: 최저점 대비 가격 변화\n" + md_table(T("g4_peak_down")) + fig("g4_peak"))
    out.append(fig("g4_model_up") + fig("g4_tree_up"))
    rules = config.TABLE_DIR / "g4_tree_rules_up.txt"
    if rules.exists():
        out.append("\n```\n" + "\n".join(rules.read_text(encoding="utf-8").splitlines()[:40]) + "\n```\n")

    out.append("\n## 5. 익일 상한가 예측 모델 (목표 5)\n")
    ex = T("experiments")
    if ex is not None:
        keep = [c for c in ["시각", "tag", "target", "model", "freq", "AUC", "AP", "기저율", "P@5", "P@10", "P@20",
                            "Lift@10", "수익_시가진입@10"] if c in ex.columns]
        out.append("### 실험 기록 (experiments.csv)\n" + md_table(ex[keep].tail(20)))
    for target in ["y_up", "y_down"]:
        for p in sorted(config.FIG_DIR.glob(f"g5_backtest_{target}*.png"))[:3]:
            out.append(fig(p.stem))
        out.append(fig(f"g5_importance_{target}"))
    for sb in sorted(config.PRED_DIR.glob("*/scoreboard.csv")):
        name = sb.parent.name
        s = pd.read_csv(sb, encoding="utf-8-sig")
        out.append(f"\n### 실전 예측 성적표 ({name})\n" + md_table(s.tail(15)) + fig(f"live_{name}"))

    rates = T("g6_volume_signal_rates")
    signal_returns = T("g6_signal_returns")
    if rates is not None:
        out.append("\n## 6. 사전 20일 거래량 3배 신호\n"
                   "D−20~D−1 각일의 거래량을 그 직전 40시장일 평균과 비교합니다. 사건일은 제외하며, "
                   "이력 불완전은 신호 없음과 구분합니다. 아래 비율은 예측 적중률이 아닙니다.\n")
        out.append(md_table(rates[rates["market"].eq("ALL")]))
        if signal_returns is not None:
            out.append("\n### 신호별 D종가→D+1·21·126 가격지수 변화\n" + md_table(signal_returns.loc[
                signal_returns["market"].eq("ALL") & signal_returns["entry"].eq("close")]))

    text = "\n".join(out)
    path = config.OUTPUT_DIR / "REPORT.md"
    path.write_text(text, encoding="utf-8")
    print(f"보고서: {path}")
    return text

