"""명령행 진입점: python -m krxlimit <명령> [옵션]

  check          API 키·서비스 승인 확인
  collect        KOSPI/KOSDAQ 일별 시세 수집 (이어받기)
  build          패널 생성 + 상/하한가 판정 점검표
  features       특징/라벨 생성
  counts|pre|post|streak   목표 1/2/3/4 분석
  analyze        목표 1~4 전체
  backtest       목표 5 워크포워드 백테스트
  ablation       특징 그룹 제거 실험
  train          최종 모델 학습·저장
  predict        다음 거래일 예측 저장
  evaluate-live  저장된 예측 채점
  daily          수집→패널→채점→(필요시 재학습)→예측 한 번에
  report         outputs/REPORT.md 생성
  all            build→features→analyze→backtest→train→report
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from datetime import datetime

import pandas as pd

from . import config

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def _panel():
    from .panel import load_panel
    return load_panel()


def _features():
    from .features import load_features
    return load_features()


def cmd_build(a):
    from .panel import build_panel, validate_limits
    from .viz import save_table
    p = build_panel()
    v = validate_limits(p)
    save_table(v, "g0_limit_validation")
    print(v.to_string(index=False))


def cmd_features(a):
    from .features import build_features
    build_features(_panel(), peer=not a.no_peer)


def cmd_analyze(a, which=("counts", "post", "pre", "streak")):
    from .events import EventContext
    p = _panel()
    ctx = EventContext(p) if set(which) & {"post", "pre", "streak"} else None
    from . import analysis_counts, analysis_post, analysis_pre, analysis_streak
    steps = {
        "counts": ("[목표1] 건수 분석", lambda: analysis_counts.run(p)),
        "pre": ("[목표2] 직전 패턴 분석", lambda: analysis_pre.run(p, ctx)),
        "post": ("[목표3] 이후 수익률 분석", lambda: analysis_post.run(p, ctx)),
        "streak": ("[목표4] 연속 상/하한가 분석", lambda: analysis_streak.run(p, _features(), ctx, split=a.split)),
    }
    for name in which:
        title, fn = steps[name]
        print(title)
        fn()
        gc.collect()
    print(f"결과: {config.TABLE_DIR}, {config.FIG_DIR}")


def _names(a):
    """--target 목록 + --suffix → 모델 이름 목록 (예: y_up_v2)."""
    return [(t, f"{t}{a.suffix or ''}") for t in a.target.split(",")]


def _model_kwargs(a):
    return dict(kind=a.model, params=json.loads(a.params) if a.params else None, neg_rate=a.neg_rate,
                drop_groups=tuple(g for g in (a.drop_groups or "").split(",") if g),
                exclude=tuple(e for e in (a.exclude or "").split(",") if e))


def cmd_backtest(a):
    from .model import run_backtest
    f = _features()
    for t in a.target.split(","):
        run_backtest(f, t, tag=a.tag, start=a.start, freq=a.freq, **_model_kwargs(a))


def cmd_ablation(a):
    from .features import FEATURE_GROUPS
    from .model import run_backtest
    f = _features()
    kw = _model_kwargs(a)
    run_backtest(f, a.target, tag=f"abl_{a.target}_all", start=a.start, freq=a.freq, baselines=False, **kw)
    for g in FEATURE_GROUPS:
        run_backtest(f, a.target, tag=f"abl_{a.target}_-{g}", start=a.start, freq=a.freq, baselines=False,
                     **{**kw, "drop_groups": (g,)})
    ex = pd.read_csv(config.TABLE_DIR / "experiments.csv", encoding="utf-8-sig")
    print(ex[ex["tag"].str.startswith(f"abl_{a.target}")][["tag", "AUC", "AP", "P@10", "Lift@10"]].to_string())


def cmd_train(a):
    from .model import train_final
    f = _features()
    for t, name in _names(a):
        train_final(f, t, name=name, **_model_kwargs(a))


def cmd_predict(a):
    from .live import predict
    p = _panel()
    for _, name in _names(a):
        predict(p, name, date=a.date, top=a.top)


def cmd_evaluate_live(a):
    from .live import evaluate_live
    p = _panel()
    for _, name in _names(a):
        evaluate_live(p, name)


def cmd_daily(a):
    from .collect import _load_calendar, collect
    from .live import evaluate_live, predict
    from .model import load_model, train_final
    from .panel import build_panel
    opened = [d for d, v in _load_calendar().items() if v == "open"]
    start = (pd.Timestamp(max(opened)) - pd.Timedelta(days=7)).strftime("%Y%m%d") if opened else config.DEFAULT_START
    collect(start)
    p = build_panel()
    names = _names(a)
    for _, name in names:
        evaluate_live(p, name)
    need = []
    for t, name in names:
        try:
            created = pd.Timestamp(load_model(name)["meta"]["created"])
            if a.retrain_days > 0 and (pd.Timestamp(datetime.now()) - created).days >= a.retrain_days:
                need.append((t, name))
        except FileNotFoundError:
            need.append((t, name))
    if need:
        from .features import build_features
        print(f"재학습: {[n for _, n in need]} (주기 {a.retrain_days}일)")
        f = build_features(p, peer=True)
        for t, name in need:
            train_final(f, t, name=name, **_model_kwargs(a))
        del f
    for _, name in names:
        predict(p, name, top=a.top)
    print("\n장 시작 전에 예측을 커밋해 두세요:  git add predictions && git commit -m \"predict\" && git push")


def cmd_report(a):
    from .report import build_report
    build_report()


def cmd_all(a):
    cmd_build(a)
    cmd_features(a)
    cmd_analyze(a)
    cmd_backtest(a)
    cmd_train(a)
    cmd_report(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m krxlimit", description="KRX 상·하한가 데이터마이닝")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        s = sub.add_parser(name, **kw)
        s.set_defaults(fn=fn)
        return s

    def model_opts(s, target="y_up"):
        s.add_argument("--target", default=target, help="y_up | y_down | y_uptouch (쉼표로 여러 개)")
        s.add_argument("--model", default=None, help="lgbm | hgb | logit (기본: lgbm 설치 시 lgbm)")
        s.add_argument("--neg-rate", type=float, default=0.1, help="학습 시 음성 표본 추출 비율")
        s.add_argument("--params", default=None, help='모델 하이퍼파라미터 JSON, 예: \'{"num_leaves": 63}\'')
        s.add_argument("--drop-groups", default="", help="제외할 특징 그룹 (쉼표)")
        s.add_argument("--exclude", default="", help="제외할 종목 유형: spac,pref,reit,mgmt")

    s = add("check", lambda a: __import__("krxlimit.collect", fromlist=["check"]).check(a.date))
    s.add_argument("--date", default=None)
    s = add("collect", lambda a: __import__("krxlimit.collect", fromlist=["collect"]).collect(
        a.start, a.end, force=a.force, workers=a.workers))
    s.add_argument("--start", default=config.DEFAULT_START)
    s.add_argument("--workers", type=int, default=6, help="동시 요청 수")
    s.add_argument("--end", default=None)
    s.add_argument("--force", action="store_true")
    add("build", cmd_build)
    s = add("features", cmd_features)
    s.add_argument("--no-peer", action="store_true", help="동조종목(테마) 특징 생략")
    for name in ["counts", "pre", "post", "streak", "analyze"]:
        which = ("counts", "post", "pre", "streak") if name == "analyze" else (name,)
        s = add(name, lambda a, w=which: cmd_analyze(a, w))
        s.add_argument("--split", default=None, help="목표4 모델 검증 시작일 (기본: 70% 시점)")
    for name, fn in [("backtest", cmd_backtest), ("ablation", cmd_ablation)]:
        s = add(name, fn)
        model_opts(s)
        s.add_argument("--start", default="2019-01-01", help="백테스트 예측 시작일")
        s.add_argument("--freq", default="QS" if name == "backtest" else "YS", help="재학습 주기: MS(월), QS(분기), YS(연)")
        s.add_argument("--tag", default=None)
    suffix_help = "모델 이름 접미사. 예: --suffix _v2 → models/y_up_v2.pkl, predictions/y_up_v2/"
    s = add("train", cmd_train)
    model_opts(s)
    s.add_argument("--suffix", default="", help=suffix_help)
    s = add("predict", cmd_predict)
    s.add_argument("--target", default="y_up,y_down")
    s.add_argument("--suffix", default="", help=suffix_help)
    s.add_argument("--date", default=None, help="기준일 YYYYMMDD (기본: 데이터 마지막 날)")
    s.add_argument("--top", type=int, default=20)
    s = add("evaluate-live", cmd_evaluate_live)
    s.add_argument("--target", default="y_up,y_down")
    s.add_argument("--suffix", default="", help=suffix_help)
    s = add("daily", cmd_daily)
    model_opts(s, "y_up,y_down")
    s.add_argument("--suffix", default="", help=suffix_help)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--retrain-days", type=int, default=7,
                   help="모델이 이 일수보다 오래되면 재학습 (0 = 재학습 안 함, 고정 모델 비교용)")
    add("report", cmd_report)
    s = add("all", cmd_all)
    model_opts(s, "y_up,y_down")
    s.add_argument("--no-peer", action="store_true")
    s.add_argument("--split", default=None)
    s.add_argument("--start", default="2019-01-01")
    s.add_argument("--freq", default="QS")
    s.add_argument("--tag", default=None)
    s.add_argument("--suffix", default="")

    a = ap.parse_args(argv)
    config.ensure_dirs()
    a.fn(a)


if __name__ == "__main__":
    main()
