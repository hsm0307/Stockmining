"""목표 5 개선: 백테스트 기반 '챔피언/도전자' 탐색.

과최적화를 막기 위한 규칙
1) 개발 구간(2019~2023)에서만 후보를 비교하고 고른다.
2) 후보는 현재 챔피언과 '같은 날' 상위 10개 적중 수를 짝지어 비교(paired t)한다.
   평균 차이가 0보다 크고 t ≥ MIN_T 일 때만 채택 → 우연한 개선을 걸러낸다.
3) 최종 챔피언은 한 번도 보지 않은 보류 구간(2024~현재)에서 기존 모델(v1)과 다시 비교한다.
4) 보류 구간에서도 나빠지지 않을 때만 새 버전(v2, v3, …)으로 승격한다.

라운드
  features : 새 특징 그룹을 하나씩 추가 → 효과 있는 그룹을 모두 합쳐 다시 확인
  params   : LightGBM 하이퍼파라미터·음성표본 비율 한 가지씩 변경
  model    : 배깅(bag3), 순위학습(rank), lgbm+rank 앙상블(ens)
  window   : 최근 N년만 학습, 스팩 제외
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_GROUPS, V1_GROUPS
from .model import TARGETS, evaluate, log_experiment, select_features, train_final, walk_forward
from .viz import NEUTRAL, UP_COLOR, plt, save_fig, save_table

DEV = ("2019-01-01", "2024-01-01")
HOLD_START = "2024-01-01"
MIN_T = 1.0
NEW_GROUPS = [g for g in FEATURE_GROUPS if g not in V1_GROUPS]

BASE_CFG = {"kind": "lgbm", "params": {}, "neg_rate": 0.1, "groups": V1_GROUPS, "exclude": [], "train_years": None}

PARAM_CANDIDATES = {
    "leaves15": {"params": {"num_leaves": 15}},
    "leaves63": {"params": {"num_leaves": 63}},
    "minchild50": {"params": {"min_child_samples": 50}},
    "minchild800": {"params": {"min_child_samples": 800}},
    "colsample0.4": {"params": {"colsample_bytree": 0.4}},
    "lambda10": {"params": {"reg_lambda": 10.0}},
    "neg0.05": {"neg_rate": 0.05},
    "neg0.3": {"neg_rate": 0.3},
}
MODEL_CANDIDATES = {"bag3": {"kind": "bag3"}, "rank": {"kind": "rank", "params": {}}, "ens": {"kind": "ens"}}
WINDOW_CANDIDATES = {"recent3y": {"train_years": 3}, "no_spac": {"exclude": ["spac"]}}


def _exp_dir(target: str):
    d = config.OUTPUT_DIR / "experiments" / target
    d.mkdir(parents=True, exist_ok=True)
    return d


def _merge(cfg: dict, change: dict) -> dict:
    out = json.loads(json.dumps(cfg))
    for k, v in change.items():
        out[k] = {**out.get(k, {}), **v} if k == "params" and v else v
    return out


def run_cfg(feats_df: pd.DataFrame, target: str, name: str, cfg: dict, start: str, end: str | None,
            freq: str, stage: str) -> dict:
    features = select_features(only_groups=cfg["groups"])
    preds = walk_forward(feats_df, target, start, end, freq=freq, kind=cfg["kind"], params=cfg["params"] or None,
                         neg_rate=cfg["neg_rate"], features=features, exclude=tuple(cfg["exclude"]),
                         train_years=cfg["train_years"], verbose=False)
    ev = evaluate(preds, target)
    hits = ev["daily"][["양성"] + [c for c in ev["daily"].columns if c.startswith("hit@")]]
    hits.to_parquet(_exp_dir(target) / f"{stage}__{name}.parquet")
    log_experiment(f"{stage}:{name}", target, {**cfg, "freq": freq, "period": f"{start}~{end or '현재'}",
                                               "n_features": len(features)}, ev["summary"])
    return {"name": name, "stage": stage, "cfg": cfg, "summary": ev["summary"], "daily": hits}


def paired(a: pd.DataFrame, b: pd.DataFrame, k: int = 10) -> tuple[float, float]:
    """같은 날짜끼리 짝지어 P@k 차이(a-b)의 평균과 t 값."""
    j = a[[f"hit@{k}"]].join(b[[f"hit@{k}"]], lsuffix="_a", rsuffix="_b", how="inner").fillna(0)
    d = (j[f"hit@{k}_a"] - j[f"hit@{k}_b"]) / k
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), float(d.mean() / se) if se and se > 0 else 0.0


class Search:
    def __init__(self, feats_df, target, freq="YS", verbose=True, dev=DEV):
        self.f, self.target, self.freq, self.verbose, self.dev = feats_df, target, freq, verbose, dev
        self.rows: list[dict] = []
        self.champ: dict | None = None

    def _row(self, res, gain, t, accepted, note=""):
        s = res["summary"]
        self.rows.append({"단계": res["stage"], "후보": res["name"], "P@5": s["P@5"], "P@10": s["P@10"],
                          "P@20": s["P@20"], "AP": s["AP"], "AUC": s["AUC"], "ΔP@10(vs 챔피언)": gain,
                          "t": t, "채택": accepted, "비고": note, "설정": json.dumps(res["cfg"], ensure_ascii=False)})
        if self.verbose:
            print(f"  [{res['stage']}] {res['name']:<22} P@10 {s['P@10']:.4f}  AP {s['AP']:.4f}  "
                  f"Δ {gain:+.4f} (t={t:+.2f}) {'✔ 채택' if accepted else ''} {note}", flush=True)

    def run(self, name, cfg, stage):
        return run_cfg(self.f, self.target, name, cfg, self.dev[0], self.dev[1], self.freq, stage)

    def baseline(self, cfg=None):
        res = self.run("v1_base", cfg or BASE_CFG, "base")
        self.champ = res
        self._row(res, 0.0, 0.0, True, "출발점(v1 설정)")

    def challenge(self, name, change, stage, accept=True):
        res = self.run(name, _merge(self.champ["cfg"], change), stage)
        gain, t = paired(res["daily"], self.champ["daily"])
        ok = accept and gain > 0 and t >= MIN_T
        self._row(res, gain, t, ok)
        return res, gain, t, ok

    def round_candidates(self, stage, candidates: dict):
        """후보마다 챔피언과 비교 → 가장 좋은 채택 후보 하나로 챔피언 교체."""
        best = None
        for name, change in candidates.items():
            res, gain, t, ok = self.challenge(name, change, stage)
            if ok and (best is None or gain > best[1]):
                best = (res, gain)
        if best:
            self.champ = best[0]
            if self.verbose:
                print(f"  → {stage} 라운드 챔피언: {best[0]['name']}", flush=True)

    def round_features(self):
        good = []
        for g in NEW_GROUPS:  # 그룹별로는 '합쳐 볼 후보'만 고르고, 채택은 합친 결과로 판단
            _, gain, t, _ = self.challenge(f"+{g}", {"groups": self.champ["cfg"]["groups"] + [g]}, "features",
                                           accept=False)
            if gain > 0 and t >= 0.5:  # 합쳐 볼 후보는 조금 느슨하게
                good.append(g)
        if len(good) >= 2:
            res, gain, t, ok = self.challenge("+" + "+".join(good), {"groups": self.champ["cfg"]["groups"] + good},
                                              "features")
            if ok:
                self.champ = res
        elif len(good) == 1:
            res, gain, t, ok = self.challenge(f"+{good[0]}(확정)", {"groups": self.champ["cfg"]["groups"] + good},
                                              "features")
            if ok:
                self.champ = res
        if self.verbose:
            print(f"  → 특징 라운드 챔피언 그룹: {self.champ['cfg']['groups']}", flush=True)


def _next_version(target: str) -> str:
    cdir = config.ROOT / "configs"
    vs = [int(p.stem.rsplit("_v", 1)[1]) for p in cdir.glob(f"{target}_v*.json") if p.stem.rsplit("_v", 1)[1].isdigit()]
    return f"{target}_v{max(vs + [1]) + 1}"


def improve(feats_df: pd.DataFrame, target: str = "y_up", rounds=("features", "params", "model", "window"),
            freq: str = "YS", promote: bool = False, base_cfg: dict | None = None,
            dev: tuple[str, str] = DEV, hold_start: str = HOLD_START) -> dict:
    t0 = datetime.now()
    print(f"[개선 탐색] {TARGETS[target]} — 개발 구간 {dev[0]}~{dev[1]} (재학습 {freq}), "
          f"보류 구간 {hold_start}~현재", flush=True)
    s = Search(feats_df, target, freq, dev=dev)
    s.baseline(base_cfg)
    if "features" in rounds:
        s.round_features()
    if "params" in rounds:
        s.round_candidates("params", PARAM_CANDIDATES)
    if "model" in rounds:
        s.round_candidates("model", MODEL_CANDIDATES)
    if "window" in rounds:
        s.round_candidates("window", WINDOW_CANDIDATES)

    board = pd.DataFrame(s.rows)
    save_table(board.round(5), f"g5_improve_dev_{target}")
    champ_cfg = s.champ["cfg"]
    print(f"\n개발 구간 최종 챔피언: {s.champ['name']}  P@10 {s.champ['summary']['P@10']:.4f} "
          f"(출발점 {s.rows[0]['P@10']:.4f})", flush=True)

    # ---- 보류 구간 확인: 분기마다 재학습하며 v1 설정과 비교
    print("\n[보류 구간 확인] 2024년~현재, 분기 재학습", flush=True)
    base = run_cfg(feats_df, target, "v1_base", base_cfg or BASE_CFG, hold_start, None, "QS", "holdout")
    if champ_cfg == (base_cfg or BASE_CFG):
        new, gain, t = base, 0.0, 0.0
    else:
        new = run_cfg(feats_df, target, "champion", champ_cfg, hold_start, None, "QS", "holdout")
        gain, t = paired(new["daily"], base["daily"])
    hold = pd.DataFrame([{"모델": r["name"], **{k: r["summary"][k] for k in ["P@5", "P@10", "P@20", "AP", "AUC"]}}
                         for r in ([base, new] if new is not base else [base])])
    hold["ΔP@10(vs v1)"] = [0.0, gain][:len(hold)]
    hold["t"] = [0.0, t][:len(hold)]
    save_table(hold.round(5), f"g5_improve_holdout_{target}")
    print(hold.round(4).to_string(index=False), flush=True)
    better = new is not base and gain >= 0
    verdict = ("보류 구간에서도 v1보다 좋거나 같음 → 승격 가능" if better else
               "보류 구간에서 v1보다 나쁨 → 승격하지 않음 (개발 구간 과최적화 가능성)" if new is not base else
               "개선 후보 없음 → v1 유지")
    print(f"판정: {verdict}", flush=True)

    _plot(board, hold, target)
    out = {"dev_board": board, "holdout": hold, "champion_cfg": champ_cfg, "verdict": verdict, "better": better}
    if promote and better:
        name = _next_version(target)
        cdir = config.ROOT / "configs"
        cdir.mkdir(exist_ok=True)
        (cdir / f"{name}.json").write_text(json.dumps({"name": name, "target": target, **champ_cfg,
                                                       "dev": s.champ["summary"], "holdout_gain_P@10": gain,
                                                       "holdout_t": t, "created": str(t0)[:19]},
                                                      ensure_ascii=False, indent=2, default=float))
        print(f"\n승격: {name} 학습 중 (전체 기간)", flush=True)
        train_final(feats_df, target, kind=champ_cfg["kind"], params=champ_cfg["params"] or None,
                    neg_rate=champ_cfg["neg_rate"], exclude=tuple(champ_cfg["exclude"]), name=name,
                    features=select_features(only_groups=champ_cfg["groups"]), train_years=champ_cfg["train_years"])
        out["promoted"] = name
    print(f"소요 {datetime.now() - t0}", flush=True)
    return out


def _plot(board: pd.DataFrame, hold: pd.DataFrame, target: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    lbl = board["단계"] + ":" + board["후보"]
    colors = np.where(board["채택"], UP_COLOR, "#bbbbbb")
    ax[0].barh(lbl[::-1], board["P@10"][::-1], color=colors[::-1])
    ax[0].axvline(board["P@10"].iloc[0], color=NEUTRAL, ls="--", label="출발점(v1)")
    ax[0].set_title(f"개발 구간(2019~2023) 후보별 P@10 — 빨강: 채택 ({TARGETS[target]})")
    lo = board["P@10"].min()
    ax[0].set_xlim(lo * 0.9, board["P@10"].max() * 1.03)
    ax[0].legend()
    ax[1].bar(hold["모델"], hold["P@10"], color=[NEUTRAL, UP_COLOR][:len(hold)])
    for x, v in enumerate(hold["P@10"]):
        ax[1].text(x, v, f"{v:.3f}", ha="center", va="bottom")
    ax[1].set_title("보류 구간(2024~) P@10")
    save_fig(fig, f"g5_improve_{target}")
