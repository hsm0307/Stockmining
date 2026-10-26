"""목표 5 개선: 10년 백테스트 기반 '챔피언/도전자' 탐색 (상한가·하한가 공통).

과최적화를 막기 위한 규칙
1) 개발 구간(2017~2023, 7년)에서만 후보를 비교하고 고른다. (2016년은 첫 학습용)
2) 후보는 현재 챔피언과 '같은 날' 상위 k개 적중 수를 짝지어 비교(paired t)한다.
   평균 차이가 0보다 크고 t ≥ MIN_T 일 때만 채택 → 우연한 개선을 걸러낸다.
   k: 상한가 10, 하한가 20 (하한가는 하루 0.4건이라 10개로는 적중이 너무 드묾)
3) 최종 챔피언과 v1 을 2017~현재 전체(약 10년)에 대해 분기마다 재학습하는 워크포워드로 다시 돌리고,
   한 번도 보지 않은 보류 구간(2024~현재)에서 나빠지지 않을 때만 새 버전(v2, v3, …)으로 승격한다.

라운드
  features : 새 특징 그룹을 하나씩 추가 → 효과 있는 그룹을 모두 합쳐 다시 확인
  params   : LightGBM 하이퍼파라미터·음성표본 비율·헷갈리는 음성(hard negative) 유지
  model    : 배깅(bag3), 순위학습(rank), lgbm+rank 앙상블(ens)
  window   : 최근 N년만 학습, 스팩 제외
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_GROUPS, V1_GROUPS
from .model import TARGETS, evaluate, log_experiment, rule_score, select_features, train_final, walk_forward
from .viz import NEUTRAL, UP_COLOR, plt, save_fig, save_table

DEV = ("2017-01-01", "2024-01-01")
HOLD_START = "2024-01-01"
MIN_T = 1.0
PRIMARY_K = {"y_up": 10, "y_down": 20, "y_uptouch": 10}
NEW_GROUPS = [g for g in FEATURE_GROUPS if g not in V1_GROUPS]

BASE_CFG = {"kind": "lgbm", "params": {}, "neg_rate": 0.1, "groups": V1_GROUPS, "exclude": [], "train_years": None,
            "hard_neg": False}

PARAM_CANDIDATES = {
    "hardneg": {"hard_neg": True},
    "hardneg+neg0.03": {"hard_neg": True, "neg_rate": 0.03},
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
WINDOW_CANDIDATES = {"recent3y": {"train_years": 3}, "recent5y": {"train_years": 5}, "no_spac": {"exclude": ["spac"]}}


def _exp_dir(target: str):
    d = config.OUTPUT_DIR / "experiments" / target
    d.mkdir(parents=True, exist_ok=True)
    return d


def _merge(cfg: dict, change: dict) -> dict:
    """설정에 변경을 덮어쓴다. 기본값과 같은 하이퍼파라미터는 지워서, 같은 뜻의 설정이 다른 키로 중복 평가되지 않게 한다."""
    from .model import LGBM_PARAMS
    out = json.loads(json.dumps(cfg))
    for k, v in change.items():
        out[k] = {**(out.get(k) or {}), **v} if k == "params" and v else v  # 기본값만 남아 None 이 된 params 도 처리
    if out.get("params"):
        out["params"] = {k: v for k, v in out["params"].items() if LGBM_PARAMS.get(k) != v} or None
    return out


def run_cfg(feats_df: pd.DataFrame, target: str, name: str, cfg: dict, start: str, end: str | None,
            freq: str, stage: str, keep_preds: bool = False, resumable: bool = False) -> dict:
    """resumable=True 이면 기간별 예측을 디스크에 남겨 두어, 긴 워크포워드가 중간에 끊겨도 다시 부르면 이어서 돈다.
    평가가 끝나면 그 임시 파일은 지운다."""
    features = select_features(only_groups=cfg["groups"])
    cache = _exp_dir(target) / "wf" / f"{stage}__{name}" if resumable else None
    preds = walk_forward(feats_df, target, start, end, freq=freq, kind=cfg["kind"], params=cfg["params"] or None,
                         neg_rate=cfg["neg_rate"], features=features, exclude=tuple(cfg["exclude"]),
                         train_years=cfg["train_years"], verbose=False, hard_neg=cfg.get("hard_neg", False),
                         cache=cache)
    if cache is not None:
        shutil.rmtree(cache, ignore_errors=True)
    ev = evaluate(preds, target)
    hits = ev["daily"][["n", "양성"] + [c for c in ev["daily"].columns if c.startswith("hit@")]]
    hits.to_parquet(_exp_dir(target) / f"{stage}__{name}.parquet")
    log_experiment(f"{stage}:{name}", target, {**cfg, "freq": freq, "period": f"{start}~{end or '현재'}",
                                               "n_features": len(features)}, ev["summary"])
    out = {"name": name, "stage": stage, "cfg": cfg, "summary": ev["summary"], "daily": hits}
    if keep_preds:
        out["preds"] = preds
    return out


def paired(a: pd.DataFrame, b: pd.DataFrame, k: int = 10) -> tuple[float, float]:
    """같은 날짜끼리 짝지어 P@k 차이(a-b)의 평균과 t 값."""
    j = a[[f"hit@{k}"]].join(b[[f"hit@{k}"]], lsuffix="_a", rsuffix="_b", how="inner").fillna(0)
    d = (j[f"hit@{k}_a"] - j[f"hit@{k}_b"]) / k
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return float(d.mean()), float(d.mean() / se) if se and se > 0 else 0.0


class Search:
    def __init__(self, feats_df, target, freq="YS", verbose=True, dev=DEV):
        self.f, self.target, self.freq, self.verbose, self.dev = feats_df, target, freq, verbose, dev
        self.k = PRIMARY_K.get(target, 10)
        self.rows: list[dict] = []
        self.champ: dict | None = None

    def _row(self, res, gain, t, accepted, note=""):
        s = res["summary"]
        self.rows.append({"단계": res["stage"], "후보": res["name"], "P@5": s["P@5"], "P@10": s["P@10"],
                          "P@20": s["P@20"], "AP": s["AP"], "AUC": s["AUC"], f"ΔP@{self.k}(vs 챔피언)": gain,
                          "t": t, "채택": accepted, "비고": note, "설정": json.dumps(res["cfg"], ensure_ascii=False)})
        if self.verbose:
            print(f"  [{res['stage']}] {res['name']:<22} P@{self.k} {s[f'P@{self.k}']:.4f}  AP {s['AP']:.4f}  "
                  f"Δ {gain:+.4f} (t={t:+.2f}) {'✔ 채택' if accepted else ''} {note}", flush=True)

    def run(self, name, cfg, stage):
        return run_cfg(self.f, self.target, name, cfg, self.dev[0], self.dev[1], self.freq, stage)

    def baseline(self, cfg=None):
        res = self.run("v1_base", cfg or BASE_CFG, "base")
        self.champ = res
        self._row(res, 0.0, 0.0, True, "출발점(v1 설정)")

    def challenge(self, name, change, stage, accept=True):
        res = self.run(name, _merge(self.champ["cfg"], change), stage)
        gain, t = paired(res["daily"], self.champ["daily"], self.k)
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
        for g in [g for g in NEW_GROUPS if g not in self.champ["cfg"]["groups"]]:  # 합쳐 볼 후보만 고르고, 채택은 합친 결과로
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
    champ_cfg, k = s.champ["cfg"], s.k
    pk = f"P@{k}"
    print(f"\n개발 구간 최종 챔피언: {s.champ['name']}  {pk} {s.champ['summary'][pk]:.4f} "
          f"(출발점 {s.rows[0][pk]:.4f})", flush=True)

    # ---- 10년 워크포워드(분기 재학습)로 v1 과 챔피언 비교 → 보류 구간(2024~)으로 판정
    ten_start = dev[0]
    print(f"\n[10년 백테스트] {ten_start}~현재, 분기 재학습", flush=True)
    base_cfg = base_cfg or BASE_CFG
    runs = {"v1": run_cfg(feats_df, target, "v1", base_cfg, ten_start, None, "QS", "10y", keep_preds=True)}
    if champ_cfg != base_cfg:
        runs["champion"] = run_cfg(feats_df, target, "champion", champ_cfg, ten_start, None, "QS", "10y",
                                   keep_preds=True)
    ten = ten_year_tables(runs, target, k, hold_start)
    gain, t = ten["hold_gain"], ten["hold_t"]
    better = "champion" in runs and gain >= 0
    verdict = (f"보류 구간(2024~)에서도 v1보다 좋거나 같음 (Δ{pk} {gain:+.4f}, t={t:+.2f}) → 승격" if better else
               f"보류 구간에서 v1보다 나쁨 (Δ{pk} {gain:+.4f}, t={t:+.2f}) → 승격하지 않음 (개발 구간 과최적화)"
               if "champion" in runs else "개선 후보 없음 → v1 유지")
    print(f"판정: {verdict}", flush=True)
    _plot(board, ten["yearly"], target, k)

    out = {"dev_board": board, "ten_year": ten, "champion_cfg": champ_cfg, "verdict": verdict, "better": better}
    if promote and better:
        name = _next_version(target)
        cdir = config.ROOT / "configs"
        cdir.mkdir(exist_ok=True)
        (cdir / f"{name}.json").write_text(json.dumps({
            "name": name, "target": target, **champ_cfg, "dev": s.champ["summary"],
            f"holdout_gain_{pk}": gain, "holdout_t": t, "created": str(t0)[:19]},
            ensure_ascii=False, indent=2, default=float))
        print(f"\n승격: {name} 학습 중 (전체 기간)", flush=True)
        train_final(feats_df, target, kind=champ_cfg["kind"], params=champ_cfg["params"] or None,
                    neg_rate=champ_cfg["neg_rate"], exclude=tuple(champ_cfg["exclude"]), name=name,
                    features=select_features(only_groups=champ_cfg["groups"]), train_years=champ_cfg["train_years"],
                    hard_neg=champ_cfg.get("hard_neg", False))
        out["promoted"] = name
    print(f"소요 {datetime.now() - t0}", flush=True)
    return out


def ten_year_tables(runs: dict, target: str, k: int, hold_start: str = HOLD_START, save: bool = True) -> dict:
    """10년 워크포워드 결과를 기간별(전체/개발/보류)·연도별로 정리. 단순 규칙 기준선도 함께.
    runs: {기준 모델 이름: 결과, (도전 모델 이름: 결과)} — 첫 번째가 기준."""
    pk = f"P@{k}"
    names = list(runs)
    base_name, new_name = names[0], (names[1] if len(names) > 1 else None)
    rule = runs[base_name]["preds"].copy()
    rule["rule"] = rule_score(rule, target)
    rule_daily = evaluate(rule, target, score="rule")["daily"]
    dailies = {"규칙": rule_daily, **{n: r["daily"] for n, r in runs.items()}}
    hs = pd.Timestamp(hold_start)
    rows = []
    for period, sel in [("10년 전체", lambda d: d.index >= d.index.min()), ("개발 2017~2023", lambda d: d.index < hs),
                        ("보류 2024~", lambda d: d.index >= hs)]:
        for name, d in dailies.items():
            dd = d[sel(d)]
            row = {"기간": period, "모델": name, "일수": len(dd), "기저율": dd["양성"].sum() / dd["n"].sum()}
            for kk in (5, 10, 20):
                row[f"P@{kk}"] = dd[f"hit@{kk}"].sum() / (kk * len(dd))
            rows.append(row)
    periods = pd.DataFrame(rows)
    periods[f"Lift@{k}"] = periods[pk] / periods["기저율"]

    yearly = []
    for name, d in dailies.items():
        g = d.groupby(d.index.year)
        y = pd.DataFrame({"일수": g.size(), "실제발생": g["양성"].sum(), pk: g[f"hit@{k}"].sum() / (k * g.size())})
        y["모델"] = name
        yearly.append(y.reset_index(names="연도"))
    yearly = pd.concat(yearly).pivot_table(index="연도", columns="모델", values=pk)
    yearly.columns = [f"{c} {pk}" for c in yearly.columns]
    yearly = yearly[[f"{n} {pk}" for n in names + ["규칙"]]]  # 기준 → 도전 → 규칙 순서

    def _cmp(a, b, mask_fn):
        aa, bb = dailies[a], dailies[b]
        return paired(aa[mask_fn(aa)], bb[mask_fn(bb)], k)
    comp = []
    for a in [n for n in dailies if n != "규칙"]:
        for period, fn in [("10년 전체", lambda d: d.index >= d.index.min()), ("보류 2024~", lambda d: d.index >= hs)]:
            g, t = _cmp(a, "규칙", fn)
            comp.append({"비교": f"{a} − 규칙", "기간": period, f"Δ{pk}": g, "t": t})
    hold_gain, hold_t = 0.0, 0.0
    if new_name:
        for period, fn in [("10년 전체", lambda d: d.index >= d.index.min()), ("개발 2017~2023", lambda d: d.index < hs),
                           ("보류 2024~", lambda d: d.index >= hs)]:
            g, t = _cmp(new_name, base_name, fn)
            comp.append({"비교": f"{new_name} − {base_name}", "기간": period, f"Δ{pk}": g, "t": t})
            if period.startswith("보류"):
                hold_gain, hold_t = g, t
    comp = pd.DataFrame(comp)
    if save:
        save_table(periods.round(5), f"g5_10y_periods_{target}")
        save_table(yearly.round(4).reset_index(), f"g5_10y_yearly_{target}")
        save_table(comp.round(5), f"g5_10y_paired_{target}")
    print(periods.round(4).to_string(index=False), flush=True)
    print(comp.round(4).to_string(index=False), flush=True)
    return {"periods": periods, "yearly": yearly, "paired": comp, "hold_gain": hold_gain, "hold_t": hold_t}


def _plot(board: pd.DataFrame, yearly: pd.DataFrame, target: str, k: int) -> None:
    pk = f"P@{k}"
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
    lbl = board["단계"] + ":" + board["후보"]
    colors = np.where(board["채택"], UP_COLOR, "#bbbbbb")
    ax[0].barh(lbl[::-1], board[pk][::-1], color=colors[::-1])
    ax[0].axvline(board[pk].iloc[0], color=NEUTRAL, ls="--", label="출발점(v1)")
    ax[0].set_title(f"개발 구간(2017~2023) 후보별 {pk} — 빨강: 채택")
    ax[0].set_xlim(board[pk].min() * 0.9, board[pk].max() * 1.03)
    ax[0].legend()
    ax[0].tick_params(axis="y", labelsize=8)
    palette = [("#1f77b4", "-"), (UP_COLOR, "-")]
    for col in yearly.columns:
        name = col.split()[0]
        c, ls = (NEUTRAL, "--") if name == "규칙" else palette.pop(0) if palette else ("k", "-")
        ax[1].plot(yearly.index, yearly[col], ls, marker="o", color=c, label=name)
    ax[1].axvspan(2023.5, yearly.index.max() + 0.5, color="orange", alpha=0.1, label="보류 구간")
    ax[1].set_title(f"10년 워크포워드 연도별 {pk} — {TARGETS[target]}")
    ax[1].legend()
    save_fig(fig, f"g5_improve_{target}")
