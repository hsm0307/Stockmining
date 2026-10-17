"""클라우드 장시간 자동 개선 루프 — 사용자가 멈출 때까지 상한가·하한가 모델을 번갈아 개선한다.

한 반복(iteration)
  1) 현재 '개발 챔피언' 설정을 조금 바꾼 후보를 만든다.
     처음에는 정해진 후보(특징 그룹, 파라미터, 모델 종류, 학습기간)를 차례로, 그 뒤에는 무작위 변이.
  2) 개발 구간(2017~2023, 연 1회 재학습 워크포워드)에서 챔피언과 같은 날끼리 짝지어 비교.
     ΔP@k > 0, t ≥ 1 이면 새 개발 챔피언.
  3) 개발 챔피언이 운영 버전(v1, v2, …)과 달라졌으면 10년(2017~현재, 분기 재학습) 백테스트로 검증.
     한 번도 고르는 데 쓰지 않은 보류 구간(2024~현재)에서 운영 버전보다 나쁘지 않으면 새 버전으로 승격
     (configs/{target}_vN.json + models/{target}_vN.pkl). 나쁘면 개발 챔피언을 운영 버전으로 되돌린다.
  4) 결과표를 저장하고 git 에 커밋·푸시한다.
상태는 outputs/evolve/state_{target}.json 에 남아 프로세스가 끊겨도 이어서 진행한다.

주의: 보류 구간을 여러 번 확인하면 그 구간에도 조금씩 맞춰질 수 있다. 진짜 검증은 앞으로 2달의 실전 예측이다.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_GROUPS, load_features
from .improve import (BASE_CFG, DEV, HOLD_START, MIN_T, MODEL_CANDIDATES, NEW_GROUPS, PARAM_CANDIDATES, PRIMARY_K,
                      WINDOW_CANDIDATES, _exp_dir, _merge, _next_version, paired, run_cfg, ten_year_tables)
from .model import TARGETS, evaluate, select_features, train_final
from .viz import NEUTRAL, UP_COLOR, plt, save_fig, save_table

# 라벨·특징 정의가 바뀌면 올린다 → 이전 세대의 상태·캐시와 섞이지 않는다.
#   v1: 최초 (다음 날 거래정지 종목을 후보에서 제외하던 라벨)
#   v2-haltlabel: 다음 날 거래정지·무효·상폐도 '실패'로 세는 라벨 + 시장경보·단기과열 특징 + 동조종목·리츠 수정
DATA_VERSION = "v2-haltlabel"
STATE_DIR = config.OUTPUT_DIR / "evolve" / DATA_VERSION
DEV_STAGE, TEN_STAGE = f"evo-{DATA_VERSION}", f"10y-{DATA_VERSION}"
PROTECTED_GROUPS = {"candle", "limit_hist"}  # hard_neg 규칙과 핵심 신호에 필요한 그룹은 빼지 않음
SPACE = {
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [50, 100, 200, 400, 800],
    "colsample_bytree": [0.4, 0.55, 0.7, 0.85, 1.0],
    "subsample": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0, 10.0, 30.0],
    "learning_rate": [0.02, 0.03, 0.05],
}
NEG_RATES = [0.03, 0.05, 0.1, 0.2, 0.3]
KINDS = ["lgbm", "bag3", "rank", "ens"]
TRAIN_YEARS = [None, 3, 5]
EST_DEV_MIN = {"lgbm": 6, "rank": 8, "bag3": 18, "ens": 15}  # 개발 평가 1회 예상 소요(분, 여유 포함)
EST_TEN_MIN = {"lgbm": 30, "rank": 40, "bag3": 90, "ens": 70}  # 10년 분기 워크포워드 1회 예상 소요(분)


def cfg_key(cfg: dict) -> str:
    c = {**cfg, "groups": sorted(cfg["groups"]), "exclude": sorted(cfg["exclude"])}
    return hashlib.sha1(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _state_path(target):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"state_{target}.json"


def load_state(target: str) -> dict:
    p = _state_path(target)
    if p.exists():
        return json.loads(p.read_text())
    return {"target": target, "iteration": 0, "dev_champion": None, "production": None, "queue_pos": 0,
            "accepted_since_validation": 0, "tested": {}, "versions": [], "validations": []}


def save_state(st: dict) -> None:
    _state_path(st["target"]).write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float))


def _dev_hits(target: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(_exp_dir(target) / f"{DEV_STAGE}__{key}.parquet")


def latest_config(target: str) -> tuple[str, dict]:
    """configs/ 의 가장 최근 승격 버전 (없으면 v1 기본 설정)."""
    cdir = config.ROOT / "configs"
    best, ver = None, 1
    for p in cdir.glob(f"{target}_v*.json"):
        tail = p.stem.rsplit("_v", 1)[1]
        if tail.isdigit() and int(tail) > ver:
            best, ver = p, int(tail)
    if best is None:
        return target, json.loads(json.dumps(BASE_CFG))
    c = json.loads(best.read_text())
    cfg = {k: c.get(k, BASE_CFG[k]) for k in BASE_CFG}
    return c["name"], cfg


def systematic(cfg: dict) -> list[tuple[str, dict]]:
    """초기 체계적 후보 목록 (improve 의 라운드와 같은 후보)."""
    out = [(f"+{g}", {"groups": cfg["groups"] + [g]}) for g in NEW_GROUPS if g not in cfg["groups"]]
    out += list(PARAM_CANDIDATES.items()) + list(MODEL_CANDIDATES.items()) + list(WINDOW_CANDIDATES.items())
    return out


def mutate(cfg: dict, rng: random.Random) -> tuple[str, dict]:
    """설정 하나(가끔 둘)를 바꾼 변이 후보."""
    changes, names = {}, []
    for _ in range(1 if rng.random() < 0.7 else 2):
        op = rng.choice(["group", "param", "param", "neg", "hard", "kind", "years", "spac"])
        if op == "group":
            g = rng.choice([g for g in FEATURE_GROUPS if g not in PROTECTED_GROUPS])
            groups = list(changes.get("groups", cfg["groups"]))
            groups = [x for x in groups if x != g] if g in groups else groups + [g]
            changes["groups"] = groups
            names.append(("-" if g in cfg["groups"] else "+") + g)
        elif op == "param":
            p = rng.choice(list(SPACE))
            v = rng.choice(SPACE[p])
            changes.setdefault("params", {})[p] = v
            names.append(f"{p}={v}")
        elif op == "neg":
            v = rng.choice(NEG_RATES)
            changes["neg_rate"] = v
            names.append(f"neg={v}")
        elif op == "hard":
            changes["hard_neg"] = not cfg.get("hard_neg", False)
            names.append(f"hard_neg={changes['hard_neg']}")
        elif op == "kind":
            v = rng.choice([k for k in KINDS if k != cfg["kind"]])
            changes["kind"] = v
            if v == "rank":
                changes["params"] = {}
            names.append(f"kind={v}")
        elif op == "years":
            v = rng.choice([y for y in TRAIN_YEARS if y != cfg["train_years"]])
            changes["train_years"] = v
            names.append(f"years={v}")
        else:
            ex = list(cfg["exclude"])
            changes["exclude"] = [e for e in ex if e != "spac"] if "spac" in ex else ex + ["spac"]
            names.append("spac " + ("포함" if "spac" in ex else "제외"))
    return " & ".join(names), changes


class Evolver:
    def __init__(self, feats: pd.DataFrame, seed: int = 0, verbose: bool = True):
        self.f = feats
        self.rng = random.Random(seed + int(time.time()))
        self.verbose = verbose

    def log(self, msg):
        if self.verbose:
            print(f"[{datetime.now():%m-%d %H:%M}] {msg}", flush=True)

    def eval_dev(self, target: str, name: str, cfg: dict) -> dict:
        key = cfg_key(cfg)
        res = run_cfg(self.f, target, key, cfg, DEV[0], DEV[1], "YS", DEV_STAGE)
        res["key"], res["label"] = key, name
        return res

    def ensure_baseline(self, st: dict) -> None:
        target = st["target"]
        if st["dev_champion"] is None:
            name, cfg = latest_config(target)  # 가장 최근 승격 버전에서 출발
            res = self.eval_dev(target, f"{name}(출발점)", cfg)
            k = PRIMARY_K[target]
            st["dev_champion"] = {"key": res["key"], "label": name, "cfg": cfg, "score": res["summary"][f"P@{k}"]}
            st["production"] = {"name": name, "key": res["key"], "cfg": cfg}
            st["tested"][res["key"]] = {"label": f"{name}(출발점)", "P@k": res["summary"][f"P@{k}"],
                                        "AP": res["summary"]["AP"], "gain": 0.0, "t": 0.0, "accepted": True,
                                        "time": f"{datetime.now():%m-%d %H:%M}", "cfg": cfg}
            st["last_event"] = f"evolve {target}: baseline P@{k}={res['summary'][f'P@{k}']:.4f}"
            self.log(f"{target}: 출발점 P@{k} {res['summary'][f'P@{k}']:.4f}")

    def next_candidate(self, st: dict) -> tuple[str, dict] | None:
        champ = st["dev_champion"]["cfg"]
        sysq = systematic(champ)
        while st["queue_pos"] < len(sysq):
            name, change = sysq[st["queue_pos"]]
            st["queue_pos"] += 1
            cfg = _merge(champ, change)
            if cfg_key(cfg) not in st["tested"]:
                return name, cfg
        for _ in range(200):
            name, change = mutate(champ, self.rng)
            cfg = _merge(champ, change)
            if cfg.get("hard_neg") and not {"candle", "limit_hist"} <= set(cfg["groups"]):
                continue
            if cfg_key(cfg) not in st["tested"]:
                return name, cfg
        return None

    def dev_step(self, st: dict) -> None:
        target, k = st["target"], PRIMARY_K[st["target"]]
        cand = self.next_candidate(st)
        if cand is None:
            return
        name, cfg = cand
        res = self.eval_dev(target, name, cfg)
        gain, t = paired(res["daily"], _dev_hits(target, st["dev_champion"]["key"]), k)
        ok = gain > 0 and t >= MIN_T
        st["tested"][res["key"]] = {"label": name, "P@k": res["summary"][f"P@{k}"], "AP": res["summary"]["AP"],
                                    "gain": gain, "t": t, "accepted": ok, "time": f"{datetime.now():%m-%d %H:%M}",
                                    "base": st["dev_champion"]["key"], "cfg": cfg}
        self.log(f"{target}: {name:<40} P@{k} {res['summary'][f'P@{k}']:.4f}  Δ{gain:+.4f} t={t:+.2f}"
                 + ("  ✔ 개발 챔피언" if ok else ""))
        st["last_event"] = (f"evolve {target}: try #{len(st['tested'])} {name} "
                            f"(dP@{k}={gain:+.4f}, t={t:+.2f}{', accepted' if ok else ''})")
        if ok:
            st["dev_champion"] = {"key": res["key"], "label": name, "cfg": cfg, "score": res["summary"][f"P@{k}"]}
            st["accepted_since_validation"] += 1
            st["queue_pos"] = 0  # 챔피언이 바뀌면 체계적 후보를 새 챔피언 기준으로 다시 훑음 (이미 본 것은 건너뜀)

    def needs_validation(self, st: dict) -> bool:
        return st["dev_champion"]["key"] != st["production"]["key"] and st["accepted_since_validation"] > 0

    def validate(self, st: dict) -> None:
        """개발 챔피언 vs 운영 버전: 10년 분기 재학습 워크포워드 → 보류 구간(2024~)으로 판정."""
        target, k = st["target"], PRIMARY_K[st["target"]]
        prod, champ = st["production"], st["dev_champion"]
        self.log(f"{target}: 10년 검증 시작 — 운영 {prod['name']} vs 개발 챔피언 '{champ['label']}'")
        runs = {}
        for label, item in [(prod["name"], prod), ("도전자", champ)]:
            cache = STATE_DIR / f"ten_{target}_{item['key']}.parquet"
            if cache.exists():
                preds = pd.read_parquet(cache)
                ev = evaluate(preds, target)
                runs[label] = {"preds": preds, "daily": ev["daily"], "summary": ev["summary"]}
            else:
                r = run_cfg(self.f, target, f"{label}_{item['key']}", item["cfg"], DEV[0], None, "QS", TEN_STAGE,
                            keep_preds=True)
                r["preds"].to_parquet(cache, index=False)
                runs[label] = r
        ten = ten_year_tables(runs, target, k, HOLD_START)
        gain, t = ten["hold_gain"], ten["hold_t"]
        pr = ten["paired"]
        ten_row = pr[(pr["기간"] == "10년 전체") & (pr["비교"] == f"도전자 − {prod['name']}")]  # 운영 버전 대비
        rec = {"time": f"{datetime.now():%Y-%m-%d %H:%M}", "production": prod["name"], "challenger": champ["label"],
               "challenger_key": champ["key"], "hold_gain": gain, "hold_t": t,
               "ten_gain": float(ten_row[f"ΔP@{k}"].iloc[0]), "ten_t": float(ten_row["t"].iloc[0])}
        if gain >= 0:
            name = _next_version(target)
            cdir = config.ROOT / "configs"
            cdir.mkdir(exist_ok=True)
            cfg = champ["cfg"]
            (cdir / f"{name}.json").write_text(json.dumps(
                {"name": name, "target": target, **cfg, "dev_label": champ["label"], f"holdout_gain_P@{k}": gain,
                 "holdout_t": t, "ten_year_gain": rec["ten_gain"], "ten_year_t": rec["ten_t"],
                "data_version": DATA_VERSION, "created": rec["time"]},
                ensure_ascii=False, indent=2, default=float))
            train_final(self.f, target, kind=cfg["kind"], params=cfg["params"] or None, neg_rate=cfg["neg_rate"],
                        exclude=tuple(cfg["exclude"]), name=name, features=select_features(only_groups=cfg["groups"]),
                        train_years=cfg["train_years"], hard_neg=cfg.get("hard_neg", False))
            st["production"] = {"name": name, "key": champ["key"], "cfg": cfg}
            per = ten["periods"]
            st["versions"].append({"name": name, "time": rec["time"], "label": champ["label"], "hold_gain": gain,
                                   "hold_t": t, "ten_gain": rec["ten_gain"],
                                   "hold_P": float(per[(per["기간"] == "보류 2024~") & (per["모델"] == "도전자")]
                                                   [f"P@{k}"].iloc[0])})
            rec["result"] = f"승격 → {name}"
            self.log(f"{target}: 보류 구간 Δ{gain:+.4f} (t={t:+.2f}) → {name} 승격")
        else:
            rec["result"] = "기각 (보류 구간에서 나쁨) → 개발 챔피언을 운영 버전으로 되돌림"
            st["dev_champion"] = {"key": prod["key"], "label": prod["name"], "cfg": prod["cfg"], "score": None}
            st["queue_pos"] = 0
            self.log(f"{target}: 보류 구간 Δ{gain:+.4f} (t={t:+.2f}) → 기각")
        st["validations"].append(rec)
        st["accepted_since_validation"] = 0
        st["last_event"] = f"evolve {target}: 10y validation — {rec['result']} (holdout dP@{k}={gain:+.4f}, t={t:+.2f})"

    def _ten_cache(self, target: str, key: str):
        return STATE_DIR / f"ten_{target}_{key}.parquet"

    def baseline_10y(self, st: dict) -> None:
        """운영 버전의 10년 워크포워드 성적(규칙 기준선 포함)을 한 번 만들어 둔다 → 이후 검증 때 재사용."""
        target, k = st["target"], PRIMARY_K[st["target"]]
        prod = st["production"]
        self.log(f"{target}: 운영 버전 {prod['name']} 10년 백테스트 (기준선)")
        r = run_cfg(self.f, target, f"{prod['name']}_{prod['key']}", prod["cfg"], DEV[0], None, "QS", TEN_STAGE,
                    keep_preds=True)
        r["preds"].to_parquet(self._ten_cache(target, prod["key"]), index=False)
        ten = ten_year_tables({prod["name"]: r}, target, k, HOLD_START)
        per = ten["periods"]
        st["ten_baseline"] = per.round(6).to_dict("records")
        row = per[(per["기간"] == "10년 전체") & (per["모델"] == prod["name"])].iloc[0]
        st["last_event"] = f"evolve {target}: 10y baseline of {prod['name']} P@{k}={row[f'P@{k}']:.4f}"

    def step(self, st: dict, minutes_left: float) -> bool:
        """한 단계 실행. 남은 시간 안에 못 끝낼 것 같으면 False."""
        self.ensure_baseline(st)
        prod = st["production"]
        if not self._ten_cache(st["target"], prod["key"]).exists() and not self.needs_validation(st):
            if minutes_left < EST_TEN_MIN.get(prod["cfg"]["kind"], 60):
                return False
            self.baseline_10y(st)
            st["iteration"] += 1
            return True
        if self.needs_validation(st):
            need = EST_TEN_MIN.get(st["dev_champion"]["cfg"]["kind"], 60)
            if not (STATE_DIR / f"ten_{st['target']}_{st['production']['key']}.parquet").exists():
                need += EST_TEN_MIN.get(st["production"]["cfg"]["kind"], 60)
            if minutes_left < need:
                return False
            self.validate(st)
        else:
            if minutes_left < EST_DEV_MIN.get(st["dev_champion"]["cfg"]["kind"], 10) + 5:
                return False
            self.dev_step(st)
        st["iteration"] += 1
        return True


def report(st: dict) -> None:
    target, k = st["target"], PRIMARY_K[st["target"]]
    rows = [{"키": key, "후보": v["label"], f"P@{k}": v["P@k"], "AP": v["AP"], "Δ(vs 당시 챔피언)": v["gain"],
             "t": v["t"], "채택": v["accepted"], "시각": v["time"]} for key, v in st["tested"].items()]
    board = pd.DataFrame(rows)  # 시도 순서
    save_table(board.sort_values(f"P@{k}", ascending=False).round(5), f"g5_evolve_board_{target}")
    save_table(pd.DataFrame(st["validations"]).round(5) if st["validations"] else pd.DataFrame(),
               f"g5_evolve_validations_{target}")
    save_table(pd.DataFrame(st["versions"]).round(5) if st["versions"] else pd.DataFrame(),
               f"g5_evolve_versions_{target}")
    if len(board) < 2:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(range(len(board)), board[f"P@{k}"], c=np.where(board["채택"], UP_COLOR, "#bbbbbb"), s=25)
    best = board[f"P@{k}"].where(board["채택"]).ffill()
    ax.step(range(len(board)), best, where="post", color=UP_COLOR, label="개발 챔피언")
    ax.axhline(board[f"P@{k}"].iloc[0], color=NEUTRAL, ls="--", label="출발점(v1)")
    ax.set_xlabel("시도 순서")
    ax.set_ylabel(f"개발 구간(2017~2023) P@{k}")
    ax.set_title(f"자동 개선 진행 — {TARGETS[target]} (빨강: 채택, 시도 {len(board)}회, "
                 f"승격 {len(st['versions'])}회)")
    ax.legend()
    save_fig(fig, f"g5_evolve_{target}")


def git_sync(message: str) -> None:
    """진행 결과를 지정 브랜치에 커밋·푸시 (실패해도 루프는 계속)."""
    root = str(config.ROOT)
    trailer = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
               "Claude-Session: https://claude.ai/code/session_01Azv4PmjewJvuqXFSnT9XVB")
    try:
        paths = [p for p in ("outputs", "configs") if (config.ROOT / p).exists()]  # 없는 경로가 있으면 git add 전체 실패
        subprocess.run(["git", "add", *paths], cwd=root, check=False, capture_output=True)
        r = subprocess.run(["git", "commit", "-q", "-m", message + trailer], cwd=root, capture_output=True)
        if r.returncode != 0:
            return
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True,
                                text=True).stdout.strip()
        for wait in (0, 2, 4, 8, 16):
            time.sleep(wait)
            if subprocess.run(["git", "push", "-q", "origin", branch], cwd=root, capture_output=True).returncode == 0:
                return
    except Exception as e:  # noqa: BLE001
        print(f"[git] {e}", flush=True)


def run(targets=("y_up", "y_down"), hours: float = 1.6, sync: bool = True) -> None:
    deadline = time.time() + hours * 3600
    feats = load_features()
    ev = Evolver(feats)
    states = {t: load_state(t) for t in targets}
    print(f"자동 개선 시작: {list(targets)}, 이번 실행 {hours}시간 (상태 파일로 이어서 진행)", flush=True)
    idle = 0
    while True:
        progressed = False
        for t in targets:
            minutes_left = (deadline - time.time()) / 60
            st = states[t]
            if not ev.step(st, minutes_left):
                continue
            progressed = True
            save_state(st)
            report(st)
            if sync:
                git_sync(st.get("last_event", f"evolve {t}"))
        if not progressed:
            idle += 1
            if idle >= 1:
                break
    print("이번 실행 종료 (다음 실행에서 이어서 진행)", flush=True)
