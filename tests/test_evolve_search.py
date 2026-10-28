"""자동 개선 루프의 무작위 변이: 같은 변경을 두 번 고르지 않고, 번번이 나빴던 변경은 대체로 건너뛴다."""
import random

from krxlimit import evolve as evo
from krxlimit.improve import _merge

CHAMP = {"groups": ["candle", "momentum", "limit_hist", "volume", "peer"], "exclude": [], "kind": "bag3",
         "train_years": 3, "neg_rate": 0.1, "hard_neg": False, "params": {"num_leaves": 63, "learning_rate": 0.05}}


def _state(results):
    """results: [(label, change, t)] — 모두 현 챔피언('c0') 대비 시도. 기록 키는 실제 설정의 cfg_key."""
    tested = {"c0": {"label": "v5(출발점)", "t": 0.0, "cfg": CHAMP}}
    for label, change, t in results:
        cfg = _merge(CHAMP, change)
        tested[evo.cfg_key(cfg)] = {"label": label, "t": t, "base": "c0", "cfg": cfg}
    return {"dev_champion": {"key": "c0", "cfg": CHAMP}, "tested": tested, "queue_pos": 10 ** 6}


def test_mutate_never_repeats_a_single_choice_change():
    rng = random.Random(0)
    for _ in range(3000):
        name, change = evo.mutate(CHAMP, rng)
        parts = name.split(" & ")
        assert len(parts) == len(set(parts)), name
        keys = [p.split("=")[0] for p in parts if "=" in p]
        assert len(keys) == len(set(keys)), name  # kind=rank & kind=lgbm, num_leaves 두 번 등 금지


def test_mutate_names_only_real_changes():
    # 이름의 각 변경이 실제 설정 차이와 일치해야 함 — 예전에는 챔피언 값과 같은 num_leaves=63 이나 neg=0.1 을 골라
    # 아무것도 안 바뀐 변경이 이름에만 남았고, 파라미터를 고른 뒤 rank 로 바꾸면 그 파라미터가 지워졌음
    tuned = {**CHAMP, "params": {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 50, "subsample": 0.6}}
    champs = (CHAMP, {**CHAMP, "params": None, "neg_rate": 0.2}, tuned,  # tuned: rank 기본값과 다른 값이 많아야
              _merge(CHAMP, {"kind": "rank", "params": {}}))              # rank 기본값을 고르는 실수가 드러남
    for champ in champs:
        rng = random.Random(1)
        rank_combos = 0
        for _ in range(4000):
            name, change = evo.mutate(champ, rng)
            cfg = _merge(champ, change)
            parts = set(name.split(" & "))
            params = {p for p in parts if p.split("=")[0] in evo.SPACE}
            got = {f"{k}={v}" for k, v in (cfg["params"] or {}).items()}
            if cfg["kind"] == "rank" != champ["kind"]:  # cfg_moves 는 rank 전환 때 파라미터 차이를 세지 않음 → 직접 확인
                assert evo.cfg_moves(champ, cfg) == parts - params, name
                assert got == params, (name, cfg["params"])  # rank 기본값에서 출발, 고른 파라미터만 남음
                rank_combos += bool(params)
            elif cfg["kind"] != champ["kind"] == "rank":
                assert evo.cfg_moves(champ, cfg) == parts - params and params <= got, (name, cfg["params"])
            else:
                assert evo.cfg_moves(champ, cfg) == parts, name
        assert champ["kind"] == "rank" or rank_combos >= 5
    # 파라미터 없이 rank 로만 바꾼 후보는 예전과 같은 설정(같은 키) → 이미 평가한 결과를 다시 돌리지 않음
    rank_only = _merge(CHAMP, {"kind": "rank", "params": {}})
    rng = random.Random(0)
    found = [c for n, c in (evo.mutate(CHAMP, rng) for _ in range(3000)) if n == "kind=rank"]
    assert found and all(evo.cfg_key(_merge(CHAMP, c)) == evo.cfg_key(rank_only) for c in found)


def test_cfg_moves_names_systematic_and_mutation_candidates_alike():
    assert evo.cfg_moves(CHAMP, _merge(CHAMP, {"params": {"min_child_samples": 50}})) == {"min_child_samples=50"}
    assert evo.cfg_moves(CHAMP, _merge(CHAMP, {"hard_neg": True, "neg_rate": 0.03})) == {"hard_neg=True", "neg=0.03"}
    assert evo.cfg_moves(CHAMP, _merge(CHAMP, {"exclude": ["spac"], "groups": CHAMP["groups"][:-1]})) == {
        "spac 제외", "-peer"}
    # rank 로 바꾸면 champion 파라미터가 지워지지만 그건 rank 변경의 일부로 본다
    assert evo.cfg_moves(CHAMP, _merge(CHAMP, {"kind": "rank", "params": {}})) == {"kind=rank"}
    # 기본값으로 되돌린 파라미터도 이름이 생긴다 (num_leaves 63 → 기본 31)
    assert evo.cfg_moves(CHAMP, _merge(CHAMP, {"params": {"num_leaves": 31}})) == {"num_leaves=31"}


def test_bad_moves_uses_solo_results_stored_under_systematic_names():
    st = _state([("minchild50", {"params": {"min_child_samples": 50}}, -1.18),  # 단독 결과는 기준보다 나음
                 ("years=5 & min_child_samples=50", {"train_years": 5, "params": {"min_child_samples": 50}}, -3.9),
                 ("kind=lgbm & min_child_samples=50", {"kind": "lgbm", "params": {"min_child_samples": 50}}, -2.7),
                 ("-peer & min_child_samples=50", {"groups": CHAMP["groups"][:-1],
                                                   "params": {"min_child_samples": 50}}, -2.1),
                 ("hardneg", {"hard_neg": True}, -4.1),
                 ("hard_neg=True & -peer", {"hard_neg": True, "groups": CHAMP["groups"][:-1]}, -4.0),
                 ("num_leaves=127 & hard_neg=True", {"hard_neg": True, "params": {"num_leaves": 127}}, -2.3)])
    st["tested"]["old"] = {"label": "hardneg", "t": 1.0, "base": "older_champion",  # 이전 챔피언 대비는 세지 않음
                           "cfg": _merge(CHAMP, {"hard_neg": True, "neg_rate": 0.3})}
    assert evo.bad_moves(st) == {"hard_neg=True"}


def test_bad_moves_needs_more_evidence_when_only_seen_in_combos():
    combos = [({"train_years": 5, "params": {"subsample": 0.6}}, -3.0),
              ({"neg_rate": 0.2, "params": {"subsample": 0.6}}, -2.5),
              ({"kind": "lgbm", "params": {"subsample": 0.6}}, -2.4)]
    st = _state([(f"c{i}", ch, t) for i, (ch, t) in enumerate(combos)])
    assert "subsample=0.6" not in evo.bad_moves(st)  # 섞여서 3번뿐 — 짝의 탓일 수 있음
    more = combos + [({"hard_neg": True, "params": {"subsample": 0.6}}, -2.2),
                     ({"neg_rate": 0.3, "params": {"subsample": 0.6}}, -2.6)]
    st = _state([(f"c{i}", ch, t) for i, (ch, t) in enumerate(more)])
    assert "subsample=0.6" in evo.bad_moves(st)


def test_bad_moves_flags_frequent_losers_relative_to_typical_try():
    # 대부분의 시도가 t≈-2 로 지는 상황: 절대 기준(매번 ≤ -2)에는 한 번(-1.7) 때문에 안 걸리지만
    # 10번 넘게 한 번도 못 이겼고 중앙값이 전형적인 시도보다 0.75 이상 낮으면 거른다
    other = [({"params": {"subsample": v, "colsample_bytree": c}, "neg_rate": n}, -1.9)
             for v in (0.6, 1.0) for c in (0.55, 0.85) for n in (0.03, 0.05, 0.2, 0.3)]
    spac = [({"exclude": ["spac"], "neg_rate": n, "params": {"num_leaves": nl}}, t)
            for (n, nl), t in zip([(0.03, 15), (0.05, 15), (0.2, 15), (0.3, 15), (0.03, 31), (0.05, 31), (0.2, 31),
                                   (0.3, 31), (0.03, 127), (0.05, 127)],
                                  [-3.5, -3.2, -3.0, -3.1, -2.9, -3.3, -1.7, -3.6, -3.0, -3.4])]
    st = _state([(f"r{i}", ch, t) for i, (ch, t) in enumerate(other + spac)])
    assert "spac 제외" in evo.bad_moves(st)
    assert "subsample=0.6" not in evo.bad_moves(st)  # 전형적인 수준(-2)으로 진 변경은 남김
    # 한 번이라도 챔피언을 넘었으면(t > 0) 남김
    spac[6] = (spac[6][0], 0.4)
    st = _state([(f"r{i}", ch, t) for i, (ch, t) in enumerate(other + spac)])
    assert "spac 제외" not in evo.bad_moves(st)


def test_next_candidate_mostly_skips_bad_moves():
    st = _state([("hardneg", {"hard_neg": True}, -4.0),
                 ("hard_neg=True & -peer", {"hard_neg": True, "groups": CHAMP["groups"][:-1]}, -4.0),
                 ("hard_neg=True & neg=0.3", {"hard_neg": True, "neg_rate": 0.3}, -4.0)])
    ev = evo.Evolver.__new__(evo.Evolver)
    ev.rng = random.Random(1)
    cands = [ev.next_candidate(st) for _ in range(2000)]
    share = sum(cfg.get("hard_neg", False) for _, cfg in cands) / len(cands)
    # hard_neg 단독은 이미 평가돼 섞인 변이만 남음: 건너뛰지 않으면 약 7~9%, BAD_SKIP(95%)만큼 건너뛰면 약 0.5% — 완전히 막지는 않음
    assert 0 < share < 0.03


def test_merge_strips_rank_params_against_rank_defaults():
    # rank 기본 learning_rate 는 0.05, lgbm 기본은 0.03 → rank 에서 0.03 은 실제 변경이라 남아야 함
    cfg = _merge(CHAMP, {"kind": "rank", "params": {}})
    assert _merge(cfg, {"params": {"learning_rate": 0.03}})["params"] == {"learning_rate": 0.03}
    assert _merge(cfg, {"params": {"learning_rate": 0.05}})["params"] is None
    assert _merge(CHAMP, {"params": {"learning_rate": 0.03}})["params"] == {"num_leaves": 63}
    # 기본값만 남아 params 가 None 인 챔피언에도 파라미터 변경을 얹을 수 있다
    none_champ = _merge(CHAMP, {"params": {"num_leaves": 31, "learning_rate": 0.03}})
    assert none_champ["params"] is None
    assert _merge(none_champ, {"params": {"subsample": 0.6}})["params"] == {"subsample": 0.6}
