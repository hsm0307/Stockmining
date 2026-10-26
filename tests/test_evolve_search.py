"""자동 개선 루프의 무작위 변이: 같은 변경을 두 번 고르지 않고, 번번이 나빴던 변경은 대체로 건너뛴다."""
import random

from krxlimit import evolve as evo

CHAMP = {"groups": ["candle", "momentum", "limit_hist", "volume"], "exclude": [], "kind": "bag3",
         "train_years": 3, "neg_rate": 0.1, "hard_neg": False, "params": None}


def _state(results):
    """results: [(label, t)] — 모두 현 챔피언('c0') 대비 시도."""
    tested = {f"k{i}": {"label": lab, "t": t, "base": "c0"} for i, (lab, t) in enumerate(results)}
    tested["c0"] = {"label": "v5(출발점)", "t": 0.0}
    return {"dev_champion": {"key": "c0", "cfg": CHAMP}, "tested": tested, "queue_pos": 10 ** 6}


def test_mutate_never_repeats_a_single_choice_change():
    rng = random.Random(0)
    for _ in range(3000):
        name, change = evo.mutate(CHAMP, rng)
        parts = name.split(" & ")
        assert len(parts) == len(set(parts)), name
        keys = [p.split("=")[0] for p in parts if "=" in p]
        assert len(keys) == len(set(keys)), name  # kind=rank & kind=lgbm, num_leaves 두 번 등 금지


def test_bad_moves_needs_repeated_clear_losses_against_current_champion():
    st = _state([("hard_neg=True", -3.9), ("hard_neg=True & -peer", -4.0), ("num_leaves=63 & hard_neg=True", -2.3),
                 ("kind=ens", -1.6), ("kind=ens & neg=0.2", -2.5), ("kind=ens & -volume", -2.4),
                 ("years=5", -3.0), ("years=5 & +peer2", -2.8)])
    st["tested"]["old"] = {"label": "years=5", "t": -3.0, "base": "older_champion"}  # 이전 챔피언 대비 기록은 세지 않음
    assert evo.bad_moves(st) == {"hard_neg=True"}  # kind=ens 는 한 번 -1.6, years=5 는 현 챔피언 대비 2번뿐


def test_next_candidate_mostly_skips_bad_moves():
    st = _state([(lab, -4.0) for lab in ["hard_neg=True", "hard_neg=True & -peer", "hard_neg=True & +market2"]])
    ev = evo.Evolver.__new__(evo.Evolver)
    ev.rng = random.Random(1)
    names = [ev.next_candidate(st)[0] for _ in range(2000)]
    share = sum("hard_neg=True" in n for n in names) / len(names)
    # 건너뛰지 않으면 약 20%(8개 연산 중 1개, 가끔 두 번 뽑음), 85% 건너뛰면 약 3~4% — 완전히 막지는 않음
    assert 0 < share < 0.08
