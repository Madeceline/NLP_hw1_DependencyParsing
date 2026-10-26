import itertools

import numpy as np
import pytest

from src.decoder import decode_mst, validate_tree


def brute_force(scores):
    best = -np.inf
    for choices in itertools.product(range(len(scores)), repeat=len(scores) - 1):
        heads = [-1] + list(choices)
        try:
            validate_tree(heads)
        except ValueError:
            continue
        score = sum(scores[i, heads[i]] for i in range(1, len(heads)))
        best = max(best, score)
    return best


def test_single_word():
    assert decode_mst([[0, 0], [7, 200]]).tolist() == [-1, 0]


def test_cycle_and_multiple_root_candidates():
    scores = np.array([[0, 0, 0, 0], [4, 0, 9, 1], [4, 9, 0, 1], [8, 1, 1, 0]], float)
    heads = decode_mst(scores)
    assert validate_tree(heads)
    assert sum(scores[i, heads[i]] for i in range(1, 4)) == brute_force(scores)


def test_nonprojective_tree():
    # 在词序上，边 1→3 与 2→4 相互交叉，构成非投射结构。
    expected = [-1, 0, 1, 1, 2]
    scores = np.zeros((5, 5))
    for dep, head in enumerate(expected[1:], 1):
        scores[dep, head] = 10
    assert decode_mst(scores).tolist() == expected


@pytest.mark.parametrize("size", [2, 3, 4, 5, 6])
def test_random_graphs_against_exhaustive_search(size):
    rng = np.random.default_rng(42 + size)
    for trial in range(20):
        scores = rng.normal(size=(size, size))
        if trial % 2 == 0:
            scores = np.round(scores)  # 同时覆盖存在并列最优解的情况
        if trial % 3 == 0:
            scores[rng.random(scores.shape) < 0.2] = -np.inf
        optimum = brute_force(scores)
        if not np.isfinite(optimum):
            with pytest.raises(ValueError):
                decode_mst(scores)
        else:
            heads = decode_mst(scores)
            assert validate_tree(heads)
            actual = sum(scores[i, heads[i]] for i in range(1, size))
            assert actual == pytest.approx(optimum, abs=1e-9)


def test_infeasible_and_invalid_inputs():
    with pytest.raises(ValueError):
        decode_mst(np.full((3, 3), -np.inf))
    with pytest.raises(ValueError):
        decode_mst(np.array([[0, 0], [np.nan, 0]]))
