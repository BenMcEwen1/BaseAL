"""
Tests for core/utils/sampling.py (ValidationStrategy).

Mirrors tests/test_sampling.py's structure for SamplingStrategy -- small,
hand-constructed inputs for the class-level unit tests (exact ground truth by
construction), plus a real-ESC10 smoke test for integration-level coverage.
ActiveLearner-level plumbing (sample_validation()/add_validation_samples(),
the leakage firewall, pool disjointness) is covered separately in
tests/test_active_learner.py.
"""
import numpy as np
import pytest

from core.utils.sampling import ValidationStrategy


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        ValidationStrategy(method="not_a_real_method")


def test_random_selects_requested_count_from_candidates():
    strat = ValidationStrategy(method="random", n_samples=5, random_state=0)
    candidates = list(range(20))
    selected, utility = strat.select(validation_candidate_indices=candidates)
    assert len(selected) == 5
    assert set(selected).issubset(set(candidates))
    assert len(utility) == len(candidates)


def test_select_clamps_n_samples_to_available_candidates():
    strat = ValidationStrategy(method="random", n_samples=100, random_state=0)
    candidates = list(range(3))
    selected, utility = strat.select(validation_candidate_indices=candidates)
    assert len(selected) == 3
    assert len(utility) == 3


def test_select_with_empty_candidates_returns_empty():
    strat = ValidationStrategy(method="random", n_samples=5)
    selected, utility = strat.select(validation_candidate_indices=[])
    assert selected == []
    assert len(utility) == 0


def test_select_raises_if_method_returns_wrong_length():
    strat = ValidationStrategy(method="custom", n_samples=2)
    strat._method_map["custom"] = lambda: np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        strat.select(validation_candidate_indices=[0, 1])


def test_custom_falls_back_to_random():
    strat = ValidationStrategy(method="custom", n_samples=3, random_state=0)
    selected, utility = strat.select(validation_candidate_indices=list(range(10)))
    assert len(selected) == 3
    assert len(utility) == 10


def test_select_never_receives_labels_for_its_own_candidates():
    """
    Direct test of the boundary this class exists to enforce: `select()`
    accepts labels/validation_labels for already-revealed pools only, never
    for the candidates it is currently choosing among. This is enforced by
    construction (callers must not pass candidate-indexed labels) -- this
    test locks in that `select()` doesn't itself assume/require any
    candidate-label information to function, and that whatever is passed for
    already-revealed pools is exactly sized to those pools, not the
    candidate pool.
    """
    strat = ValidationStrategy(method="random", n_samples=2, random_state=0)
    candidates = [10, 11, 12, 13]
    revealed_validation_indices = [0, 1]
    revealed_validation_labels = np.array([1, 0])  # aligned with revealed_validation_indices only

    selected, utility = strat.select(
        validation_candidate_indices=candidates,
        validation_labeled_indices=revealed_validation_indices,
        validation_labels=revealed_validation_labels,
    )
    assert len(selected) == 2
    # select() stored exactly what was passed -- no broader label array ever
    # touched the candidate indices.
    assert strat.validation_labeled_indices == revealed_validation_indices
    assert len(strat.validation_labels) == len(revealed_validation_indices)
    assert set(selected).issubset(set(candidates))


# ---------------------------------------------------------------------------
# Real-data smoke test
# ---------------------------------------------------------------------------

def test_validation_strategy_smoke_on_real_esc10_embeddings(real_esc10):
    rng = np.random.default_rng(0)
    n = real_esc10["n_samples"]
    candidates = rng.choice(n, size=50, replace=False).tolist()

    strat = ValidationStrategy(method="random", n_samples=10, random_state=0)
    selected, utility = strat.select(validation_candidate_indices=candidates)
    assert len(selected) == 10
    assert set(selected).issubset(set(candidates))
    assert len(utility) == 50
