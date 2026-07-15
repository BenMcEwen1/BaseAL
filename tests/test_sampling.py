"""
Tests for core/utils/sampling.py (SamplingStrategy, WarmupStrategy).

Most tests below use small hand-constructed synthetic inputs on purpose --
they check exact algorithmic invariants (e.g. "the most ambiguous prediction
gets picked first") that require known ground truth by construction, which
real data can't give deterministically. A separate real-data smoke-test
section at the bottom of this file exercises every strategy against the real
ESC10 embeddings (1024-d, realistic distribution) to catch integration-level
issues (e.g. FAISS/HNSW behaviour at real scale) the synthetic tests can't.
"""
import numpy as np
import pytest

from core.active_learner import ActiveLearner
from core.utils.sampling import SamplingStrategy, WarmupStrategy, densityEstimation


# ---------------------------------------------------------------------------
# SamplingStrategy: construction / dispatch
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    with pytest.raises(ValueError):
        SamplingStrategy(method="not_a_real_method")


def test_random_selects_requested_count_from_unlabeled():
    strat = SamplingStrategy(method="random", n_samples=5, random_state=0)
    unlabeled = list(range(20))
    selected, utility = strat.select(unlabeled_indices=unlabeled)
    assert len(selected) == 5
    assert set(selected).issubset(set(unlabeled))
    assert len(utility) == len(unlabeled)


def test_select_clamps_n_samples_to_available_unlabeled():
    strat = SamplingStrategy(method="random", n_samples=100, random_state=0)
    unlabeled = list(range(3))
    selected, utility = strat.select(unlabeled_indices=unlabeled)
    assert len(selected) == 3
    assert len(utility) == 3


def test_select_with_empty_unlabeled_returns_empty():
    strat = SamplingStrategy(method="random", n_samples=5)
    selected, utility = strat.select(unlabeled_indices=[])
    assert selected == []
    assert len(utility) == 0


def test_select_raises_if_method_returns_wrong_length():
    strat = SamplingStrategy(method="custom", n_samples=2)
    # Monkeypatch the dispatched function to return a mismatched-length array.
    strat._method_map["custom"] = lambda: np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1])


def test_custom_falls_back_to_random():
    strat = SamplingStrategy(method="custom", n_samples=3, random_state=0)
    selected, utility = strat.select(unlabeled_indices=list(range(10)))
    assert len(selected) == 3
    assert len(utility) == 10


# ---------------------------------------------------------------------------
# margin / margin_multilabel
# ---------------------------------------------------------------------------

def test_margin_requires_predictions():
    strat = SamplingStrategy(method="margin", n_samples=2)
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1])


def test_margin_prefers_most_ambiguous_prediction():
    # Sample 0: near-certain (large margin) -> low utility.
    # Sample 1: perfectly ambiguous (zero margin) -> high utility.
    predictions = np.array([
        [0.99, 0.01],
        [0.50, 0.50],
    ])
    strat = SamplingStrategy(method="margin", n_samples=1)
    selected, utility = strat.select(unlabeled_indices=[0, 1], predictions=predictions)
    assert selected == [1]
    assert utility[1] > utility[0]


def test_margin_multilabel_prefers_boundary_predictions():
    # Sample 0: every label prob far from 0.5 (confident) -> low utility.
    # Sample 1: every label prob exactly 0.5 (max ambiguity) -> high utility.
    predictions = np.array([
        [0.95, 0.05, 0.9],
        [0.5, 0.5, 0.5],
    ])
    strat = SamplingStrategy(method="margin_multilabel", n_samples=1)
    selected, utility = strat.select(unlabeled_indices=[0, 1], predictions=predictions)
    assert selected == [1]
    assert utility[1] == pytest.approx(1.0)
    assert utility[0] < utility[1]


# ---------------------------------------------------------------------------
# BALD (MC-dropout)
# ---------------------------------------------------------------------------

def test_bald_requires_mc_predictions():
    strat = SamplingStrategy(method="bald", n_samples=2)
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1])


def test_bald_prefers_high_disagreement_between_passes():
    # Sample 0: every MC pass agrees (confident + consistent) -> low BALD score.
    # Sample 1: passes flip-flop between classes (high epistemic uncertainty).
    agree = np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01]])
    disagree = np.array([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]])
    mc_predictions = np.stack([agree, disagree], axis=1)  # (passes, samples, classes)

    strat = SamplingStrategy(method="bald", n_samples=1)
    selected, utility = strat.select(
        unlabeled_indices=[0, 1],
        predictions=mc_predictions.mean(axis=0),
        mc_predictions=mc_predictions,
    )
    assert selected == [1]
    assert utility[1] > utility[0]


# ---------------------------------------------------------------------------
# coreset_farthest / nn_disagreement (embedding-based, FAISS-backed)
# ---------------------------------------------------------------------------

def test_coreset_farthest_requires_embeddings():
    strat = SamplingStrategy(method="coreset_farthest", n_samples=2)
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1], labeled_indices=[2])


def test_coreset_farthest_falls_back_to_random_with_no_labeled():
    embeddings = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    strat = SamplingStrategy(method="coreset_farthest", n_samples=3, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=list(range(10)), embeddings=embeddings, labeled_indices=[],
    )
    assert len(selected) == 3
    assert len(utility) == 10


def test_coreset_farthest_prefers_point_far_from_labeled_anchor():
    # Anchor cluster near the origin; one unlabeled point is far away. The
    # greedy k-center seed is randomised, so a single-sample (k=1) draw isn't
    # guaranteed to land on the outlier -- but with k=2, the outlier is so
    # much farther than every other candidate that it is picked on *either*
    # the seed step or the following greedy-farthest step, regardless of
    # which near point the random seed lands on. This makes membership (not
    # exact index equality) a stable invariant to test against.
    rng = np.random.default_rng(0)
    near = rng.normal(scale=0.1, size=(8, 4)).astype(np.float32)
    far_point = np.array([[50.0, 50.0, 50.0, 50.0]], dtype=np.float32)
    embeddings = np.vstack([near, far_point])

    labeled = [0, 1]  # two of the "near" points
    unlabeled = list(range(2, 9))  # remaining near points + the far point (idx 8)

    strat = SamplingStrategy(method="coreset_farthest", n_samples=2, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=unlabeled, embeddings=embeddings, labeled_indices=labeled,
    )
    assert len(selected) == 2
    assert 8 in selected


def test_nn_disagreement_requires_embeddings_and_predictions():
    strat = SamplingStrategy(method="nn_disagreement", n_samples=2)
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1])


def test_nn_disagreement_falls_back_to_random_without_labels():
    embeddings = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    predictions = np.tile([0.5, 0.5], (10, 1))
    strat = SamplingStrategy(method="nn_disagreement", n_samples=3, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=list(range(10)),
        embeddings=embeddings,
        predictions=predictions,
        labeled_indices=[],
        labels=None,
    )
    assert len(selected) == 3


def test_nn_disagreement_smoke_with_full_context():
    rng = np.random.default_rng(0)
    n = 20
    embeddings = rng.normal(size=(n, 6)).astype(np.float32)
    predictions = rng.random(size=(n, 2)).astype(np.float32)
    labeled = list(range(0, 8))
    unlabeled = list(range(8, n))
    labels = np.array([i % 2 for i in labeled])  # aligned with sorted(labeled)

    strat = SamplingStrategy(method="nn_disagreement", n_samples=4, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=unlabeled,
        embeddings=embeddings,
        predictions=predictions,
        labeled_indices=labeled,
        labels=labels,
    )
    assert len(selected) == 4
    assert set(selected).issubset(set(unlabeled))
    assert len(utility) == len(unlabeled)


# ---------------------------------------------------------------------------
# scikit-activeml-backed strategies (sklearn_coreset / sklearn_typiclust)
# ---------------------------------------------------------------------------

def test_sklearn_coreset_requires_embeddings():
    strat = SamplingStrategy(method="sklearn_coreset", n_samples=2)
    with pytest.raises(ValueError):
        strat.select(unlabeled_indices=[0, 1], labeled_indices=[2])


def test_sklearn_coreset_falls_back_to_random_with_no_labeled():
    embeddings = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
    strat = SamplingStrategy(method="sklearn_coreset", n_samples=3, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=list(range(10)), embeddings=embeddings, labeled_indices=[],
    )
    assert len(selected) == 3


def test_sklearn_coreset_smoke_with_labeled_anchor():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 4)).astype(np.float32)
    labeled = [0, 1, 2]
    unlabeled = list(range(3, 20))

    strat = SamplingStrategy(method="sklearn_coreset", n_samples=4, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=unlabeled, embeddings=embeddings, labeled_indices=labeled,
    )
    assert len(selected) == 4
    assert set(selected).issubset(set(unlabeled))


def test_sklearn_typiclust_smoke():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 4)).astype(np.float32)
    labeled = [0, 1, 2]
    unlabeled = list(range(3, 20))

    strat = SamplingStrategy(method="sklearn_typiclust", n_samples=4, random_state=0)
    selected, utility = strat.select(
        unlabeled_indices=unlabeled, embeddings=embeddings, labeled_indices=labeled,
    )
    assert len(selected) == 4
    assert set(selected).issubset(set(unlabeled))


# ---------------------------------------------------------------------------
# WarmupStrategy
# ---------------------------------------------------------------------------

def test_warmup_unknown_method_raises():
    with pytest.raises(ValueError):
        WarmupStrategy(method="not_a_real_method")


def test_warmup_random_selects_n_samples():
    embeddings = np.random.default_rng(0).normal(size=(15, 4)).astype(np.float32)
    warmup = WarmupStrategy(method="random", n_samples=5, random_state=0)
    selected = warmup.select(candidate_indices=np.arange(15), embeddings=embeddings)
    assert len(selected) == 5
    assert set(selected).issubset(set(range(15)))


def test_warmup_density_prefers_points_in_dense_region():
    # `densityEstimation(..., method='knn')` inverts k-NN distance to a
    # proximity score (see core/utils/sampling.py), so a point surrounded by
    # many close neighbours scores higher than an isolated outlier -- density
    # warmup should therefore prefer points inside the cluster.
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.05, size=(9, 3)).astype(np.float32)
    outlier = np.array([[20.0, 20.0, 20.0]], dtype=np.float32)
    embeddings = np.vstack([cluster, outlier])

    warmup = WarmupStrategy(method="density", n_samples=1, random_state=0)
    selected = warmup.select(candidate_indices=np.arange(10), embeddings=embeddings)
    assert len(selected) == 1
    assert selected[0] != 9  # should not pick the outlier


def test_densityestimation_knn_scores_dense_point_above_outlier():
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.05, size=(200, 16)).astype(np.float32)
    outlier = np.array([[20.0] * 16], dtype=np.float32)
    embeddings = np.vstack([cluster, outlier])

    density = densityEstimation(embeddings=embeddings, method="knn", k=20, beta=1)
    assert density[200] < density[:200].min()


def test_warmup_select_returns_empty_when_n_samples_zero():
    embeddings = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    warmup = WarmupStrategy(method="random", n_samples=0)
    selected = warmup.select(candidate_indices=np.arange(5), embeddings=embeddings)
    assert selected == []


def test_warmup_custom_falls_back_to_density():
    embeddings = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    warmup = WarmupStrategy(method="custom", n_samples=3, random_state=0)
    selected = warmup.select(candidate_indices=np.arange(10), embeddings=embeddings)
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# Real-data integration smoke tests (real ESC10 embeddings, 1024-d)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_sampling_context(real_esc10):
    """
    A partially warmed-up ActiveLearner on the real ESC10 embeddings, reused
    across the parametrized smoke tests below so the (real, ~1s) model/data
    setup only happens once per test session.
    """
    learner = ActiveLearner(
        embeddings_dir=real_esc10["embeddings_dir"],
        annotations_path=real_esc10["labels_path"],
        model_name=real_esc10["model_name"],
        dataset_name="ESC10",
        hidden_dim=8,
        sampling_strategy="random",
        n_samples_per_iteration=5,
        device="cpu",
        verbose=False,
        pretrain_samples=20,
        warmup_strategy="random",
        warmup_epochs=1,
    )
    predictions = learner._predict_all()
    unlabeled_list = sorted(learner.unlabeled_indices)
    labeled_list = sorted(learner.labeled_indices)
    return {
        "embeddings": learner.embeddings,
        "predictions": predictions,
        "unlabeled_indices": unlabeled_list,
        "labeled_indices": labeled_list,
        "labels": learner.labels[labeled_list],
    }


@pytest.mark.parametrize("method", [
    "random", "margin", "bald", "coreset_farthest", "nn_disagreement",
    "sklearn_coreset", "sklearn_typiclust",
])
def test_sampling_strategy_smoke_on_real_esc10_embeddings(real_sampling_context, method):
    ctx = real_sampling_context
    strat = SamplingStrategy(method=method, n_samples=8, random_state=0)
    kwargs = dict(
        unlabeled_indices=ctx["unlabeled_indices"],
        predictions=ctx["predictions"],
        embeddings=ctx["embeddings"],
        labeled_indices=ctx["labeled_indices"],
        labels=ctx["labels"],
    )
    if method == "bald":
        # BALD needs repeated MC-dropout passes; fabricate 3 near-identical
        # passes by jittering the single-pass predictions slightly. This is
        # a smoke test only -- no claim is made about selection quality.
        rng = np.random.default_rng(0)
        base = ctx["predictions"]
        mc = np.clip(
            np.stack([base, base, base])
            + rng.normal(scale=0.01, size=(3,) + base.shape).astype(np.float32),
            0, 1,
        )
        kwargs["mc_predictions"] = mc

    selected, utility = strat.select(**kwargs)
    assert len(selected) == 8
    assert set(selected).issubset(set(ctx["unlabeled_indices"]))
    assert len(utility) == len(ctx["unlabeled_indices"])


def test_warmup_density_selects_central_points_on_real_esc10_embeddings(real_esc10):
    embeddings = np.load(real_esc10["embeddings_dir"] / ".cache" / "embeddings.npy")
    n = len(embeddings)
    warmup = WarmupStrategy(method="density", n_samples=40, random_state=0)
    selected = np.array(warmup.select(candidate_indices=np.arange(n), embeddings=embeddings))
    assert len(selected) == 40
    assert len(set(selected.tolist())) == 40

    # Regression check for the density-inversion fix: selected samples should
    # sit much closer to the dataset centroid than an unbiased random sample.
    centroid = embeddings.mean(axis=0)
    dist_to_centroid = np.linalg.norm(embeddings - centroid, axis=1)
    rng = np.random.default_rng(1)
    random_sample = rng.choice(n, size=40, replace=False)
    assert dist_to_centroid[selected].mean() < dist_to_centroid[random_sample].mean()
