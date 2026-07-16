"""
Tests for core/active_learner.py (ActiveLearner).

Uses the real ESC10 demo dataset shipped in-repo (`real_esc10` fixture:
800 samples, 10 classes, 1024-d BirdNet embeddings, real 640/160
train/validation split) wherever it's usable as-is. Multilabel coverage
still uses synthetic data since ESC10 is single-label only.
"""
import json

import numpy as np
import pytest
import torch
import yaml

from core.active_learner import ActiveLearner


def build_learner(dataset, **overrides):
    kwargs = dict(
        embeddings_dir=dataset["embeddings_dir"],
        annotations_path=dataset["labels_path"],
        model_name=dataset["model_name"],
        dataset_name="ESC10",
        hidden_dim=8,
        learning_rate=0.01,
        sampling_strategy="random",
        n_samples_per_iteration=5,
        device="cpu",
        verbose=False,
    )
    kwargs.update(overrides)
    return ActiveLearner(**kwargs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_load_data_basic_shapes(real_esc10):
    learner = build_learner(real_esc10)
    assert learner.embeddings.shape == (real_esc10["n_samples"], real_esc10["embedding_dim"])
    assert learner.is_multilabel is False
    assert len(learner.label_to_idx) == len(real_esc10["class_names"])
    assert learner.labeled_indices == set()
    # ESC10's labels.csv carries a real validation split, so unlabeled excludes it.
    assert len(learner.unlabeled_indices) == real_esc10["n_samples"] - real_esc10["n_validation"]


def test_multilabel_detection_and_criterion(multilabel_dataset):
    learner = build_learner(multilabel_dataset, sampling_strategy="margin_multilabel")
    assert learner.is_multilabel is True
    assert learner.labels.shape == (multilabel_dataset["n_samples"], len(multilabel_dataset["class_names"]))
    assert isinstance(learner.criterion, torch.nn.BCEWithLogitsLoss)


def test_single_label_criterion(real_esc10):
    learner = build_learner(real_esc10)
    assert isinstance(learner.criterion, torch.nn.CrossEntropyLoss)


def test_validation_samples_are_held_out_of_both_pools(real_esc10):
    learner = build_learner(real_esc10)
    assert len(learner.validation_indices) == real_esc10["n_validation"]
    assert learner.validation_indices.isdisjoint(learner.unlabeled_indices)
    assert learner.validation_indices.isdisjoint(learner.labeled_indices)
    total_pool = len(learner.labeled_indices) + len(learner.unlabeled_indices) + len(learner.validation_indices)
    assert total_pool == real_esc10["n_samples"]


def test_embedding_cache_exists_and_matches_after_load(real_esc10):
    build_learner(real_esc10)
    cache_dir = real_esc10["embeddings_dir"] / ".cache"
    assert (cache_dir / "manifest.json").exists()
    assert (cache_dir / "embeddings.npy").exists()

    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert len(manifest["filenames"]) == real_esc10["n_samples"]


def test_second_load_hits_cache_and_matches_first(real_esc10):
    first = build_learner(real_esc10)
    second = build_learner(real_esc10)
    np.testing.assert_allclose(np.asarray(first.embeddings), np.asarray(second.embeddings))


# ---------------------------------------------------------------------------
# sample() / add_samples()
# ---------------------------------------------------------------------------

def test_sample_returns_subset_of_unlabeled(real_esc10):
    learner = build_learner(real_esc10, n_samples_per_iteration=5)
    selected = learner.sample()
    assert len(selected) == 5
    assert set(selected).issubset(learner.unlabeled_indices)


def test_sample_override_n_samples_restores_default(real_esc10):
    learner = build_learner(real_esc10, n_samples_per_iteration=5)
    selected = learner.sample(n_samples=3)
    assert len(selected) == 3
    assert learner.sampling_strategy.n_samples == 5  # restored after override


def test_sample_on_fully_labeled_pool_returns_empty(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(list(learner.unlabeled_indices))
    assert learner.sample() == []


def test_add_samples_moves_indices_and_tracks_annotation_cost_single_label(real_esc10):
    learner = build_learner(real_esc10)
    selected = learner.sample(n_samples=4)
    learner.add_samples(selected)
    assert learner.labeled_indices == set(selected)
    assert len(learner.unlabeled_indices) == (real_esc10["n_samples"] - real_esc10["n_validation"]) - 4
    assert learner._pending_annotation_cost == 4  # one label event per sample


def test_add_samples_annotation_cost_counts_positive_labels_for_multilabel(multilabel_dataset):
    learner = build_learner(multilabel_dataset, sampling_strategy="margin_multilabel")
    selected = learner.sample(n_samples=6)
    learner.add_samples(selected)
    expected_cost = int(learner.labels[selected].sum())
    assert learner._pending_annotation_cost == expected_cost
    assert expected_cost >= 6  # some samples have 2 positive labels


# ---------------------------------------------------------------------------
# train_step()
# ---------------------------------------------------------------------------

def test_train_step_with_no_labeled_samples_returns_zero_metrics(real_esc10):
    learner = build_learner(real_esc10)
    metrics = learner.train_step(epochs=1, batch_size=8)
    assert metrics["loss"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert learner.training_history == []


def test_train_step_returns_expected_metric_keys(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=10))
    metrics = learner.train_step(epochs=2, batch_size=4)

    expected_keys = {
        "loss", "accuracy", "f1_score", "mAP",
        "loss_sd", "accuracy_sd", "f1_score_sd", "mAP_sd",
        "n_labeled", "n_unlabeled", "repeats", "epochs", "batch_size",
        "sampling_time_s", "annotation_cost", "calibration",
        "aulc_mAP", "aulc_accuracy", "aulc_f1_score",
    }
    assert expected_keys.issubset(metrics.keys())
    assert metrics["n_labeled"] == 10
    assert metrics["annotation_cost"] == 10
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(learner.training_history) == 1


def test_train_step_calibration_payload_shape(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=15))
    metrics = learner.train_step(epochs=2, batch_size=4)
    calibration = metrics["calibration"]
    assert calibration["n_bins"] == 10
    assert len(calibration["bin_confidences"]) == 10
    assert len(calibration["bin_accuracies"]) == 10
    assert len(calibration["bin_counts"]) == 10
    assert isinstance(calibration["ece"], float)


def test_train_step_per_class_ap_stays_internal_and_matches_reported_map(real_esc10):
    """
    Chunk 0 regression test: train_step()'s public metrics dict must be
    byte-for-byte unchanged by the internal per-class AP capture -- no new
    key leaks into the returned dict or the export()/YAML schema. The
    internal-only self._last_per_class_ap (reserved for the future
    class-set-matched active-testing truth) should be consistent with the
    publicly reported mAP for a single-repeat run.
    """
    learner = build_learner(real_esc10, repeats=1)
    assert learner._last_per_class_ap == {}  # untouched before any training

    learner.add_samples(learner.sample(n_samples=15))
    metrics = learner.train_step(epochs=2, batch_size=4)

    # No new key leaked into the public metrics dict.
    assert "per_class_ap" not in metrics
    assert "aps_by_class" not in metrics

    # Internal state is populated and consistent with the reported mAP
    # (exact equality holds only because repeats=1 -- mAP is otherwise a
    # mean across repeats while _last_per_class_ap is last-repeat-only, the
    # same convention already used for calibration data).
    assert learner._last_per_class_ap
    assert all(0.0 <= ap <= 1.0 for ap in learner._last_per_class_ap.values())
    assert np.mean(list(learner._last_per_class_ap.values())) == pytest.approx(metrics["mAP"], abs=1e-5)


def test_train_step_resets_pending_supplementary_metrics(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=5))
    learner.train_step(epochs=1, batch_size=4)
    assert learner._pending_sampling_time == 0.0
    assert learner._pending_annotation_cost == 0


def test_train_step_uses_validation_set_when_present(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=10))
    learner.train_step(epochs=1, batch_size=4)
    # Validation samples are never trained on or sampled.
    assert learner.validation_indices.isdisjoint(learner.labeled_indices)


# ---------------------------------------------------------------------------
# compute_aulc()
# ---------------------------------------------------------------------------

def test_compute_aulc_returns_zero_with_no_history():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.training_history = []
    assert learner.compute_aulc("mAP") == 0.0


def test_compute_aulc_matches_manual_trapezoid():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.training_history = [
        {"n_labeled": 10, "mAP": 0.2},
        {"n_labeled": 20, "mAP": 0.4},
        {"n_labeled": 30, "mAP": 0.6},
    ]
    expected = float(np.trapz([0, 0.2, 0.4, 0.6], [0, 10, 20, 30]) / 30)
    assert learner.compute_aulc("mAP") == pytest.approx(expected)


def test_compute_aulc_zero_when_x_range_collapses():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.training_history = [{"n_labeled": 0, "mAP": 0.5}]
    assert learner.compute_aulc("mAP") == 0.0


# ---------------------------------------------------------------------------
# Warm-up pretraining
# ---------------------------------------------------------------------------

def test_pretrain_warmup_selects_samples_and_records_cycle_zero(real_esc10):
    learner = build_learner(
        real_esc10, pretrain_samples=8, warmup_strategy="random",
        warmup_epochs=1, warmup_batch_size=4,
    )
    assert len(learner.labeled_indices) == 8
    assert len(learner.training_history) == 1
    assert learner.training_history[0]["warmup"] is True


# ---------------------------------------------------------------------------
# export()
# ---------------------------------------------------------------------------

def test_export_produces_expected_yaml_schema(real_esc10, tmp_path):
    learner = build_learner(real_esc10)
    for _ in range(2):
        learner.add_samples(learner.sample(n_samples=5))
        learner.train_step(epochs=1, batch_size=4)

    output_path = tmp_path / "submission.yaml"
    learner.export(str(output_path), author_lastname="Test", institute_abbreviation="XX", max_budget=100)

    with open(output_path) as f:
        submission = yaml.safe_load(f)

    assert submission["dataset"] == "ESC10"
    assert submission["author_lastname"] == "Test"
    assert len(submission["learning_curve"]) == 2
    assert submission["learning_curve"][0]["cycle"] == 1
    assert submission["learning_curve"][1]["cycle"] == 2

    for row in submission["learning_curve"]:
        for stripped_key in ("calibration", "loss", "loss_sd", "n_unlabeled", "repeats"):
            assert stripped_key not in row

    supplementary = submission["supplementary"]
    assert supplementary["n_cycles"] == 2
    assert supplementary["total_annotation_cost"] == 10
    cost = supplementary["computational_cost"]
    for key in ("model_parameters", "epochs_per_cycle", "n_cycles", "cost_method", "baseline_n_cycles", "baseline_cost", "relative_cost"):
        assert key in cost


def test_export_marks_warmup_as_cycle_zero(real_esc10, tmp_path):
    learner = build_learner(
        real_esc10, pretrain_samples=6, warmup_strategy="random", warmup_epochs=1,
    )
    learner.add_samples(learner.sample(n_samples=5))
    learner.train_step(epochs=1, batch_size=4)

    output_path = tmp_path / "submission.yaml"
    learner.export(str(output_path))
    with open(output_path) as f:
        submission = yaml.safe_load(f)

    cycles = [row["cycle"] for row in submission["learning_curve"]]
    assert cycles == [0, 1]
    assert submission["learning_curve"][0]["warmup"] is True


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

def test_get_state_keys_and_uncertainty_length(real_esc10):
    learner = build_learner(real_esc10)
    learner.sample(n_samples=5)  # populates self.uncertainties as a side effect
    state = learner.get_state()

    expected_keys = {
        "n_labeled", "n_unlabeled", "labeled_indices", "unlabeled_indices",
        "training_history", "num_classes", "labels", "uncertainties", "is_multilabel",
    }
    assert expected_keys.issubset(state.keys())
    assert len(state["uncertainties"]) == real_esc10["n_samples"]
    assert state["num_classes"] == len(real_esc10["class_names"])


# ---------------------------------------------------------------------------
# get_embeddings_3d() / geometric projections
# ---------------------------------------------------------------------------

def test_get_embeddings_3d_invalid_projection_raises(real_esc10):
    learner = build_learner(real_esc10)
    with pytest.raises(ValueError):
        learner.get_embeddings_3d(projection="not_a_real_projection")


def test_get_embeddings_3d_euclidean_shape(real_esc10):
    learner = build_learner(real_esc10)
    coords = learner.get_embeddings_3d(projection="euclidean", max_embeddings=200)
    assert coords.shape == (200, 3)


def test_get_embeddings_3d_spherical_projection_has_unit_norm(real_esc10):
    learner = build_learner(real_esc10)
    coords = learner.get_embeddings_3d(projection="spherical", max_embeddings=200)
    norms = np.linalg.norm(coords, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_get_embeddings_3d_respects_max_embeddings_subsampling(real_esc10):
    learner = build_learner(real_esc10)
    coords = learner.get_embeddings_3d(max_embeddings=10)
    assert coords.shape == (10, 3)


# ---------------------------------------------------------------------------
# sample_validation() / add_validation_samples() -- "active testing" v2, chunk 1
# ---------------------------------------------------------------------------

def test_sample_validation_returns_subset_of_validation_pool(real_esc10):
    learner = build_learner(real_esc10, n_validation_samples_per_iteration=5)
    selected = learner.sample_validation()
    assert len(selected) == 5
    assert set(selected).issubset(learner.validation_indices)


def test_sample_validation_override_n_samples_restores_default(real_esc10):
    learner = build_learner(real_esc10, n_validation_samples_per_iteration=5)
    selected = learner.sample_validation(n_samples=3)
    assert len(selected) == 3
    assert learner.validation_strategy.n_samples == 5  # restored after override


def test_sample_validation_on_fully_revealed_pool_returns_empty(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_validation_samples(list(learner.validation_indices))
    assert learner.sample_validation() == []


def test_add_validation_samples_moves_indices_and_tracks_cost(real_esc10):
    learner = build_learner(real_esc10)
    selected = learner.sample_validation(n_samples=7)
    learner.add_validation_samples(selected)
    assert learner.validation_labeled_indices == set(selected)
    assert learner._pending_validation_annotation_cost == 7  # one event per sample, single-label


def test_add_validation_samples_never_touches_training_pools(real_esc10):
    learner = build_learner(real_esc10)
    selected = learner.sample_validation(n_samples=7)
    learner.add_validation_samples(selected)
    assert learner.labeled_indices == set()
    assert len(learner.unlabeled_indices) == real_esc10["n_samples"] - real_esc10["n_validation"]


def test_add_samples_ignores_validation_indices_if_passed(real_esc10):
    # A malformed/adversarial add_samples() call including validation indices
    # must not move them into the training pool -- add_samples() only ever
    # intersects with unlabeled_indices, which excludes validation by
    # construction.
    learner = build_learner(real_esc10)
    some_validation_indices = list(learner.validation_indices)[:5]
    learner.add_samples(some_validation_indices)
    assert learner.labeled_indices.isdisjoint(learner.validation_indices)


def test_disjointness_invariant_across_interleaved_calls(real_esc10):
    """
    Core no-leakage-into-training guarantee: no matter how sample()/
    add_samples()/sample_validation()/add_validation_samples() are
    interleaved, labeled_indices and validation_labeled_indices never
    intersect, and validation_labeled_indices always stays inside
    validation_indices.
    """
    learner = build_learner(real_esc10)

    for _ in range(4):
        learner.add_samples(learner.sample(n_samples=5))
        learner.add_validation_samples(learner.sample_validation(n_samples=3))

        assert learner.labeled_indices.isdisjoint(learner.validation_labeled_indices)
        assert learner.validation_labeled_indices <= learner.validation_indices
        assert learner.labeled_indices.isdisjoint(learner.validation_indices)


def test_validation_strategy_select_never_receives_candidate_labels(real_esc10):
    """
    Leakage-firewall test (highest-severity item from the active-testing
    design review): sample_validation() must never pass labels for the
    validation candidates it's currently choosing among -- only for pools
    already "paid for" (revealed training labels, previously revealed
    validation labels). This is the security boundary the whole mechanism
    depends on: without it, a strategy could peek at unrevealed validation
    labels and cheat the entire budgeted-validation design.
    """
    learner = build_learner(real_esc10)
    # Reveal a few training and validation samples first so both
    # "already-revealed" pools are non-empty -- the interesting case.
    learner.add_samples(learner.sample(n_samples=8))
    learner.add_validation_samples(learner.sample_validation(n_samples=4))

    captured = {}
    original_select = learner.validation_strategy.select

    def spy_select(**kwargs):
        captured.update(kwargs)
        return original_select(**kwargs)

    learner.validation_strategy.select = spy_select
    learner.sample_validation(n_samples=5)

    candidate_set = set(captured["validation_candidate_indices"])

    # (1) The already-revealed validation pool passed in is exactly the
    # participant's own prior reveals -- and disjoint from this call's
    # candidates (a revealed sample can't simultaneously be a candidate).
    assert set(captured["validation_labeled_indices"]) == learner.validation_labeled_indices
    assert candidate_set.isdisjoint(captured["validation_labeled_indices"])

    # (2) validation_labels is sized to the revealed pool only -- never large
    # enough to be indexable by a candidate position.
    assert len(captured["validation_labels"]) == len(captured["validation_labeled_indices"])

    # (3) Training labels passed through are sized to the labeled (training)
    # pool only, same firewall sample() already relies on.
    assert set(captured["labeled_indices"]) == learner.labeled_indices
    assert len(captured["labels"]) == len(captured["labeled_indices"])

    # (4) No candidate index appears in either "already revealed" pool.
    assert candidate_set.isdisjoint(captured["labeled_indices"])


# ---------------------------------------------------------------------------
# compute_scored_class_truth() -- "active testing" v2, chunk 3
# ---------------------------------------------------------------------------

def test_per_class_positive_counts_single_label():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.is_multilabel = False
    learner.label_to_idx = {"a": 0, "b": 1, "c": 2}
    labels_subset = np.array([0, 0, 1, 2, 2, 2])
    counts = learner._per_class_positive_counts(labels_subset)
    assert counts == {0: 2, 1: 1, 2: 3}


def test_per_class_positive_counts_multilabel():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.is_multilabel = True
    learner.label_to_idx = {"a": 0, "b": 1, "c": 2}
    labels_subset = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=np.float32)
    counts = learner._per_class_positive_counts(labels_subset)
    assert counts == {0: 2, 1: 2, 2: 1}


def test_compute_scored_class_truth_unmeasured_before_any_training():
    learner = ActiveLearner.__new__(ActiveLearner)
    learner._last_per_class_ap = {}
    learner.validation_labeled_indices = set()
    learner.min_positives_per_class = 10
    result = learner.compute_scored_class_truth()
    assert result == {
        'mAP_theta_scored': None,
        'n_classes_scored': 0,
        'n_classes_unmeasured': None,  # "haven't looked", not a count of 0
        'scored_class_indices': [],
    }


def test_compute_scored_class_truth_unmeasured_with_zero_validation_budget():
    # Zero validation revealed -- "haven't looked" -- distinct from having
    # revealed validation but zero classes clearing the threshold (see
    # test_compute_scored_class_truth_drops_rare_classes_exactly, where
    # n_classes_unmeasured IS a real count because validation WAS revealed).
    learner = ActiveLearner.__new__(ActiveLearner)
    learner._last_per_class_ap = {0: 0.5, 1: 0.7}
    learner.validation_labeled_indices = set()
    learner.min_positives_per_class = 10
    result = learner.compute_scored_class_truth()
    assert result['mAP_theta_scored'] is None
    assert result['n_classes_scored'] == 0
    assert result['n_classes_unmeasured'] is None


def test_compute_scored_class_truth_drops_rare_classes_exactly(dataset_factory):
    """
    Hand-constructed scenario with a known validation-class distribution:
    dataset_factory(n_samples=60, n_classes=3, with_validation=True) puts
    samples 0-11 in validation, with class = i % 3 (so class 0 = {0,3,6,9},
    class 1 = {1,4,7,10}, class 2 = {2,5,8,11}, 4 each). Revealing only
    [0,3,6,9,1,4] gives class 0 four positives, class 1 two, class 2 zero.
    With min_positives_per_class=3, only class 0 should be scored.
    """
    dataset = dataset_factory(n_samples=60, n_classes=3, with_validation=True)
    learner = build_learner(dataset, min_positives_per_class=3)
    learner.add_samples(learner.sample(n_samples=10))
    learner.train_step(epochs=1, batch_size=4)
    assert set(learner._last_per_class_ap.keys()) == {0, 1, 2}  # all 3 have full-pool support

    learner.add_validation_samples([0, 3, 6, 9, 1, 4])
    result = learner.compute_scored_class_truth()

    assert result['scored_class_indices'] == [0]
    assert result['n_classes_scored'] == 1
    assert result['n_classes_unmeasured'] == 2  # validation WAS revealed; a real count, not None
    assert result['mAP_theta_scored'] == pytest.approx(learner._last_per_class_ap[0])


def test_compute_scored_class_truth_does_not_mutate_state(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=10))
    learner.train_step(epochs=1, batch_size=4)
    learner.add_validation_samples(learner.sample_validation(n_samples=30))

    before_labels = learner.labels.copy()
    before_validation_labeled = set(learner.validation_labeled_indices)
    before_last_ap = dict(learner._last_per_class_ap)
    before_history_len = len(learner.training_history)

    learner.compute_scored_class_truth()
    learner.compute_scored_class_truth(min_positives_per_class=1)  # different arg too

    np.testing.assert_array_equal(learner.labels, before_labels)
    assert learner.validation_labeled_indices == before_validation_labeled
    assert learner._last_per_class_ap == before_last_ap
    assert len(learner.training_history) == before_history_len


def test_compute_scored_class_truth_smoke_on_real_esc10(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=20))
    learner.train_step(epochs=2, batch_size=4)

    # Reveal all real validation samples -- ESC10's 10 classes are balanced
    # (~16 validation samples each), comfortably clearing the default
    # min_positives_per_class=10 threshold for most/all classes.
    learner.add_validation_samples(list(learner.validation_indices))
    result = learner.compute_scored_class_truth()

    assert result['n_classes_scored'] >= 5
    assert result['n_classes_scored'] + result['n_classes_unmeasured'] == len(learner._last_per_class_ap)
    assert 0.0 <= result['mAP_theta_scored'] <= 1.0


# ---------------------------------------------------------------------------
# compute_validation_estimate() -- "active testing" v2, chunk 4
# ---------------------------------------------------------------------------

def test_winkler_score_covered_pays_only_width():
    score, covered = ActiveLearner._winkler_score(lower=0.3, upper=0.5, theta=0.4, alpha=0.10)
    assert covered is True
    assert score == pytest.approx(0.2)  # width only


def test_winkler_score_below_lower_pays_miss_penalty():
    score, covered = ActiveLearner._winkler_score(lower=0.3, upper=0.5, theta=0.2, alpha=0.10)
    assert covered is False
    # width(0.2) + (2/alpha)*(lower-theta) = 0.2 + 20*0.1 = 2.2
    assert score == pytest.approx(2.2)


def test_winkler_score_above_upper_pays_miss_penalty():
    score, covered = ActiveLearner._winkler_score(lower=0.3, upper=0.5, theta=0.6, alpha=0.10)
    assert covered is False
    # width(0.2) + (2/alpha)*(theta-upper) = 0.2 + 20*0.1 = 2.2
    assert score == pytest.approx(2.2)


def test_winkler_score_wide_interval_always_costs_at_least_its_width():
    # "Wide-and-useless must not be cheap": a maximally wide [0, 1] interval
    # pays its full width regardless of coverage.
    score, covered = ActiveLearner._winkler_score(lower=0.0, upper=1.0, theta=0.5, alpha=0.10)
    assert covered is True
    assert score == pytest.approx(1.0)


def test_winkler_score_smaller_alpha_increases_miss_penalty():
    # A miss under a stricter (smaller-alpha) interval is penalised harder.
    score_90, _ = ActiveLearner._winkler_score(lower=0.3, upper=0.5, theta=0.2, alpha=0.10)
    score_95, _ = ActiveLearner._winkler_score(lower=0.3, upper=0.5, theta=0.2, alpha=0.05)
    assert score_95 > score_90


# ---------------------------------------------------------------------------
# _range_compose() -- prevalence-fill for unmeasured classes
# ---------------------------------------------------------------------------

def test_range_compose_matches_worked_example_nothing_scored():
    # Cycle 0 from the prevalence-fill spec: 4 classes, nothing revealed,
    # all four unmeasured. lower = mean(prevalences), upper = mean(1s).
    prevalence = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.05}  # dog, rain, chainsaw, crying_baby
    lower, upper = ActiveLearner._range_compose(
        class_lower_bounds=np.array([]),
        class_upper_bounds=np.array([]),
        scored_set=set(),
        prevalence=prevalence,
        num_classes=4,
    )
    assert lower == pytest.approx(0.20)
    assert upper == pytest.approx(1.00)


def test_range_compose_matches_worked_example_one_class_scored():
    # Cycle 5 from the prevalence-fill spec: only "dog" (class 0) scored,
    # bootstraps to [0.62, 0.78]; the other three stay at [prevalence, 1].
    # The upper bound is NOT 1.0 -- dog's measured ceiling of 0.78 drags it
    # down, which is the whole point of measurement constraining both ends.
    prevalence = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.05}
    lower, upper = ActiveLearner._range_compose(
        class_lower_bounds=np.array([0.62]),
        class_upper_bounds=np.array([0.78]),
        scored_set={0},
        prevalence=prevalence,
        num_classes=4,
    )
    assert lower == pytest.approx(0.28)
    assert upper == pytest.approx(0.945)


def test_range_compose_all_classes_scored_reduces_to_plain_macro_average():
    # Cycle 12 from the spec: all four measured -- the unmeasured machinery
    # switches itself off automatically (no prevalence contribution at all).
    prevalence = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.05}  # irrelevant here
    lowers = np.array([0.9, 0.8, 0.85, 0.95])
    uppers = np.array([0.95, 0.9, 0.92, 0.99])
    lower, upper = ActiveLearner._range_compose(
        class_lower_bounds=lowers,
        class_upper_bounds=uppers,
        scored_set={0, 1, 2, 3},
        prevalence=prevalence,
        num_classes=4,
    )
    assert lower == pytest.approx(float(lowers.mean()))
    assert upper == pytest.approx(float(uppers.mean()))


def test_range_compose_avoids_clt_narrowing_vs_naive_independent_resampling():
    """
    Regression test for the CRITICAL IMPLEMENTATION POINT in the
    prevalence-fill spec: unmeasured classes' [prevalence, 1] ranges must be
    combined ONCE, arithmetically (range composition), not resampled
    independently per bootstrap replicate. Independent resampling lets the
    central limit theorem concentrate the macro-average, manufacturing a
    narrow, confident-looking interval over classes that were never
    observed. This reproduces the naive (wrong) approach via Monte Carlo, on
    the exact cycle-5 worked example, and confirms its interval is
    substantially narrower than the correct range-composed interval over the
    SAME inputs.
    """
    prevalence = {0: 0.30, 1: 0.25, 2: 0.20, 3: 0.05}
    scored_set = {0}
    correct_lower, correct_upper = ActiveLearner._range_compose(
        class_lower_bounds=np.array([0.62]),
        class_upper_bounds=np.array([0.78]),
        scored_set=scored_set,
        prevalence=prevalence,
        num_classes=4,
    )
    correct_width = correct_upper - correct_lower  # 0.665

    # Naive (wrong) approach: draw each unmeasured class independently from
    # its [prevalence, 1] range inside every replicate (plus dog drawn
    # independently from its own bootstrapped range), macro-average, then
    # take a percentile interval -- exactly the mistake the spec warns
    # against.
    rng = np.random.default_rng(0)
    n_replicates = 5000
    dog_draws          = rng.uniform(0.62, 0.78, size=n_replicates)
    rain_draws         = rng.uniform(prevalence[1], 1.0, size=n_replicates)
    chainsaw_draws     = rng.uniform(prevalence[2], 1.0, size=n_replicates)
    crying_baby_draws  = rng.uniform(prevalence[3], 1.0, size=n_replicates)
    naive_macro = (dog_draws + rain_draws + chainsaw_draws + crying_baby_draws) / 4
    naive_lower, naive_upper = np.percentile(naive_macro, [5, 95])
    naive_width = naive_upper - naive_lower

    # The naive interval must be substantially narrower -- the CLT
    # concentrates the average of independent draws around its mean, which
    # range composition (a fixed, worst-case bound) never does. (Empirically
    # ~0.35 vs. the correct 0.665, a ~48% reduction -- 0.6x leaves comfortable
    # margin against simulation noise without being a vacuous check.)
    assert naive_width < correct_width * 0.6


# ---------------------------------------------------------------------------
# theta_coherent / always-defined full-task fields
# ---------------------------------------------------------------------------

def test_theta_coherent_matches_prevalence_fill_formula_on_synthetic_example():
    """
    theta_coherent = real per-class AP for scored classes, prevalence for
    every other class -- computed by hand here and checked against
    compute_validation_estimate()'s value, on a minimal synthetic setup
    (no real training).
    """
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.alpha = 0.10
    learner.n_bootstrap = 20
    learner.min_positives_per_class = 2
    learner.is_multilabel = False
    learner.label_to_idx = {"a": 0, "b": 1}
    learner.mc_dropout_passes = 1
    learner.validation_labeled_indices = {0, 1, 2}
    learner.labels = np.array([0, 0, 1])  # class 0: 2 revealed positives, class 1: 1
    learner._last_per_class_ap = {0: 0.9, 1: 0.4}  # both classes "eligible"
    learner._predict_all = lambda: np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]])

    result = learner.compute_validation_estimate(random_state=0)

    assert result["scored_class_indices"] == [0]  # only class 0 clears threshold=2
    # prevalence: n_revealed=3; class 0 positives=2 -> 2/3, class 1 positives=1 -> 1/3
    expected_theta_coherent = np.mean([0.9, 1 / 3])
    assert result["theta_coherent"] == pytest.approx(expected_theta_coherent)
    assert result["theta_coherent_delta"] == pytest.approx(
        result["mAP_theta_scored"] - expected_theta_coherent
    )


def test_compute_validation_estimate_always_defined_with_nothing_trained_or_revealed():
    """
    Even before any training or validation has happened at all
    (_last_per_class_ap empty, validation_labeled_indices empty), the
    full-task fields must still be real numbers -- prevalence defaults to
    0.0 for every class with zero revealed samples (see
    _class_prevalence()), so lcb=0.0, ucb=1.0, theta_coherent=0.0, and
    winkler/covered are computable. This path must not call _predict_all()
    at all (deliberately left undefined here -- would raise if it were).
    """
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.alpha = 0.10
    learner.n_bootstrap = 20
    learner.min_positives_per_class = 5
    learner.is_multilabel = False
    learner.label_to_idx = {"a": 0, "b": 1, "c": 2}
    learner.mc_dropout_passes = 1
    learner.validation_labeled_indices = set()
    learner.labels = np.array([])
    learner._last_per_class_ap = {}

    result = learner.compute_validation_estimate(random_state=0)

    assert result["mAP_theta_scored"] is None
    assert result["mAP_est"] is None
    assert result["theta_coherent"] == pytest.approx(0.0)
    assert result["theta_coherent_delta"] is None
    assert result["lcb"] == pytest.approx(0.0)
    assert result["ucb"] == pytest.approx(1.0)
    assert result["lcb_zero_fill"] == pytest.approx(0.0)
    assert result["winkler"] is not None
    assert isinstance(result["covered"], bool)


def test_macro_ap_over_classes_matches_direct_sklearn_call():
    from sklearn.metrics import average_precision_score

    learner = ActiveLearner.__new__(ActiveLearner)
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    labels_binary = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)

    result = learner._macro_ap_over_classes(probs, labels_binary, class_indices=[0, 1])
    expected = np.mean([
        average_precision_score(labels_binary[:, 0], probs[:, 0]),
        average_precision_score(labels_binary[:, 1], probs[:, 1]),
    ])
    assert result == pytest.approx(expected)


def test_macro_ap_over_classes_skips_classes_with_no_positives_in_rows():
    learner = ActiveLearner.__new__(ActiveLearner)
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels_binary = np.array([[1, 0], [1, 0]], dtype=np.float32)  # class 1 has zero positives here

    from sklearn.metrics import average_precision_score
    result = learner._macro_ap_over_classes(probs, labels_binary, class_indices=[0, 1])
    expected = average_precision_score(labels_binary[:, 0], probs[:, 0])  # class 1 skipped
    assert result == pytest.approx(expected)


def test_macro_ap_over_classes_returns_zero_when_all_classes_skipped():
    learner = ActiveLearner.__new__(ActiveLearner)
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels_binary = np.zeros((2, 2), dtype=np.float32)  # no positives anywhere
    assert learner._macro_ap_over_classes(probs, labels_binary, class_indices=[0, 1]) == 0.0


def test_compute_validation_estimate_unmeasured_with_zero_validation_budget(real_esc10):
    learner = build_learner(real_esc10)
    learner.add_samples(learner.sample(n_samples=10))
    learner.train_step(epochs=1, batch_size=4)

    result = learner.compute_validation_estimate()
    # Scored-only diagnostics: unmeasured -- None.
    for key in ("mAP_est", "lcb_scored", "ucb_scored", "width_scored"):
        assert result[key] is None
    assert result["mAP_theta_scored"] is None
    assert result["alpha"] == learner.alpha
    assert result["n_bootstrap"] == learner.n_bootstrap

    # Full-task range-composed fields and theta_coherent are always real,
    # even with zero validation revealed -- prevalence defaults to 0.0 for
    # every class in that case (see _class_prevalence()), so every class
    # contributes [0.0, 1.0] and theta_coherent is 0.0.
    assert result["theta_coherent"] == pytest.approx(0.0)
    assert result["theta_coherent_delta"] is None  # mAP_theta_scored is None
    assert result["lcb"] == pytest.approx(0.0)
    assert result["ucb"] == pytest.approx(1.0)
    assert result["width"] == pytest.approx(1.0)
    assert result["lcb_zero_fill"] == pytest.approx(0.0)
    assert result["winkler"] is not None
    assert isinstance(result["covered"], bool)


def test_compute_validation_estimate_interval_sanity_on_real_esc10(real_esc10):
    # Model init and sample() draws are unseeded, so whether the SCORED-ONLY
    # bootstrap lands on a normal or a degenerate (all-null) outcome isn't
    # fixed run to run (see test_compute_validation_estimate_degenerate_bootstrap_*
    # for a deterministic version of the degenerate case). Assert the
    # invariant that holds either way: scored-only fields are null all
    # together or measured all together, never a mix. The full-task
    # range-composed fields (lcb/ucb/width/theta_coherent/winkler/covered/
    # lcb_zero_fill) are always defined regardless -- they don't depend on
    # the scored-only macro bootstrap being non-degenerate.
    learner = build_learner(real_esc10, n_bootstrap=200)
    learner.add_samples(learner.sample(n_samples=20))
    learner.train_step(epochs=2, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))  # reveal all -> clears threshold

    result = learner.compute_validation_estimate(random_state=0)

    assert result["n_classes_scored"] >= 5
    assert result["alpha"] == learner.alpha
    assert result["n_bootstrap"] == 200
    assert 0.0 <= result["lcb_zero_fill"] <= 1.0
    assert 0.0 <= result["lcb"] <= result["ucb"] <= 1.0
    assert result["width"] == pytest.approx(result["ucb"] - result["lcb"])
    assert result["theta_coherent"] is not None
    assert result["winkler"] >= result["width"] - 1e-9  # width is a lower bound on the score
    assert isinstance(result["covered"], bool)

    est_fields = ["mAP_est", "lcb_scored", "ucb_scored", "width_scored"]
    if result["mAP_est"] is None:
        assert all(result[f] is None for f in est_fields)
    else:
        assert all(result[f] is not None for f in est_fields)
        assert 0.0 <= result["mAP_est"] <= 1.0
        assert 0.0 <= result["lcb_scored"] <= result["ucb_scored"] <= 1.0
        assert result["width_scored"] == pytest.approx(result["ucb_scored"] - result["lcb_scored"])


def test_compute_validation_estimate_theta_matches_compute_scored_class_truth(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=50)
    learner.add_samples(learner.sample(n_samples=20))
    learner.train_step(epochs=2, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))

    scored_truth = learner.compute_scored_class_truth()
    estimate = learner.compute_validation_estimate(random_state=0)

    assert estimate["mAP_theta_scored"] == scored_truth["mAP_theta_scored"]
    assert estimate["scored_class_indices"] == scored_truth["scored_class_indices"]


def test_compute_validation_estimate_reproducible_with_fixed_random_state(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=100)
    learner.add_samples(learner.sample(n_samples=20))
    learner.train_step(epochs=2, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))

    result_a = learner.compute_validation_estimate(random_state=42)
    result_b = learner.compute_validation_estimate(random_state=42)

    assert result_a["lcb_scored"] == result_b["lcb_scored"]
    assert result_a["ucb_scored"] == result_b["ucb_scored"]
    assert result_a["lcb"] == result_b["lcb"]
    assert result_a["ucb"] == result_b["ucb"]


def test_compute_validation_estimate_does_not_mutate_state(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=50)
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)
    learner.add_validation_samples(learner.sample_validation(n_samples=30))

    before_labels = learner.labels.copy()
    before_validation_labeled = set(learner.validation_labeled_indices)
    before_labeled = set(learner.labeled_indices)
    before_history_len = len(learner.training_history)

    learner.compute_validation_estimate(random_state=0)

    np.testing.assert_array_equal(learner.labels, before_labels)
    assert learner.validation_labeled_indices == before_validation_labeled
    assert learner.labeled_indices == before_labeled
    assert len(learner.training_history) == before_history_len


def test_compute_validation_estimate_uses_configured_defaults(real_esc10):
    learner = build_learner(real_esc10, alpha=0.20, n_bootstrap=30)
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))

    result = learner.compute_validation_estimate()
    assert result["alpha"] == 0.20
    assert result["n_bootstrap"] == 30


def test_compute_validation_estimate_degenerate_bootstrap_returns_null_and_no_coverage():
    """
    A single scored class with exactly one revealed positive sample: every
    bootstrap resample (with replacement, from a pool of 1) trivially
    reproduces that same row, so the SCORED-ONLY macro interval collapses to
    a point. Per the degenerate-bootstrap guard, this must report None for
    mAP_est/lcb_scored/ucb_scored/width_scored -- not a spuriously "perfect"
    interval -- and must not count a scored-only coverage event.
    mAP_theta_scored and the class-set fields (computed independently of the
    bootstrap) are unaffected.

    The full-task range-composed fields (lcb/ucb/width/theta_coherent/
    winkler/covered/lcb_zero_fill) do NOT inherit this null: they're built
    from PER-CLASS percentile bounds, not the macro-average bootstrap, so a
    single scored class's degenerate (but still real-valued) per-class
    bounds compose into a real, if degenerate, full-task interval.
    """
    learner = ActiveLearner.__new__(ActiveLearner)
    learner.alpha = 0.10
    learner.n_bootstrap = 50
    learner.min_positives_per_class = 1
    learner.is_multilabel = False
    learner.label_to_idx = {"a": 0, "b": 1}
    learner.mc_dropout_passes = 1
    learner.validation_labeled_indices = {0}  # a single revealed sample
    learner.labels = np.array([0])            # that sample belongs to class 0
    learner._last_per_class_ap = {0: 0.5, 1: 0.5}  # both classes have full-pool support
    learner._predict_all = lambda: np.array([[0.9, 0.1]])  # that sample's prediction

    result = learner.compute_validation_estimate(random_state=0)

    assert result["scored_class_indices"] == [0]
    assert result["mAP_theta_scored"] is not None  # class-set truth is unaffected
    for field in ("mAP_est", "lcb_scored", "ucb_scored", "width_scored"):
        assert result[field] is None

    assert result["lcb_zero_fill"] is not None
    assert 0.0 <= result["lcb_zero_fill"] <= 1.0
    assert result["theta_coherent"] is not None
    assert result["lcb"] is not None and result["ucb"] is not None
    assert result["winkler"] is not None
    assert isinstance(result["covered"], bool)


def test_compute_scored_class_truth_differs_across_independent_repeats(dataset_factory):
    """
    Two independent ActiveLearner instances (simulating two outer repeats)
    trained on the same dataset but revealing DIFFERENT validation subsets
    end up with DIFFERENT scored class sets S_r -- and each computes its own
    mAP_theta_scored purely from its own state, never a shared value. This
    is the scenario the "active testing" design explicitly calls out: S2 !=
    S1 is expected and correct, not a bug, and theta_scored must be
    recomputed per repeat against that repeat's own class set.

    Uses the same dataset_factory(n_samples=60, n_classes=3,
    with_validation=True) layout as
    test_compute_scored_class_truth_drops_rare_classes_exactly: validation
    indices 0-11, class = i % 3, so class 0 = {0,3,6,9}, class 1 =
    {1,4,7,10}, class 2 = {2,5,8,11}, 4 each.
    """
    dataset = dataset_factory(n_samples=60, n_classes=3, with_validation=True)

    repeat1 = build_learner(dataset, min_positives_per_class=3)
    repeat1.add_samples(repeat1.sample(n_samples=10))
    repeat1.train_step(epochs=1, batch_size=4)
    repeat1.add_validation_samples([0, 3, 6, 9, 1, 4])  # class 0: 4 positives, class 1: 2, class 2: 0
    result1 = repeat1.compute_scored_class_truth()

    repeat2 = build_learner(dataset, min_positives_per_class=3)
    repeat2.add_samples(repeat2.sample(n_samples=10))
    repeat2.train_step(epochs=1, batch_size=4)
    repeat2.add_validation_samples([1, 4, 7, 10, 2, 5, 8])  # class 1: 4 positives, class 2: 3, class 0: 0
    result2 = repeat2.compute_scored_class_truth()

    assert result1["scored_class_indices"] == [0]
    assert result2["scored_class_indices"] == [1, 2]
    assert result1["scored_class_indices"] != result2["scored_class_indices"]

    # Each repeat's theta is computed purely from its own _last_per_class_ap
    # -- confirm it matches that repeat's own values, not a shared/global one.
    assert result1["mAP_theta_scored"] == pytest.approx(repeat1._last_per_class_ap[0])
    assert result2["mAP_theta_scored"] == pytest.approx(
        np.mean([repeat2._last_per_class_ap[1], repeat2._last_per_class_ap[2]])
    )

    # Aggregating the two repeats' theta values is a plain mean of two
    # independently-computed numbers -- exactly what ActiveLearner.agg_mean_sd does.
    mean, sd, n = ActiveLearner.agg_mean_sd(
        [{"mAP_theta_scored": result1["mAP_theta_scored"]}, {"mAP_theta_scored": result2["mAP_theta_scored"]}],
        "mAP_theta_scored",
    )
    assert n == 2
    assert mean == pytest.approx((result1["mAP_theta_scored"] + result2["mAP_theta_scored"]) / 2)


# ---------------------------------------------------------------------------
# ActiveLearner.agg_mean_sd() -- loop-level repeat aggregation utility
# ---------------------------------------------------------------------------

def test_agg_mean_sd_sd_is_null_with_fewer_than_two_measured():
    mean, sd, n = ActiveLearner.agg_mean_sd([{"x": 0.7}, {"x": None}], "x")
    assert mean == pytest.approx(0.7)
    assert sd is None  # single observation -- not the same as zero variance
    assert n == 1


def test_agg_mean_sd_returns_none_when_nothing_measured():
    mean, sd, n = ActiveLearner.agg_mean_sd([{"x": None}, {"x": None}], "x")
    assert mean is None
    assert sd is None
    assert n == 0


def test_agg_mean_sd_matches_manual_computation_with_two_measured():
    mean, sd, n = ActiveLearner.agg_mean_sd([{"x": 0.4}, {"x": 0.6}], "x")
    assert mean == pytest.approx(0.5)
    assert sd == pytest.approx(np.std([0.4, 0.6]))
    assert n == 2


# ---------------------------------------------------------------------------
# training_history / aulc_lcb wiring -- "active testing" v2, chunk 5
# ---------------------------------------------------------------------------

def test_train_step_folds_active_testing_fields_into_metrics_with_unmeasured_defaults(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=20)
    learner.add_samples(learner.sample(n_samples=10))
    metrics = learner.train_step(epochs=1, batch_size=4)  # no validation ever revealed

    # Scored-only diagnostics: unmeasured -- None.
    for key in ("mAP_theta_scored", "mAP_est", "lcb_scored", "ucb_scored", "width_scored"):
        assert metrics[key] is None
    assert metrics["n_classes_scored"] == 0
    assert metrics["scored_class_indices"] == []
    assert metrics["alpha"] == learner.alpha
    assert metrics["n_bootstrap"] == 20
    assert metrics["lcb_scored_zeroed"] == 0.0
    assert metrics["aulc_lcb_scored"] == 0.0

    # Full-task range-composed fields are always real, even fully unmeasured:
    # nothing revealed -> prevalence 0.0 for every class -> lcb=0.0, ucb=1.0.
    assert metrics["theta_coherent"] == pytest.approx(0.0)
    assert metrics["lcb"] == pytest.approx(0.0)
    assert metrics["ucb"] == pytest.approx(1.0)
    assert metrics["lcb_zero_fill"] == pytest.approx(0.0)
    assert metrics["winkler"] is not None
    assert isinstance(metrics["covered"], bool)
    assert metrics["aulc_lcb"] == 0.0  # single point on the curve -> zero area


def test_aulc_lcb_scored_matches_manual_compute_aulc_over_zeroed_field(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=20)
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))
    learner.add_samples(learner.sample(n_samples=15))
    metrics = learner.train_step(epochs=1, batch_size=4)

    assert metrics["aulc_lcb_scored"] == pytest.approx(learner.compute_aulc("lcb_scored_zeroed"))
    assert learner.training_history[-1]["aulc_lcb_scored"] == pytest.approx(metrics["aulc_lcb_scored"])
    assert metrics["aulc_lcb"] == pytest.approx(learner.compute_aulc("lcb"))
    assert metrics["aulc_lcb_zero_fill"] == pytest.approx(learner.compute_aulc("lcb_zero_fill"))


def test_aulc_lcb_scored_zero_anchors_unmeasured_cycles_then_rises_once_validated(real_esc10):
    """
    Core chunk-5 behaviour, now on the SCORED-ONLY curve (aulc_lcb_scored):
    cycles before any validation is revealed contribute zero (proven-zero
    performance, not "unknown"); once validation is revealed and clears the
    threshold, the curve -- and its AULC -- becomes genuinely positive,
    UNLESS the bootstrap happens to be degenerate (small n_bootstrap here,
    plus unseeded model init/training, means this isn't fixed run to run --
    see test_compute_validation_estimate_degenerate_bootstrap_* for a
    deterministic version of that case). Assert the invariant that holds
    either way.
    """
    learner = build_learner(real_esc10, n_bootstrap=20)

    # Cycle 1: no validation revealed yet.
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)
    assert learner.training_history[0]["lcb_scored"] is None
    assert learner.training_history[0]["lcb_scored_zeroed"] == 0.0
    assert learner.training_history[0]["aulc_lcb_scored"] == 0.0

    # Cycle 2: reveal validation (all of it, comfortably clears the default threshold).
    learner.add_validation_samples(list(learner.validation_indices))
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)

    lower = learner.training_history[1]["lcb_scored"]
    if lower is None:
        # Degenerate bootstrap this run -- zeroed field and aulc_lcb_scored
        # both stay at the unmeasured floor, same as cycle 1.
        assert learner.training_history[1]["lcb_scored_zeroed"] == 0.0
        assert learner.training_history[1]["aulc_lcb_scored"] == 0.0
    else:
        assert learner.training_history[1]["lcb_scored_zeroed"] == pytest.approx(lower)
        assert learner.training_history[1]["aulc_lcb_scored"] > 0.0

    # The full-task lcb curve (bare aulc_lcb) is never null and is >= the
    # zero-fill companion at every cycle (prevalence >= 0 for unmeasured
    # classes).
    assert learner.training_history[0]["lcb"] is not None
    assert learner.training_history[0]["lcb"] >= learner.training_history[0]["lcb_zero_fill"]
    assert learner.training_history[1]["lcb"] is not None
    assert learner.training_history[1]["lcb"] >= learner.training_history[1]["lcb_zero_fill"]


def test_export_strips_internal_active_testing_bookkeeping_fields(real_esc10, tmp_path):
    learner = build_learner(real_esc10, n_bootstrap=20)
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)
    learner.add_validation_samples(list(learner.validation_indices))
    learner.add_samples(learner.sample(n_samples=15))
    learner.train_step(epochs=1, batch_size=4)

    output_path = tmp_path / "submission.yaml"
    learner.export(str(output_path))
    with open(output_path) as f:
        submission = yaml.safe_load(f)

    for row in submission["learning_curve"]:
        assert "scored_class_indices" not in row
        assert "lcb_scored_zeroed" not in row
        # But the meaningful active-testing fields survive export.
        assert "aulc_lcb" in row
        assert "aulc_lcb_scored" in row
        assert "aulc_lcb_zero_fill" in row
        assert "mAP_theta_scored" in row
        assert "theta_coherent" in row


def test_train_step_reports_n_validated(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=10)
    learner.add_samples(learner.sample(n_samples=10))
    metrics = learner.train_step(epochs=1, batch_size=4)
    assert metrics["n_validated"] == 0

    learner.add_validation_samples(learner.sample_validation(n_samples=12))
    learner.add_samples(learner.sample(n_samples=10))
    metrics = learner.train_step(epochs=1, batch_size=4)
    assert metrics["n_validated"] == 12
    assert learner.training_history[-1]["n_validated"] == 12


def test_train_step_reports_theta_scored_delta_and_zero_fill_aulc(real_esc10):
    learner = build_learner(real_esc10, n_bootstrap=20)
    learner.add_samples(learner.sample(n_samples=15))
    metrics = learner.train_step(epochs=1, batch_size=4)  # no validation revealed yet
    assert metrics["mAP_theta_scored_delta"] is None
    assert metrics["aulc_lcb_zero_fill"] == 0.0  # always defined, zero-anchored

    learner.add_validation_samples(list(learner.validation_indices))
    learner.add_samples(learner.sample(n_samples=15))
    metrics = learner.train_step(epochs=1, batch_size=4)

    if metrics["mAP_theta_scored"] is not None:
        assert metrics["mAP_theta_scored_delta"] == pytest.approx(
            metrics["mAP_theta_scored"] - metrics["mAP"]
        )
    assert metrics["aulc_lcb_zero_fill"] == pytest.approx(
        learner.compute_aulc("lcb_zero_fill")
    )
