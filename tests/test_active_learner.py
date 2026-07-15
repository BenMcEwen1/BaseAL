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
