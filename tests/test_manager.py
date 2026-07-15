"""
Tests for core/active_learner.py (Manager). Uses the real ESC10 demo dataset
(`real_esc10` fixture) -- comparing "Random" vs "Margin" sampling on the same
demo data mirrors the exact pattern in the shipped `core/config.yml`.
"""
import json
from pathlib import Path

import pytest
import yaml

from core.active_learner import Manager

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_config(tmp_path, dataset, strategies):
    """Write a Manager-compatible config.yml with one experiment per strategy,
    all pointed at the same (real) dataset -- exactly how core/config.yml
    compares sampling strategies against ESC10 today."""
    experiments = []
    for name, strategy in strategies:
        experiments.append({
            "name": name,
            "embeddings_dir": str(dataset["embeddings_dir"]),
            "annotations_path": str(dataset["labels_path"]),
            "model_name": dataset["model_name"],
            "hidden_dim": 8,
            "sampling_strategy": strategy,
            "learning_rate": 0.01,
            "repeats": 1,
            "device": "cpu",
            "pretrain_samples": None,
        })
    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump({"experiments": experiments}, f)
    return config_path


def test_manager_requires_experiments_key(tmp_path):
    config_path = tmp_path / "bad_config.yml"
    with open(config_path, "w") as f:
        yaml.dump({"not_experiments": []}, f)
    with pytest.raises(ValueError):
        Manager(config_path, verbose=False)


def test_manager_loads_all_configured_experiments(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random"), ("Margin", "margin")])

    manager = Manager(config_path, verbose=False)
    assert manager.experiment_names == ["Random", "Margin"]
    assert len(manager.experiments) == 2


def test_manager_relative_paths_resolve_against_base_dir():
    # Mirrors the real core/config.yml pattern: paths given relative to the
    # repo root, exactly as a participant's config file would.
    config_path = REPO_ROOT / "core" / "config.yml"
    manager = Manager(config_path, base_dir=REPO_ROOT, verbose=False)
    assert len(manager.experiments) >= 1
    assert manager.experiments[0].embeddings.shape[0] > 0


def test_manager_run_sequential_returns_metrics_per_experiment(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random"), ("Margin", "margin")])
    manager = Manager(config_path, verbose=False)

    results = manager.run(n_samples=5, epochs=1, batch_size=4, parallel=False)
    assert set(results.keys()) == {"Random", "Margin"}
    for metrics in results.values():
        assert "accuracy" in metrics
        assert metrics["n_labeled"] == 5


def test_manager_run_parallel_returns_same_shape_as_sequential(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random"), ("Margin", "margin")])
    manager = Manager(config_path, verbose=False)

    results = manager.run(n_samples=5, epochs=1, batch_size=4, parallel=True)
    assert set(results.keys()) == {"Random", "Margin"}
    for metrics in results.values():
        assert "accuracy" in metrics


def test_manager_add_dynamically_appends_experiment(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random")])
    manager = Manager(config_path, verbose=False)
    assert len(manager.experiments) == 1

    manager.add({
        "embeddings_dir": real_esc10["embeddings_dir"],
        "annotations_path": real_esc10["labels_path"],
        "model_name": real_esc10["model_name"],
        "hidden_dim": 8,
        "sampling_strategy": "random",
        "device": "cpu",
        "verbose": False,
    }, name="Extra")

    assert manager.experiment_names[-1] == "Extra"
    assert len(manager.experiments) == 2


def test_manager_save_writes_per_experiment_and_summary_json(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random")])
    manager = Manager(config_path, verbose=False)
    manager.run(n_samples=5, epochs=1, batch_size=4, parallel=False)

    output_dir = tmp_path / "results"
    manager.save(output_dir=output_dir)

    saved_files = list(output_dir.glob("Random_*.json"))
    assert len(saved_files) == 1
    with open(saved_files[0]) as f:
        payload = json.load(f)
    assert payload["experiment_name"] == "Random"
    assert "aulc_mAP" in payload

    summary_files = list(output_dir.glob("summary_*.json"))
    assert len(summary_files) == 1
    with open(summary_files[0]) as f:
        summary = json.load(f)
    assert summary["num_experiments"] == 1
    assert summary["experiments"][0]["name"] == "Random"


def test_manager_get_summary_reflects_current_state(tmp_path, real_esc10):
    config_path = write_config(tmp_path, real_esc10, [("Random", "random")])
    manager = Manager(config_path, verbose=False)

    summary = manager.get_summary()
    assert summary["num_experiments"] == 1
    assert summary["experiments"][0]["n_labeled"] == 0

    manager.run(n_samples=5, epochs=1, batch_size=4, parallel=False)
    summary = manager.get_summary()
    assert summary["experiments"][0]["n_labeled"] == 5
