"""
Tests for core/utils/visualization.py (visualize_embeddings).

Happy-path tests use the real ESC10 demo dataset (`real_esc10` fixture).
Error-path, degenerate-input, and multilabel-specific tests stay on
synthetic fixtures for concrete reasons: they need deliberately
malformed/tiny inputs the real dataset can't provide, or (for multilabel)
there's no real multilabel demo data checked into this repo.

Regression coverage: `_resolve_embedding_names` tries '{stem}.npy' first,
falling back to '{stem}_{model_name}.npy'. Real ESC10 embeddings use the
former, the synthetic fixtures use the latter -- so together these tests
exercise both branches of that fallback.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from core.utils.visualization import visualize_embeddings


def test_visualize_embeddings_basic_shapes(real_esc10):
    result = visualize_embeddings(real_esc10["root"], max_reference_samples=200)
    try:
        assert result.coordinates.shape == (real_esc10["n_samples"], 2)
        assert bool(result.is_multilabel) is False
        assert len(result.label_names) == real_esc10["n_samples"]
        assert len(result.reduction_steps) >= 1
        assert result.reduction_steps[0] == "StandardScaler"
    finally:
        plt.close(result.figure)


def test_visualize_embeddings_without_label_coloring(real_esc10):
    result = visualize_embeddings(real_esc10["root"], with_label=False, max_reference_samples=200)
    try:
        assert result.coordinates.shape == (real_esc10["n_samples"], 2)
    finally:
        plt.close(result.figure)


def test_visualize_embeddings_missing_dataset_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        visualize_embeddings(tmp_path / "does_not_exist")


def test_visualize_embeddings_missing_labels_csv_raises(tmp_path):
    root = tmp_path / "no_labels"
    (root / "embeddings" / "model").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        visualize_embeddings(root)


def test_visualize_embeddings_missing_embeddings_dir_raises(tmp_path):
    root = tmp_path / "no_embeddings"
    root.mkdir()
    (root / "labels.csv").write_text("filename,label\na.wav,x\n")
    with pytest.raises(FileNotFoundError):
        visualize_embeddings(root)


def test_visualize_embeddings_multiple_model_dirs_raises(simple_dataset):
    extra_model_dir = simple_dataset["root"] / "embeddings" / "another_model"
    extra_model_dir.mkdir()
    with pytest.raises(ValueError):
        visualize_embeddings(simple_dataset["root"])


def test_visualize_embeddings_multilabel_names(multilabel_dataset):
    result = visualize_embeddings(multilabel_dataset["root"])
    try:
        assert bool(result.is_multilabel) is True
        assert any(";" in name for name in result.label_names)
    finally:
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# Embedding filename convention fallback ('{stem}.npy' vs '{stem}_{model}.npy')
# ---------------------------------------------------------------------------

def test_visualize_embeddings_resolves_real_esc10_unsuffixed_filenames(real_esc10):
    # Real ESC10 embeddings are named plain '{stem}.npy' (no model suffix) --
    # this is a regression test for that specific convention.
    idxes = np.array([0, 1, 2, 3, 4])
    result = visualize_embeddings(real_esc10["root"], idxes=idxes)
    try:
        assert result.coordinates.shape == (5, 2)
    finally:
        plt.close(result.figure)


def test_visualize_embeddings_resolves_suffixed_filenames(simple_dataset):
    # The synthetic fixture uses '{stem}_{model_name}.npy' -- the fallback
    # convention (e.g. BirdSet-style embeddings).
    result = visualize_embeddings(simple_dataset["root"])
    try:
        assert result.coordinates.shape == (simple_dataset["n_samples"], 2)
    finally:
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# idxes validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_idxes,error", [
    (np.array([[0, 1], [2, 3]]), ValueError),        # not 1D
    (np.array([], dtype=int), ValueError),            # empty
    (np.array([True, False]), TypeError),             # bool dtype
    (np.array([0.5, 1.5]), TypeError),                 # non-integer
    (np.array([-1, 0]), ValueError),                   # negative
    (np.array([0, 0, 1]), ValueError),                 # duplicates
])
def test_visualize_embeddings_rejects_invalid_idxes(real_esc10, bad_idxes, error):
    with pytest.raises(error):
        visualize_embeddings(real_esc10["root"], idxes=bad_idxes)


def test_visualize_embeddings_rejects_out_of_range_idxes(real_esc10):
    n = real_esc10["n_samples"]
    with pytest.raises(IndexError):
        visualize_embeddings(real_esc10["root"], idxes=np.array([0, n + 5]))


def test_visualize_embeddings_with_idxes_subset(real_esc10):
    idxes = np.array([0, 1, 2, 3, 4])
    result = visualize_embeddings(real_esc10["root"], idxes=idxes)
    try:
        assert result.coordinates.shape == (5, 2)
        np.testing.assert_array_equal(result.idxes, idxes)
    finally:
        plt.close(result.figure)


# ---------------------------------------------------------------------------
# max_reference_samples validation
# ---------------------------------------------------------------------------

def test_visualize_embeddings_rejects_non_positive_max_reference_samples(real_esc10):
    with pytest.raises(ValueError):
        visualize_embeddings(real_esc10["root"], max_reference_samples=0)


def test_visualize_embeddings_rejects_non_integer_max_reference_samples(real_esc10):
    with pytest.raises(TypeError):
        visualize_embeddings(real_esc10["root"], max_reference_samples=3.5)


def test_visualize_embeddings_rejects_non_bool_with_label(real_esc10):
    with pytest.raises(TypeError):
        visualize_embeddings(real_esc10["root"], with_label="yes")


# ---------------------------------------------------------------------------
# Degenerate / small-n fallback projection path
# ---------------------------------------------------------------------------

def test_visualize_embeddings_tiny_dataset_uses_fallback_projection(dataset_factory):
    tiny = dataset_factory(n_samples=2, n_classes=2)
    result = visualize_embeddings(tiny["root"])
    try:
        assert result.coordinates.shape == (2, 2)
        assert "DegenerateProjection" in result.reduction_steps
    finally:
        plt.close(result.figure)
