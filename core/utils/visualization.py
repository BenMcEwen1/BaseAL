"""
Notebook-friendly embedding visualisation utilities for BaseAL datasets.

Usage
-----
from core.utils.visualization import visualize_embeddings
import numpy as np

viz = visualize_embeddings("ESC10_BASEAL", idxes=np.arange(500), with_label=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.colors import hsv_to_rgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


RANDOM_STATE = 0
LEGEND_CLASS_LIMIT = 15
COLOR_WHEEL_SATURATION = 0.8
COLOR_WHEEL_VALUE = 0.9
COLOR_WHEEL_OFFSET = 0.0


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    dataset_path: Path
    labels_path: Path
    embeddings_dir: Path
    model_name: str


@dataclass(frozen=True, slots=True)
class PreparedVisualizationData:
    idxes: np.ndarray
    embeddings: np.ndarray
    all_embedding_names: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    label_indices_for_color: np.ndarray
    class_names: list[str]
    annotations: pd.DataFrame
    is_multilabel: bool
    model_name: str
    dataset_path: Path
    embeddings_dir: Path


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    figure: Figure
    axes: Axes
    idxes: np.ndarray
    coordinates: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray
    label_names: list[str]
    label_indices_for_color: np.ndarray
    class_names: list[str]
    annotations: pd.DataFrame
    is_multilabel: bool
    dataset_path: Path
    embeddings_dir: Path
    model_name: str
    reduction_steps: tuple[str, ...]


def visualize_embeddings(
    dataset_path: str | Path,
    idxes: Optional[np.ndarray | list[int] | tuple[int, ...]] = None,
    with_label: bool = True,
    max_reference_samples: int = 5000,
) -> VisualizationResult:
    """
    Load, reduce and plot BaseAL embeddings with a single notebook call.

    Args:
        dataset_path: BaseAL dataset root containing ``labels.csv`` and ``embeddings/``.
        idxes: Optional 0-based indices into ``labels.csv`` rows. Missing embedding
            files are not pre-filtered and will raise when the selected samples are
            loaded. When ``None``, all annotation rows are visualised.
        with_label: Whether to colour points by label.
        max_reference_samples: Maximum number of reference samples used to fit the
            scaler and dimensionality reducer. When the dataset has at most this
            many samples, the full dataset is used. Otherwise a fixed random subset
            of this size is used as the fit reference set.

    Returns:
        VisualizationResult containing the matplotlib objects and sampled data.
    """
    _validate_inputs(
        with_label=with_label,
        max_reference_samples=max_reference_samples,
    )

    dataset = _resolve_dataset(dataset_path)
    prepared = _prepare_visualization_data(dataset, idxes=idxes)
    reference_idxes = _select_reference_idxes(
        n_dataset_samples=len(prepared.all_embedding_names),
        max_reference_samples=max_reference_samples,
    )
    reference_embeddings = _load_reference_embeddings(
        embeddings_dir=prepared.embeddings_dir,
        all_embedding_names=prepared.all_embedding_names,
        reference_idxes=reference_idxes,
        selected_idxes=prepared.idxes,
        selected_embeddings=prepared.embeddings,
    )
    coordinates, reduction_steps = _reduce_embeddings(
        reference_embeddings=reference_embeddings,
        embeddings=prepared.embeddings,
    )
    figure, axes = _plot_embeddings(
        coordinates=coordinates,
        label_indices_for_color=prepared.label_indices_for_color,
        class_names=prepared.class_names,
        with_label=with_label,
        dataset_name=prepared.dataset_path.name,
        model_name=prepared.model_name,
        is_multilabel=prepared.is_multilabel,
    )

    return VisualizationResult(
        figure=figure,
        axes=axes,
        idxes=prepared.idxes,
        coordinates=coordinates,
        embeddings=prepared.embeddings,
        labels=prepared.labels,
        label_names=prepared.label_names,
        label_indices_for_color=prepared.label_indices_for_color,
        class_names=prepared.class_names,
        annotations=prepared.annotations,
        is_multilabel=prepared.is_multilabel,
        dataset_path=prepared.dataset_path,
        embeddings_dir=prepared.embeddings_dir,
        model_name=prepared.model_name,
        reduction_steps=reduction_steps,
    )


def _validate_inputs(with_label: bool, max_reference_samples: int) -> None:
    if not isinstance(with_label, bool):
        raise TypeError(
            f"with_label must be a bool, got {type(with_label).__name__}"
        )
    if isinstance(max_reference_samples, bool) or not isinstance(
        max_reference_samples, Integral
    ):
        raise TypeError(
            "max_reference_samples must be an integer, "
            f"got {type(max_reference_samples).__name__}"
        )
    if max_reference_samples <= 0:
        raise ValueError("max_reference_samples must be a positive integer")


def _resolve_dataset(dataset_path: str | Path) -> ResolvedDataset:
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {dataset_root}")

    labels_path = dataset_root / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Expected labels.csv under dataset root, but it was not found: {labels_path}"
        )

    embeddings_root = dataset_root / "embeddings"
    if not embeddings_root.exists():
        raise FileNotFoundError(
            f"Expected embeddings directory under dataset root, but it was not found: {embeddings_root}"
        )

    candidate_dirs = sorted(
        path for path in embeddings_root.iterdir() if path.is_dir()
    )

    if not candidate_dirs:
        raise FileNotFoundError(
            f"No embedding model directory was found under: {embeddings_root}"
        )

    if len(candidate_dirs) > 1:
        candidate_names = ", ".join(path.name for path in candidate_dirs)
        raise ValueError(
            "dataset_path must resolve to a dataset with exactly one embedding model "
            "directory because this API visualises one dataset/model pair at a time. "
            f"Found multiple model directories under {embeddings_root}: {candidate_names}"
        )

    embeddings_dir = candidate_dirs[0]
    return ResolvedDataset(
        dataset_path=dataset_root,
        labels_path=labels_path,
        embeddings_dir=embeddings_dir,
        model_name=embeddings_dir.name,
    )


def _prepare_visualization_data(
    dataset: ResolvedDataset,
    idxes: Optional[np.ndarray | list[int] | tuple[int, ...]],
) -> PreparedVisualizationData:
    annotations = pd.read_csv(dataset.labels_path)
    required_columns = {"filename", "label"}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"labels.csv is missing required columns: {missing}. "
            f"Found columns: {annotations.columns.tolist()}"
        )

    label_strings = annotations["label"].astype(str).str.strip()
    filename_strings = annotations["filename"].astype(str).str.strip()

    is_multilabel = label_strings.str.contains(";", regex=False).any()
    class_names = _extract_class_names(label_strings)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    stems = filename_strings.str.rsplit(".", n=1).str[0]
    annotations = annotations.reset_index(drop=True).copy()
    annotations["annotation_index"] = np.arange(len(annotations), dtype=np.int64)
    annotations["matched_index"] = annotations["annotation_index"].to_numpy(copy=True)
    annotations["embedding_name"] = (stems + f"_{dataset.model_name}.npy").to_numpy()
    annotations["label_string"] = label_strings.to_numpy()

    selected_idxes, sampled_annotations = _select_annotations_by_idxes(
        annotations,
        idxes=idxes,
    )
    sampled_embeddings = _load_embedding_matrix(
        embeddings_dir=dataset.embeddings_dir,
        embedding_names=sampled_annotations["embedding_name"].tolist(),
    )
    labels, label_names, label_indices_for_color = _build_labels(
        label_strings=sampled_annotations["label_string"].tolist(),
        label_to_idx=label_to_idx,
        class_names=class_names,
        is_multilabel=is_multilabel,
    )

    return PreparedVisualizationData(
        idxes=selected_idxes,
        embeddings=sampled_embeddings,
        all_embedding_names=annotations["embedding_name"].to_numpy(copy=True),
        labels=labels,
        label_names=label_names,
        label_indices_for_color=label_indices_for_color,
        class_names=class_names,
        annotations=sampled_annotations,
        is_multilabel=is_multilabel,
        model_name=dataset.model_name,
        dataset_path=dataset.dataset_path,
        embeddings_dir=dataset.embeddings_dir,
    )


def _extract_class_names(label_strings: pd.Series) -> list[str]:
    classes = sorted(
        {
            token.strip()
            for token in label_strings.str.split(";").explode().dropna()
            if token.strip()
        }
    )
    if not classes:
        raise ValueError("No valid labels were found in labels.csv")
    return classes


def _select_annotations_by_idxes(
    annotations: pd.DataFrame,
    idxes: Optional[np.ndarray | list[int] | tuple[int, ...]],
) -> tuple[np.ndarray, pd.DataFrame]:
    normalized_idxes = _normalize_idxes(idxes, n_available=len(annotations))
    if normalized_idxes is None:
        selected = annotations.copy().reset_index(drop=True)
        return selected["matched_index"].to_numpy(dtype=np.int64), selected

    selected = annotations.iloc[normalized_idxes].copy().reset_index(drop=True)
    return normalized_idxes, selected


def _normalize_idxes(
    idxes: Optional[np.ndarray | list[int] | tuple[int, ...]],
    n_available: int,
) -> Optional[np.ndarray]:
    if idxes is None:
        return None

    idxes_array = np.asarray(idxes)
    if idxes_array.ndim != 1:
        raise ValueError(
            f"idxes must be a 1D array-like of integers, got shape {idxes_array.shape}"
        )
    if idxes_array.size == 0:
        raise ValueError("idxes must not be empty when provided")
    if idxes_array.dtype == np.bool_ or np.issubdtype(idxes_array.dtype, np.bool_):
        raise TypeError("idxes must contain integer indices, not booleans")
    if not np.issubdtype(idxes_array.dtype, np.integer):
        raise TypeError(
            f"idxes must contain integers, got dtype {idxes_array.dtype}"
        )

    normalized = idxes_array.astype(np.int64, copy=False)
    if np.any(normalized < 0):
        raise ValueError("idxes must be non-negative")
    if np.any(normalized >= n_available):
        raise IndexError(
            f"idxes contains values outside the valid range [0, {n_available - 1}]"
        )
    if len(np.unique(normalized)) != len(normalized):
        raise ValueError("idxes must not contain duplicates")

    return normalized


def _load_embedding_matrix(
    embeddings_dir: Path,
    embedding_names: list[str],
) -> np.ndarray:
    if not embedding_names:
        raise ValueError("No embedding filenames were provided for loading")

    sample_embedding = _flatten_embedding(
        np.load(embeddings_dir / embedding_names[0], allow_pickle=False)
    )
    embedding_dim = int(sample_embedding.shape[0])
    embeddings = np.empty((len(embedding_names), embedding_dim), dtype=np.float32)
    embeddings[0] = sample_embedding

    for row_idx, embedding_name in enumerate(embedding_names[1:], start=1):
        embeddings[row_idx] = _flatten_embedding(
            np.load(embeddings_dir / embedding_name, allow_pickle=False)
        )

    return embeddings


def _flatten_embedding(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32)
    if array.ndim == 1:
        return array
    if len(array) == 1:
        return np.asarray(array[0], dtype=np.float32).reshape(-1)
    return array.reshape(-1).astype(np.float32, copy=False)


def _build_labels(
    label_strings: list[str],
    label_to_idx: dict[str, int],
    class_names: list[str],
    is_multilabel: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    if is_multilabel:
        labels = np.zeros((len(label_strings), len(class_names)), dtype=np.float32)
        label_names: list[str] = []
        label_indices_for_color = np.zeros(len(label_strings), dtype=np.int64)

        for row_idx, label_string in enumerate(label_strings):
            for token in label_string.split(";"):
                label = token.strip()
                if label:
                    labels[row_idx, label_to_idx[label]] = 1.0

            active_indices = np.flatnonzero(labels[row_idx])
            if len(active_indices):
                label_names.append(";".join(class_names[idx] for idx in active_indices))
                label_indices_for_color[row_idx] = int(active_indices[0])
            else:
                label_names.append("none")
                label_indices_for_color[row_idx] = 0

        return labels, label_names, label_indices_for_color

    labels = np.array([label_to_idx[label_string] for label_string in label_strings], dtype=np.int64)
    label_names = [class_names[label_idx] for label_idx in labels]
    return labels, label_names, labels.copy()


def _select_reference_idxes(
    n_dataset_samples: int,
    max_reference_samples: int,
) -> np.ndarray:
    if n_dataset_samples <= max_reference_samples:
        return np.arange(n_dataset_samples, dtype=np.int64)

    rng = np.random.default_rng(RANDOM_STATE)
    reference_idxes = rng.choice(
        n_dataset_samples,
        size=max_reference_samples,
        replace=False,
    ).astype(np.int64, copy=False)
    reference_idxes.sort()
    return reference_idxes


def _load_reference_embeddings(
    embeddings_dir: Path,
    all_embedding_names: np.ndarray,
    reference_idxes: np.ndarray,
    selected_idxes: np.ndarray,
    selected_embeddings: np.ndarray,
) -> np.ndarray:
    selected_row_by_idx = {
        int(sample_idx): row_idx for row_idx, sample_idx in enumerate(selected_idxes)
    }
    selected_row_positions = [
        selected_row_by_idx.get(int(reference_idx)) for reference_idx in reference_idxes
    ]

    if all(row_idx is not None for row_idx in selected_row_positions):
        return np.asarray(
            selected_embeddings[np.array(selected_row_positions, dtype=np.int64)],
            dtype=np.float32,
        )

    reference_embedding_names = all_embedding_names[reference_idxes].tolist()
    return _load_embedding_matrix(
        embeddings_dir=embeddings_dir,
        embedding_names=reference_embedding_names,
    )


def _reduce_embeddings(
    reference_embeddings: np.ndarray,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if reference_embeddings.ndim != 2:
        raise ValueError(
            f"reference_embeddings must be a 2D array, got shape {reference_embeddings.shape}"
        )
    n_reference_samples, n_features = reference_embeddings.shape
    if n_reference_samples == 0:
        raise ValueError("At least one reference embedding is required for visualisation")
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2D array, got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != n_features:
        raise ValueError(
            "Reference embeddings and selected embeddings must share the same feature "
            f"dimension, got {n_features} and {embeddings.shape[1]}"
        )

    reduction_steps: list[str] = ["StandardScaler"]
    scaler = StandardScaler()
    scaled_reference_embeddings = scaler.fit_transform(reference_embeddings)
    scaled_embeddings = scaler.transform(embeddings)

    if n_reference_samples == 1:
        return np.zeros((len(embeddings), 2), dtype=np.float32), tuple(
            reduction_steps + ["DegenerateProjection"]
        )

    reduced_reference = scaled_reference_embeddings
    reduced_embeddings = scaled_embeddings
    if n_features > 64:
        n_components = min(64, n_reference_samples, n_features)
        if n_components >= 2:
            pca = PCA(n_components=n_components)
            reduced_reference = pca.fit_transform(scaled_reference_embeddings)
            reduced_embeddings = pca.transform(scaled_embeddings)
            reduction_steps.append(f"PCA({n_components})")
        else:
            projected = _fallback_projection(scaled_embeddings)
            return projected, tuple(reduction_steps + ["DegenerateProjection"])

    if n_reference_samples < 3:
        projected = _fallback_projection(reduced_embeddings)
        return projected, tuple(reduction_steps + ["DegenerateProjection"])

    n_neighbors = min(15, n_reference_samples - 1)
    if n_neighbors < 2:
        projected = _fallback_projection(reduced_embeddings)
        return projected, tuple(reduction_steps + ["DegenerateProjection"])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    reducer.fit(reduced_reference)
    coordinates = reducer.transform(reduced_embeddings).astype(np.float32, copy=False)
    reduction_steps.append("UMAP(2)")
    return coordinates, tuple(reduction_steps)


def _fallback_projection(embeddings: np.ndarray) -> np.ndarray:
    n_samples, n_features = embeddings.shape
    if n_features >= 2:
        coordinates = embeddings[:, :2]
    elif n_features == 1:
        coordinates = np.column_stack([embeddings[:, 0], np.zeros(n_samples, dtype=np.float32)])
    else:
        coordinates = np.zeros((n_samples, 2), dtype=np.float32)

    coordinates = np.asarray(coordinates, dtype=np.float32)
    if n_samples == 2 and np.allclose(coordinates[0], coordinates[1]):
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    return coordinates


def _plot_embeddings(
    coordinates: np.ndarray,
    label_indices_for_color: np.ndarray,
    class_names: list[str],
    with_label: bool,
    dataset_name: str,
    model_name: str,
    is_multilabel: bool,
) -> tuple[Figure, Axes]:
    figure, axes = plt.subplots(figsize=(10, 8), constrained_layout=True)
    point_size = _point_size(len(coordinates))

    if with_label:
        present_indices, class_colors = _present_class_colors(label_indices_for_color)
        axes.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            color=class_colors[label_indices_for_color],
            s=point_size,
            alpha=0.8,
            linewidths=0.0,
        )
        _maybe_add_legend(
            axes=axes,
            present_indices=present_indices,
            class_names=class_names,
            class_colors=class_colors,
        )
        title_label = "with labels"
    else:
        axes.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            color="#4C78A8",
            s=point_size,
            alpha=0.75,
            linewidths=0.0,
        )
        title_label = "without labels"

    dataset_mode = "multi-label" if is_multilabel else "multi-class"
    axes.set_title(
        f"{dataset_name} ({model_name})\n"
        f"{dataset_mode}, n={len(coordinates)}, {title_label}"
    )
    axes.set_xlabel("UMAP-1")
    axes.set_ylabel("UMAP-2")
    axes.grid(alpha=0.2, linewidth=0.5)

    return figure, axes


def _point_size(n_points: int) -> float:
    if n_points <= 500:
        return 20.0
    if n_points <= 2_000:
        return 12.0
    if n_points <= 10_000:
        return 8.0
    return 5.0


def _present_class_colors(
    label_indices_for_color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    present_indices = np.unique(label_indices_for_color).astype(int)
    max_class_index = int(present_indices.max()) if len(present_indices) else -1
    class_colors = np.zeros((max_class_index + 1, 4), dtype=np.float32)

    if len(present_indices) == 0:
        return present_indices, class_colors

    if len(present_indices) == 1:
        hues = np.array([2.0 / 3.0], dtype=np.float32)
    else:
        hues = (
            np.linspace(0.0, 1.0, len(present_indices), endpoint=False, dtype=np.float32)
            + COLOR_WHEEL_OFFSET
        ) % 1.0

    hsv = np.column_stack(
        [
            hues,
            np.full(len(present_indices), COLOR_WHEEL_SATURATION, dtype=np.float32),
            np.full(len(present_indices), COLOR_WHEEL_VALUE, dtype=np.float32),
        ]
    )
    rgb = hsv_to_rgb(hsv)

    for palette_idx, class_idx in enumerate(present_indices):
        class_colors[class_idx, :3] = rgb[palette_idx]
        class_colors[class_idx, 3] = 1.0

    return present_indices, class_colors


def _maybe_add_legend(
    axes: Axes,
    present_indices: np.ndarray,
    class_names: list[str],
    class_colors: np.ndarray,
) -> None:
    if len(present_indices) == 0:
        return

    if len(present_indices) > LEGEND_CLASS_LIMIT:
        axes.text(
            1.02,
            0.98,
            f"Legend omitted\n{len(present_indices)} classes in sample",
            transform=axes.transAxes,
            va="top",
            ha="left",
            fontsize=9,
        )
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=class_colors[class_idx],
            markeredgecolor="none",
            label=class_names[class_idx],
        )
        for class_idx in present_indices
    ]
    axes.legend(
        handles=handles,
        title="Label",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
    )


plot_embeddings = visualize_embeddings


__all__ = [
    "PreparedVisualizationData",
    "ResolvedDataset",
    "VisualizationResult",
    "plot_embeddings",
    "visualize_embeddings",
]
