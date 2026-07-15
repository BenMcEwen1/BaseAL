"""
Shared pytest fixtures for BaseAL tests.

Most tests exercise the real ESC10 demo dataset shipped in-repo under
`ESC10_BASEAL/` (see `real_esc10` below) -- this is the actual data path
users hit, so it's preferred over synthetic data wherever it's usable as-is.

Synthetic "BaseAL dataset" directories (see `dataset_factory`) are still used
where real data can't cover the case:
  - multilabel-format labels (ESC10 is single-label only; multilabel demo
    data such as BirdSet/ATBFL isn't checked into the repo)
  - deliberately malformed/degenerate inputs for error-path and edge-case
    tests (missing files, tiny n, invalid idxes, etc.)
  - sampling-strategy unit tests that need exact, hand-constructed ground
    truth (e.g. "this prediction is maximally ambiguous") to assert a
    specific ranking

Synthetic datasets follow the same on-disk contract the framework expects:

    {dataset}/
    +-- embeddings/{model_name}/{stem}_{model_name}.npy
    +-- labels.csv   # filename, label, [validation]
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend -- must be set before pyplot is imported anywhere

import numpy as np
import pandas as pd
import pytest

MODEL_NAME = "testmodel"
EMBEDDING_DIM = 16

REPO_ROOT = Path(__file__).resolve().parent.parent
ESC10_ROOT = REPO_ROOT / "ESC10_BASEAL"
ESC10_MODEL_NAME = "birdnet"


def make_dataset(
    root: Path,
    n_samples: int = 40,
    n_classes: int = 4,
    embedding_dim: int = EMBEDDING_DIM,
    model_name: str = MODEL_NAME,
    multilabel: bool = False,
    with_validation: bool = False,
    separable: bool = True,
    seed: int = 0,
) -> dict:
    """
    Build a synthetic BaseAL-format dataset under `root` and return a dict of
    useful paths/metadata for tests to consume.
    """
    rng = np.random.default_rng(seed)
    embeddings_dir = root / "embeddings" / model_name
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    class_names = [f"class{i}" for i in range(n_classes)]
    rows = []

    # Class "centroids" spread out so a linear classifier can separate them.
    centroids = np.zeros((n_classes, embedding_dim), dtype=np.float32)
    for c in range(n_classes):
        centroids[c, c % embedding_dim] = 6.0 if separable else 0.0

    for i in range(n_samples):
        stem = f"sample_{i:03d}"
        cls = i % n_classes
        emb = rng.normal(scale=0.5, size=embedding_dim).astype(np.float32)
        emb += centroids[cls]
        np.save(embeddings_dir / f"{stem}_{model_name}.npy", emb)

        if multilabel:
            # Every sample keeps its "primary" class plus, for 1-in-3
            # samples, a secondary class -- exercises the multilabel path.
            labels = {class_names[cls]}
            if i % 3 == 0:
                labels.add(class_names[(cls + 1) % n_classes])
            label_str = ";".join(sorted(labels))
        else:
            label_str = class_names[cls]

        row = {"filename": f"{stem}.wav", "label": label_str}
        if with_validation:
            row["validation"] = "true" if i < max(2, n_samples // 5) else "false"
        rows.append(row)

    labels_path = root / "labels.csv"
    pd.DataFrame(rows).to_csv(labels_path, index=False)

    return {
        "root": root,
        "embeddings_dir": embeddings_dir,
        "labels_path": labels_path,
        "class_names": class_names,
        "model_name": model_name,
        "n_samples": n_samples,
        "embedding_dim": embedding_dim,
    }


@pytest.fixture
def dataset_factory(tmp_path):
    """Factory fixture: call with kwargs to build a fresh synthetic dataset."""
    counter = {"n": 0}

    def _factory(**kwargs):
        counter["n"] += 1
        root = tmp_path / f"dataset_{counter['n']}"
        return make_dataset(root, **kwargs)

    return _factory


@pytest.fixture
def simple_dataset(dataset_factory):
    """A small, separable, single-label dataset with no validation split."""
    return dataset_factory(n_samples=40, n_classes=4)


@pytest.fixture
def multilabel_dataset(dataset_factory):
    """A small multilabel dataset (semicolon-separated labels).

    No real multilabel demo data ships in this repo (ESC10 is single-label
    only), so this is a deliberate synthetic exception.
    """
    return dataset_factory(n_samples=30, n_classes=4, multilabel=True)


@pytest.fixture(scope="session")
def real_esc10() -> dict:
    """
    The real ESC10 demo dataset shipped in-repo (`ESC10_BASEAL/`): 800
    samples, 10 classes, 1024-d BirdNet embeddings, with a real 640/160
    train-vs-validation split already encoded in `labels.csv`.

    Read-only: every test using this fixture must pass the full, unmodified
    `labels_path`/`embeddings_dir` through so the on-disk embedding cache
    manifest still matches and no test ever triggers a cache rebuild that
    would rewrite files inside the checked-in dataset directory.
    """
    embeddings_dir = ESC10_ROOT / "embeddings" / ESC10_MODEL_NAME
    labels_path = ESC10_ROOT / "labels.csv"
    if not labels_path.exists() or not embeddings_dir.exists():
        pytest.skip("Real ESC10_BASEAL demo data not present in this checkout")

    df = pd.read_csv(labels_path)
    class_names = sorted(df["label"].unique().tolist())
    sample_embedding = np.load(next(embeddings_dir.glob("*.npy")))
    n_validation = 0
    if "validation" in df.columns:
        n_validation = int(df["validation"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())

    return {
        "root": ESC10_ROOT,
        "embeddings_dir": embeddings_dir,
        "labels_path": labels_path,
        "class_names": class_names,
        "model_name": ESC10_MODEL_NAME,
        "n_samples": len(df),
        "n_validation": n_validation,
        "embedding_dim": int(sample_embedding.shape[-1]),
    }
