"""
Sampling strategies for active learning
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import logging
# from skactiveml.pool import UncertaintySampling  # Not currently used in the code; temporarily commented out.
import faiss

logger = logging.getLogger(__name__)


def densityEstimation(embeddings: Optional[np.ndarray] = None, method='cosine', beta: int = 1, k: int = 20):
    if method == 'cosine':
        similarity = cosine_similarity(embeddings)
    elif method == 'euclidean':
        similarity = euclidean_distances(embeddings)
    elif method == 'knn':
        knn = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distance, _ = knn.kneighbors(embeddings)
        similarity = distance.T
    else:
        raise Exception("Unknown similarity estimation method, ")

    density = np.power(np.sum(similarity, axis=0) / np.sum(similarity, axis=0).max(), beta)
    return density


def _sample_pool(unlabeled: np.ndarray, pool_size: int, rng: np.random.Generator) -> np.ndarray:
    if pool_size <= 0 or len(unlabeled) <= pool_size:
        return unlabeled
    return rng.choice(unlabeled, size=pool_size, replace=False)


def _build_hnsw_index(x: np.ndarray, m: int = 32, ef_search: int = 128) -> faiss.Index:
    d = int(x.shape[1])
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efSearch = ef_search
    index.add(x.astype(np.float32, copy=False))
    return index


def _project_with_pca(x: np.ndarray, out_dim: int, train_rows: int = 20000) -> np.ndarray:
    if out_dim <= 0 or out_dim >= x.shape[1]:
        return x
    n_train = min(train_rows, x.shape[0])
    if n_train <= out_dim:
        return x
    pca = faiss.PCAMatrix(x.shape[1], out_dim)
    pca.train(x[:n_train].astype(np.float32, copy=False))
    return pca.apply_py(x.astype(np.float32, copy=False))


class SamplingStrategy:
    """
    Unified sampling strategy class that handles all sampling methods.

    This class contains the selection logic and various sampling methods.
    Data is stored as instance attributes and accessed by sampling methods.
    """

    def __init__(self, method: str = "random", n_samples: int = 20, random_state: Optional[int] = None):
        """
        Initialize sampling strategy

        Args:
            method: Sampling method to use ('random', 'margin', 'custom', 'margin_multilabel', 'coreset_farthest', 'nn_disagreement')
            n_samples: Number of samples to select per iteration
            random_state: Optional random seed for reproducibility
        """
        self.method = method
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

        # Available sampling methods
        available_methods = [
            'random',
            'margin',
            'custom',
            'margin_multilabel',
            'coreset_farthest',
            'nn_disagreement',
        ]

        if method not in available_methods:
            raise ValueError(
                f"Unknown sampling strategy: {method}. "
                f"Available strategies: {available_methods}"
            )

        # Map method names to their implementation functions
        self._method_map = {
            'random': self._random,
            'margin': self._margin,
            'custom': self._custom,
            'margin_multilabel': self._margin_multilabel,
            'coreset_farthest': self._coreset_farthest,
            'nn_disagreement': self._nn_disagreement,
        }

        # Data attributes (see selct)
        self.unlabeled_indices = None
        self.predictions = None
        self.embeddings = None
        self.model = None
        self.annotations = None
        self.labeled_indices = None
        self.labels = None

        logger.info(f"Initialized SamplingStrategy with method='{method}' and n_samples={n_samples}")

    def select(self,
               unlabeled_indices: List[int],
               predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None,
               model=None,
               annotations: Optional[pd.DataFrame] = None,
               labeled_indices: Optional[List[int]] = None,
               labels: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples for annotation and compute per-sample uncertainties.

        This is the main selection method that stores the input data as instance
        attributes and calls the appropriate sampling method.

        Args:
            unlabeled_indices: List/array of unlabeled sample indices
            predictions: Optional numpy array of model predictions (N x num_classes)
            embeddings: Optional numpy array of embeddings (N x embedding_dim)
            model: Optional reference to the model itself
            annotations: Optional DataFrame containing annotation data and metadata
            labeled_indices: Optional list/array of labeled sample indices
            labels: Optional ground-truth labels for all samples

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected sample indices
                - uncertainties: Normalized uncertainty scores for unlabeled samples [0, 1]
                  where 1 = maximum uncertainty, 0 = complete certainty
        """
        if len(unlabeled_indices) == 0:
            logger.warning("No unlabeled samples available for selection")
            return [], np.array([])

        # Store data as instance attributes for sampling methods to access
        self.unlabeled_indices = unlabeled_indices
        self.predictions = predictions
        self.embeddings = embeddings
        self.model = model
        self.annotations = annotations
        self.labeled_indices = labeled_indices if labeled_indices is not None else []
        self.labels = labels

        # print(len(self.labels))

        # Call the appropriate sampling method to get uncertainties
        sampling_func = self._method_map[self.method]
        uncertainties = np.asarray(sampling_func(), dtype=np.float32)

        if len(uncertainties) != len(unlabeled_indices):
            raise ValueError(
                f"Sampling method '{self.method}' returned {len(uncertainties)} scores, "
                f"expected {len(unlabeled_indices)}"
            )

        # Select samples with highest uncertainties
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(uncertainties)[-n_samples:]  # Highest uncertainties

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        logger.info(f"Selected {len(selected)} samples using {self.method} sampling")
        logger.info(f"utility range: min={uncertainties.min():.4f}, max={uncertainties.max():.4f}, mean={uncertainties.mean():.4f}")

        return selected, uncertainties

    def _random(self) -> np.ndarray:
        """
        Random sampling strategy - assigns equal utility to all samples.

        For random sampling, all samples have equal utility (1.0), so selection
        is effectively random.

        Returns:
            utility: Array of 1.0 for all unlabeled samples (equal utility)
        """
        # Sample random utility to make top-k selection random.
        utility = self.rng.random(len(self.unlabeled_indices), dtype=np.float32)
        return utility

    def _margin(self) -> np.ndarray:
        """
        Margin sampling - selects samples with smallest margin between top two predictions.

        The margin is the difference between the highest and second-highest predicted
        class probabilities. Smaller margins indicate more ambiguous predictions.

        Returns:
            utility: Normalized utility scores (1 - margin) for all unlabeled samples [0, 1]
        """
        if self.predictions is None:
            raise ValueError("Margin sampling requires predictions")

        unlabeled_preds = self.predictions[self.unlabeled_indices]

        # Sort predictions for each sample to get top 2
        sorted_preds = np.sort(unlabeled_preds, axis=1)

        # Calculate margin: difference between top two predictions
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]

        # Uncertainty = 1 - margin (smaller margin = higher uncertainty, already normalized to [0, 1])
        utility = 1.0 - margins

        logger.info(f"Margin sampling - margins min: {margins.min():.4f}, max: {margins.max():.4f}")
        logger.info(f"Margin sampling - utility min: {utility.min():.4f}, max: {utility.max():.4f}")
        return utility

    def _custom(self) -> np.ndarray:
        """
        Custom sampling template.

        INSTRUCTIONS FOR IMPLEMENTING CUSTOM SAMPLING:
        ===============================================

        1. This method should compute utility scores for all unlabeled samples.

        2. The utility scores should be normalized to [0, 1] where:
           - 1.0 = maximum utility (highest priority for annotation)
           - 0.0 = lowest utility (lowest priority for annotation)

        3. Available instance attributes (set by select() method):
           - self.unlabeled_indices: List of indices in the unlabeled pool
           - self.predictions: Model predictions array of shape (n_total_samples, num_classes)
                              Contains probabilities for all classes
           - self.embeddings: Full embeddings array of shape (n_total_samples, embedding_dim)
                             The raw feature vectors before classification
           - self.model: Reference to the trained model (if you need to extract features/gradients)
           - self.annotations: DataFrame containing annotation data and metadata
                              Can contain custom metadata fields for advanced sampling strategies

        Returns:
            utility: Array of utility scores for samples [0, 1]
        """
        # TODO: Implement your custom sampling logic here
        # For now, default to random sampling
        logger.warning("Custom sampling not implemented, falling back to random sampling")
        return self._random()

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if len(scores) == 0:
            return np.array([], dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        if np.isclose(max_v, min_v):
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_v) / (max_v - min_v)

    @staticmethod
    def _labels_to_prob_matrix(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert single-label or multi-label targets to a probability-like matrix."""
        if labels.ndim == 2:
            return labels.astype(np.float32, copy=False)

        one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        idx = labels.astype(np.int64, copy=False)
        one_hot[np.arange(labels.shape[0]), idx] = 1.0
        return one_hot

    def _margin_multilabel(self) -> np.ndarray:
        """
        Marginal query for multi-label classification:
        utility = 1 - 2 * min(|p - 0.5|) for each sample.
        """
        if self.predictions is None:
            raise ValueError("margin_multilabel requires predictions")

        unlabeled_probs = self.predictions[self.unlabeled_indices]
        margins = np.min(np.abs(unlabeled_probs - 0.5), axis=1)
        utility = np.clip(1.0 - 2.0 * margins, 0.0, 1.0).astype(np.float32)
        return utility

    def _coreset_farthest(self) -> np.ndarray:
        """
        Coreset farthest query with FAISS acceleration:
        1) approximate pool subsampling
        2) FAISS HNSW nearest-anchor distance
        3) far-distance preselection
        4) optional FAISS PCA + farthest-first refinement
        """
        if self.embeddings is None:
            raise ValueError("coreset_farthest requires embeddings")

        unlabeled = np.asarray(self.unlabeled_indices, dtype=int)
        k = min(self.n_samples, len(unlabeled))
        utility = np.zeros(len(unlabeled), dtype=np.float32)
        if len(unlabeled) == 0 or k <= 0:
            return utility

        labeled = np.asarray(self.labeled_indices, dtype=int)
        if len(labeled) == 0:
            return self._random()

        approx_pool_size = 80000
        anchor_limit = 30000
        preselect_factor = 8
        pca_dim = 64

        pool = _sample_pool(unlabeled, approx_pool_size, self.rng)
        x_pool = self.embeddings[pool].astype(np.float32, copy=False)
        if len(pool) == 0:
            return utility

        if len(labeled) > anchor_limit:
            labeled = self.rng.choice(labeled, size=anchor_limit, replace=False)
        x_labeled = self.embeddings[labeled].astype(np.float32, copy=False)

        anchor_index = _build_hnsw_index(x_labeled)
        dists, _ = anchor_index.search(x_pool, 1)
        dists = dists.reshape(-1)

        # Base utility for the sampled pool from anchor distances.
        pool_utility = 0.95 * self._normalize(dists)
        unlabeled_pos = {idx: pos for pos, idx in enumerate(unlabeled)}
        for pool_pos, global_idx in enumerate(pool):
            utility[unlabeled_pos[int(global_idx)]] = pool_utility[pool_pos]

        preselect = min(len(pool), max(k, k * preselect_factor))
        far_order = np.argsort(-dists)[:preselect]
        picked_indices = pool[far_order]
        x_picked = x_pool[far_order]

        x_proj = _project_with_pca(x_picked, out_dim=pca_dim)
        if x_proj.shape[0] == 0:
            return utility

        target_k = min(k, x_proj.shape[0])
        selected_local = np.empty((target_k,), dtype=int)
        min_dist = np.full((x_proj.shape[0],), np.inf, dtype=np.float32)

        seed_choice = int(self.rng.integers(0, x_proj.shape[0]))
        selected_local[0] = seed_choice
        seed_vec = x_proj[seed_choice]
        min_dist = np.minimum(min_dist, np.sum((x_proj - seed_vec) ** 2, axis=1))
        min_dist[seed_choice] = -np.inf

        selected_count = 1
        for _ in range(1, target_k):
            nxt = int(np.argmax(min_dist))
            if not np.isfinite(min_dist[nxt]):
                break
            selected_local[selected_count] = nxt
            selected_count += 1
            nxt_vec = x_proj[nxt]
            min_dist = np.minimum(min_dist, np.sum((x_proj - nxt_vec) ** 2, axis=1))
            min_dist[nxt] = -np.inf

        selected_indices = picked_indices[selected_local[:selected_count]]
        for rank, global_idx in enumerate(selected_indices):
            utility[unlabeled_pos[int(global_idx)]] = 1.0 - rank * 1e-6

        return utility.astype(np.float32)

    def _nn_disagreement(self) -> np.ndarray:
        """
        Nearest-neighbor disagreement query.
        Uses pool subsampling + FAISS HNSW neighbors.
        Disagreement is mean absolute difference between model probabilities and
        neighborhood label distribution estimated from labeled samples.
        """
        if self.embeddings is None or self.predictions is None:
            raise ValueError("nn_disagreement requires embeddings and predictions")
        if self.labels is None:
            raise ValueError("nn_disagreement requires labels")

        unlabeled = np.asarray(self.unlabeled_indices, dtype=int)
        k = min(self.n_samples, len(unlabeled))
        utility = np.zeros(len(unlabeled), dtype=np.float32)
        if len(unlabeled) == 0:
            return utility

        labeled = np.asarray(self.labeled_indices, dtype=int)
        if len(labeled) == 0:
            return self._random()

        # Keep pool size fixed for now (same style as coreset_farthest).
        approx_pool_size = 80000
        nn_train_limit = 50000
        n_neighbors = 15

        pool = _sample_pool(unlabeled, approx_pool_size, self.rng)
        if len(pool) == 0 or k <= 0:
            return utility

        if len(labeled) > nn_train_limit:
            labeled = self.rng.choice(labeled, size=nn_train_limit, replace=False)

        x_pool = self.embeddings[pool].astype(np.float32, copy=False)
        x_labeled = self.embeddings[labeled].astype(np.float32, copy=False)

        model_probs = self.predictions[pool]
        if model_probs.ndim == 1:
            model_probs = model_probs[:, None]
        num_classes = model_probs.shape[1]

        label_probs = self._labels_to_prob_matrix(np.asarray(self.labels), num_classes)
        labeled_targets = label_probs[labeled]

        index = _build_hnsw_index(x_labeled)
        nn_k = min(n_neighbors, len(labeled))
        _, nbr_idx = index.search(x_pool, nn_k)
        nn_probs = labeled_targets[nbr_idx].mean(axis=1)

        if nn_probs.shape[1] != model_probs.shape[1]:
            common_dim = min(nn_probs.shape[1], model_probs.shape[1])
            logger.warning(
                "Probability dimension mismatch in nn_disagreement: "
                f"model={model_probs.shape[1]}, nn={nn_probs.shape[1]}. "
                f"Using first {common_dim} dims."
            )
            nn_probs = nn_probs[:, :common_dim]
            model_probs = model_probs[:, :common_dim]

        disagreement = np.mean(np.abs(model_probs - nn_probs), axis=1)
        pool_utility = self._normalize(disagreement)

        unlabeled_pos = {idx: pos for pos, idx in enumerate(unlabeled)}
        for pool_pos, global_idx in enumerate(pool):
            utility[unlabeled_pos[int(global_idx)]] = pool_utility[pool_pos]

        order = np.argsort(-disagreement)
        selected_indices = pool[order[:k]]
        for rank, global_idx in enumerate(selected_indices):
            utility[unlabeled_pos[int(global_idx)]] = 1.0 - rank * 1e-6

        return utility.astype(np.float32)
