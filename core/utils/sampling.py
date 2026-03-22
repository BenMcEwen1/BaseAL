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
            'bald',
            # 'margin_multilabel',
            'coreset_farthest',
            'nn_disagreement',
            'margin_multilabel',
            'sklearn_coreset',
            'sklearn_typiclust',
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
            'bald': self._bald,
            # 'margin_multilabel': self._margin_multilabel,
            'coreset_farthest': self._coreset_farthest,
            'nn_disagreement': self._nn_disagreement,
            'margin_multilabel': self._margin_multilabel,
            'sklearn_coreset': self._sklearn_coreset,
            'sklearn_typiclust': self._sklearn_typiclust,
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
               labels: Optional[np.ndarray] = None,
               mc_predictions: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples for annotation and compute per-sample utility.

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
            mc_predictions: Optional repeated forward-pass predictions (mc_passes, N, num_classes)

        Returns:
            Tuple of (selected_indices, utility):
                - selected_indices: List of selected sample indices
                - utility: Normalized utility scores for unlabeled samples [0, 1]
                  where 1 = maximum utility, 0 = lowest utility
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
        self.mc_predictions = mc_predictions

        # print(len(self.labels))

        # Call the appropriate sampling method to get utility scores
        sampling_func = self._method_map[self.method]
        utility = np.asarray(sampling_func(), dtype=np.float32)

        if len(utility) != len(unlabeled_indices):
            raise ValueError(
                f"Sampling method '{self.method}' returned {len(utility)} scores, "
                f"expected {len(unlabeled_indices)}"
            )

        # Select samples with highest utility
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(utility)[-n_samples:]  # Highest uncertainties

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        logger.info(f"Selected {len(selected)} samples using {self.method} sampling")
        logger.info(f"utility range: min={utility.min():.4f}, max={utility.max():.4f}, mean={utility.mean():.4f}")

        return selected, utility

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

        1. This method computes utility scores for all unlabeled samples.

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
    
    def _bald(self) -> np.ndarray:
        # self.mc_predictions: (n_passes, n_samples, n_classes)

        if self.mc_predictions is None:
            raise ValueError("MC predictions unavailable, this method requires multiple mc_dropout_passes to be set")

        mc = self.mc_predictions[:, self.unlabeled_indices, :]  # (n_passes, n_unlabeled, n_classes)
        mean_p = mc.mean(axis=0)                                # (n_unlabeled, n_classes)
        H_mean = -np.sum(mean_p * np.log(mean_p + 1e-8), axis=1)          # predictive entropy
        mean_H = -np.mean(np.sum(mc * np.log(mc + 1e-8), axis=2), axis=0) # expected entropy
        return H_mean - mean_H                                              # BALD score

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

    # def _margin_multilabel(self) -> np.ndarray:
    #     """
    #     Marginal query for multi-label classification:
    #     utility = 1 - 2 * min(|p - 0.5|) for each sample.
    #     """
    #     if self.predictions is None:
    #         raise ValueError("margin_multilabel requires predictions")

    #     unlabeled_probs = self.predictions[self.unlabeled_indices]
    #     margins = np.min(np.abs(unlabeled_probs - 0.5), axis=1)
    #     utility = np.clip(1.0 - 2.0 * margins, 0.0, 1.0).astype(np.float32)
    #     return utility

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

    def _margin_multilabel(self) -> np.ndarray:
        """
        Margin sampling with mean aggregation for multilabel classification.

        For each sample, computes the mean distance of each label probability from
        the decision boundary (|p - 0.5|) and averages across labels. Samples
        closest to the boundary (lowest mean margin) are most uncertain.

        This is the multilabel analogue of:
            skactiveml.pool.UncertaintySampling(method='margin_sampling')
        which cannot be used directly here because it requires a classifier object
        to call predict_proba — unnecessary when predictions are already available.

        scikit-activeml reference:
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.UncertaintySampling.html

        Returns:
            utility: Normalized scores [0, 1] where 1 = on the decision boundary
        """
        if self.predictions is None:
            raise ValueError("margin_multilabel requires predictions")

        unlabeled_preds = self.predictions[self.unlabeled_indices]

        # Per-label distance from the decision boundary: |p - 0.5| in [0, 0.5]
        # Mean across labels: small value => high uncertainty
        mean_margin = np.mean(np.abs(unlabeled_preds - 0.5), axis=1)

        # Normalize to [0, 1]: 0.5 is the max possible mean_margin
        utility = (1.0 - (mean_margin / 0.5)).astype(np.float32)

        logger.info(f"margin_multilabel - mean_margin min: {mean_margin.min():.4f}, max: {mean_margin.max():.4f}")
        return utility

    def _sklearn_coreset(self) -> np.ndarray:
        """
        Greedy k-center (CoreSet) sampling using scikit-activeml.

        Selects a diverse set of samples by minimising the maximum distance from
        any unlabeled point to its nearest selected (or already-labeled) point.

        scikit-activeml reference:
            skactiveml.pool.CoreSet
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.CoreSet.html

        Returns:
            utility: Normalized min-distance-to-labeled scores [0, 1]
                     where 1 = farthest from any labeled point
        """
        from skactiveml.pool import CoreSet
        from skactiveml.utils import MISSING_LABEL

        if self.embeddings is None:
            raise ValueError("sklearn_coreset requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        # Cold-start fallback: no labeled samples means no anchor for distance computation.
        if len(self.labeled_indices) == 0:
            logger.warning("sklearn_coreset: no labeled samples, falling back to random")
            return self._random()

        # Build y: labeled samples get a dummy label (0), unlabeled get MISSING_LABEL.
        # CoreSet only uses y to distinguish labeled from unlabeled — label values are ignored.
        y = np.zeros(n_total, dtype=float)
        y[self.unlabeled_indices] = MISSING_LABEL

        strategy = CoreSet(missing_label=MISSING_LABEL)

        selected_indices, utilities = strategy.query(
            X=self.embeddings,
            y=y,
            candidates=np.array(self.unlabeled_indices),
            batch_size=n_samples,
            return_utilities=True,
        )

        # utilities[0] = initial min-distance-to-labeled for every candidate — use for
        # visualisation / relative ranking of non-selected samples.
        util_scores = utilities[0][self.unlabeled_indices]
        utility = self._normalize(np.clip(util_scores, 0, None)) * 0.99

        # Override utility to 1.0 for the samples scikit-activeml actually chose via its
        # greedy k-center algorithm. This ensures select() reproduces the greedy selection
        # rather than a naive top-K on the initial distances (which clusters picks together).
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(f"sklearn_coreset - selected {len(selected_indices)} samples via greedy k-center")
        return utility.astype(np.float32)

    def _sklearn_typiclust(self) -> np.ndarray:
        """
        TypiClust sampling using scikit-activeml.

        Clusters the embedding space (KMeans) then, for each cluster without a
        labeled sample, selects the most typical (highest-density) unlabeled point.
        Promotes coverage and typicality rather than outlier selection.

        scikit-activeml reference:
            skactiveml.pool.TypiClust
            https://scikit-activeml.github.io/latest/generated/skactiveml.pool.TypiClust.html

        Returns:
            utility: Normalized typicality scores [0, 1]
                     where 1 = most typical in its cluster
        """
        from skactiveml.pool import TypiClust
        from skactiveml.utils import MISSING_LABEL

        if self.embeddings is None:
            raise ValueError("sklearn_typiclust requires embeddings")

        n_total = self.embeddings.shape[0]
        n_samples = min(self.n_samples, len(self.unlabeled_indices))

        # Project to a lower-dimensional space before clustering.
        # KMeans is slow and noisy in high dimensions; PCA to 64-D gives a large
        # speed-up with minimal loss of cluster structure (same as _coreset_farthest).
        pca_dim = 64
        X = _project_with_pca(self.embeddings.astype(np.float32, copy=False), out_dim=pca_dim)

        # Build y: labeled samples get a dummy label (0), unlabeled get MISSING_LABEL.
        y = np.zeros(n_total, dtype=float)
        y[self.unlabeled_indices] = MISSING_LABEL

        strategy = TypiClust(missing_label=MISSING_LABEL)

        selected_indices, utilities = strategy.query(
            X=X,
            y=y,
            candidates=np.array(self.unlabeled_indices),
            batch_size=n_samples,
            return_utilities=True,
        )

        # utilities[0] = typicality scores; -inf means the cluster is already covered by a
        # labeled sample so the point should not be selected — map to 0 for display.
        util_scores = utilities[0][self.unlabeled_indices]
        util_scores = np.where(np.isfinite(util_scores), util_scores, 0.0)
        utility = self._normalize(util_scores) * 0.99

        # Override utility to 1.0 for the samples TypiClust actually chose (one per
        # uncovered cluster). This ensures select() reproduces TypiClust's coverage-aware
        # selection rather than a naive top-K that may pick multiple from the same cluster.
        unlabeled_pos = {idx: pos for pos, idx in enumerate(self.unlabeled_indices)}
        for idx in selected_indices:
            if idx in unlabeled_pos:
                utility[unlabeled_pos[idx]] = 1.0

        logger.info(f"sklearn_typiclust - typicality min: {util_scores.min():.4f}, max: {util_scores.max():.4f}")
        return utility.astype(np.float32)
