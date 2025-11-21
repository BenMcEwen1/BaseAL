"""
Sampling strategies for active learning
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors


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


class SamplingStrategy(ABC):
    """Base class for all sampling strategies"""

    def __init__(self, n_samples: int = 20):
        """
        Initialize sampling strategy

        Args:
            n_samples: Number of samples to select per iteration
        """
        self.n_samples = n_samples

    @abstractmethod
    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples for annotation and compute per-sample uncertainties

        Args:
            unlabeled_indices: List/array of unlabeled sample indices
            predictions: Optional numpy array of model predictions (N x num_classes)
            embeddings: Optional numpy array of embeddings
            model: Optional reference to the model itself

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected sample indices
                - uncertainties: Normalized uncertainty scores for unlabeled samples [0, 1]
                  where 1 = maximum uncertainty, 0 = complete certainty
        """
        pass


class RandomSampling(SamplingStrategy):
    """Random sampling strategy - only needs indices"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:
        """
        Randomly select samples from unlabeled pool
        For random sampling, all samples have equal uncertainty (1.0)

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Not used
            embeddings: Not used
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of randomly selected indices
                - uncertainties: Array of 1.0 for all unlabeled samples (equal uncertainty)
        """
        n_samples = min(self.n_samples, len(unlabeled_indices))
        selected = np.random.choice(unlabeled_indices, size=n_samples, replace=False).tolist()

        # For random sampling, all samples have equal uncertainty
        uncertainties = np.ones(len(unlabeled_indices))

        return selected, uncertainties


class EntropySampling(SamplingStrategy):
    """Entropy-based sampling - selects samples with highest prediction entropy"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples with highest prediction entropy

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected indices with highest entropy
                - uncertainties: Normalized entropy scores for all unlabeled samples [0, 1]
        """
        if predictions is None:
            raise ValueError("EntropySampling requires predictions")

        # Calculate entropy for unlabeled samples only
        unlabeled_preds = predictions[unlabeled_indices]

        # Compute entropy: -sum(p * log(p))
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(unlabeled_preds * np.log(unlabeled_preds + epsilon), axis=1)

        # Normalize entropy to [0, 1]
        # Max entropy = log(num_classes) for uniform distribution
        num_classes = unlabeled_preds.shape[1]
        max_entropy = np.log(num_classes)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy

        # Select samples with highest entropy
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(entropy)[-n_samples:]

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        return selected, normalized_entropy


class UncertaintySampling(SamplingStrategy):
    """Uncertainty sampling - selects samples with lowest maximum probability (least confident)"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples with lowest confidence (lowest max probability)

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected indices with lowest confidence
                - uncertainties: Normalized uncertainty scores (1 - max_prob) for all unlabeled samples [0, 1]
        """
        if predictions is None:
            raise ValueError("UncertaintySampling requires predictions")

        unlabeled_preds = predictions[unlabeled_indices]
        max_probs = np.max(unlabeled_preds, axis=1)

        # Uncertainty = 1 - max_probability (already normalized to [0, 1])
        uncertainties = 1.0 - max_probs

        # Select samples with lowest maximum probability (most uncertain)
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(max_probs)[:n_samples]  # Lowest confidence first

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        return selected, uncertainties


class MarginSampling(SamplingStrategy):
    """Margin sampling - selects samples with smallest margin between top two predictions"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:
        """
        Select samples with smallest margin between top two class probabilities

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected indices with smallest margins
                - uncertainties: Normalized uncertainty scores (1 - margin) for all unlabeled samples [0, 1]
        """
        if predictions is None:
            raise ValueError("MarginSampling requires predictions")

        unlabeled_preds = predictions[unlabeled_indices]

        # Sort predictions for each sample to get top 2
        sorted_preds = np.sort(unlabeled_preds, axis=1)

        # Calculate margin: difference between top two predictions
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]

        # Uncertainty = 1 - margin (smaller margin = higher uncertainty, already normalized to [0, 1])
        uncertainties = 1.0 - margins

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MarginSampling - margins min: {margins.min():.4f}, max: {margins.max():.4f}")
        logger.info(f"MarginSampling - uncertainties min: {uncertainties.min():.4f}, max: {uncertainties.max():.4f}")

        # Select samples with smallest margins (most ambiguous)
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(margins)[:n_samples]  # Smallest margins first

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        return selected, uncertainties


class Anomaly(SamplingStrategy):
    """
    """

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> Tuple[List[int], np.ndarray]:

        if predictions is None:
            raise ValueError("MarginSampling requires predictions")

        # passing logits as predictions here...
        temperature = 2

        print(predictions.shape)
        unlabeled_preds = predictions[unlabeled_indices]
        logits = unlabeled_preds


        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Compute softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Get maximum probability
        max_probs = np.max(probs, axis=1)

        anomaly_score = 1 - max_probs

        # Debug logging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"MarginSampling - margins min: {max_probs.min():.4f}, max: {max_probs.max():.4f}")
        logger.info(f"MarginSampling - uncertainties min: {anomaly_score.min():.4f}, max: {anomaly_score.max():.4f}")

        # Select samples with smallest margins (most ambiguous)
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(max_probs)[:n_samples]  # Smallest margins first

        selected = np.array(unlabeled_indices)[top_indices].tolist()

        return selected, anomaly_score


class MarginDiversitySamplingDensity(SamplingStrategy):
    """
    Margin sampling with diversification across the embedding space.

    This strategy combines uncertainty-based selection (margin sampling) with
    diversity-based selection to ensure selected samples are both uncertain
    and spread out across the feature space.
    """

    def __init__(self, n_samples: int = 20, candidate_multiplier: float = 3.0):
        """
        Initialize margin diversity sampling strategy

        Args:
            n_samples: Number of samples to select per iteration
            candidate_multiplier: Factor to determine candidate pool size
                                (candidate_pool_size = n_samples * candidate_multiplier)
        """
        super().__init__(n_samples)
        self.candidate_multiplier = candidate_multiplier

    def _compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between embeddings

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Using broadcasting to compute all pairwise distances efficiently
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        norms_squared = np.sum(embeddings ** 2, axis=1, keepdims=True)
        distances = norms_squared + norms_squared.T - 2 * np.dot(embeddings, embeddings.T)
        # Clip negative values due to numerical errors
        distances = np.sqrt(np.maximum(distances, 0))
        return distances

    def _greedy_diversity_selection(self,
                                     candidate_embeddings: np.ndarray,
                                     candidate_indices: np.ndarray,
                                     n_select: int) -> List[int]:
        """
        Greedily select diverse samples using k-center approach

        Starts with the sample farthest from the origin, then iteratively
        selects samples that are farthest from already selected samples.

        Args:
            candidate_embeddings: Embeddings of candidate samples (n_candidates, embedding_dim)
            candidate_indices: Original indices of candidates
            n_select: Number of samples to select

        Returns:
            List of selected indices from candidate_indices
        """
        n_candidates = len(candidate_embeddings)

        if n_candidates <= n_select:
            return candidate_indices.tolist()

        # Initialize selection with sample farthest from origin
        norms = np.linalg.norm(candidate_embeddings, axis=1)
        first_idx = np.argmax(norms)

        selected_mask = np.zeros(n_candidates, dtype=bool)
        selected_mask[first_idx] = True
        selected_count = 1

        # Track minimum distance from each candidate to any selected sample
        min_distances = np.full(n_candidates, np.inf)

        # Compute distances to first selected sample
        first_embedding = candidate_embeddings[first_idx:first_idx+1]
        distances_to_first = np.linalg.norm(
            candidate_embeddings - first_embedding, axis=1
        )
        min_distances = np.minimum(min_distances, distances_to_first)

        # Greedily select remaining samples
        while selected_count < n_select:
            # Exclude already selected samples
            min_distances[selected_mask] = -np.inf

            # Select sample with maximum distance to nearest selected sample
            next_idx = np.argmax(min_distances)
            selected_mask[next_idx] = True
            selected_count += 1

            # Update minimum distances
            next_embedding = candidate_embeddings[next_idx:next_idx+1]
            distances_to_next = np.linalg.norm(
                candidate_embeddings - next_embedding, axis=1
            )
            min_distances = np.minimum(min_distances, distances_to_next)

        # Return original indices of selected samples
        selected_local_indices = np.where(selected_mask)[0]
        return candidate_indices[selected_local_indices].tolist()

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None, density = True) -> Tuple[List[int], np.ndarray]:
        """
        Select samples with smallest margins and high diversity in embedding space

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Embeddings array (N x embedding_dim) - REQUIRED
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected indices with small margins and high diversity
                - uncertainties: Normalized uncertainty scores (1 - margin) for all unlabeled samples [0, 1]
        """
        if predictions is None:
            raise ValueError("MarginDiversitySampling requires predictions")
        if embeddings is None:
            raise ValueError("MarginDiversitySampling requires embeddings")

        import logging
        logger = logging.getLogger(__name__)

        unlabeled_preds = predictions[unlabeled_indices]
        unlabeled_embeddings = embeddings[unlabeled_indices]

        # Step 1: Compute margin-based uncertainties
        sorted_preds = np.sort(unlabeled_preds, axis=1)
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]
        uncertainties = 1.0 - margins

        if density:
            uncertainties = uncertainties * densityEstimation(unlabeled_embeddings)

        logger.info(f"MarginDiversitySampling - margins min: {margins.min():.4f}, max: {margins.max():.4f}")
        logger.info(f"MarginDiversitySampling - uncertainties min: {uncertainties.min():.4f}, max: {uncertainties.max():.4f}")

        # Step 2: Select candidate pool based on margin uncertainty
        n_samples = min(self.n_samples, len(unlabeled_indices))
        candidate_pool_size = min(
            int(n_samples * self.candidate_multiplier),
            len(unlabeled_indices)
        )

        # Get indices of top uncertain samples (smallest margins)
        candidate_local_indices = np.argsort(margins)[:candidate_pool_size]
        candidate_embeddings = unlabeled_embeddings[candidate_local_indices]

        logger.info(f"MarginDiversitySampling - candidate pool size: {candidate_pool_size}, target samples: {n_samples}")

        # Step 3: Apply diversity selection on candidate pool
        selected_local_indices = self._greedy_diversity_selection(
            candidate_embeddings,
            candidate_local_indices,
            n_samples
        )

        # Convert local indices (within unlabeled set) to global indices
        selected = np.array(unlabeled_indices)[selected_local_indices].tolist()

        logger.info(f"MarginDiversitySampling - selected {len(selected)} diverse samples from {candidate_pool_size} candidates")

        return selected, uncertainties
    

class MarginDiversitySampling(SamplingStrategy):
    """
    Margin sampling with diversification across the embedding space.

    This strategy combines uncertainty-based selection (margin sampling) with
    diversity-based selection to ensure selected samples are both uncertain
    and spread out across the feature space.
    """

    def __init__(self, n_samples: int = 20, candidate_multiplier: float = 3.0):
        """
        Initialize margin diversity sampling strategy

        Args:
            n_samples: Number of samples to select per iteration
            candidate_multiplier: Factor to determine candidate pool size
                                (candidate_pool_size = n_samples * candidate_multiplier)
        """
        super().__init__(n_samples)
        self.candidate_multiplier = candidate_multiplier

    def _compute_pairwise_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between embeddings

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Using broadcasting to compute all pairwise distances efficiently
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        norms_squared = np.sum(embeddings ** 2, axis=1, keepdims=True)
        distances = norms_squared + norms_squared.T - 2 * np.dot(embeddings, embeddings.T)
        # Clip negative values due to numerical errors
        distances = np.sqrt(np.maximum(distances, 0))
        return distances

    def _greedy_diversity_selection(self,
                                     candidate_embeddings: np.ndarray,
                                     candidate_indices: np.ndarray,
                                     n_select: int) -> List[int]:
        """
        Greedily select diverse samples using k-center approach

        Starts with the sample farthest from the origin, then iteratively
        selects samples that are farthest from already selected samples.

        Args:
            candidate_embeddings: Embeddings of candidate samples (n_candidates, embedding_dim)
            candidate_indices: Original indices of candidates
            n_select: Number of samples to select

        Returns:
            List of selected indices from candidate_indices
        """
        n_candidates = len(candidate_embeddings)

        if n_candidates <= n_select:
            return candidate_indices.tolist()

        # Initialize selection with sample farthest from origin
        norms = np.linalg.norm(candidate_embeddings, axis=1)
        first_idx = np.argmax(norms)

        selected_mask = np.zeros(n_candidates, dtype=bool)
        selected_mask[first_idx] = True
        selected_count = 1

        # Track minimum distance from each candidate to any selected sample
        min_distances = np.full(n_candidates, np.inf)

        # Compute distances to first selected sample
        first_embedding = candidate_embeddings[first_idx:first_idx+1]
        distances_to_first = np.linalg.norm(
            candidate_embeddings - first_embedding, axis=1
        )
        min_distances = np.minimum(min_distances, distances_to_first)

        # Greedily select remaining samples
        while selected_count < n_select:
            # Exclude already selected samples
            min_distances[selected_mask] = -np.inf

            # Select sample with maximum distance to nearest selected sample
            next_idx = np.argmax(min_distances)
            selected_mask[next_idx] = True
            selected_count += 1

            # Update minimum distances
            next_embedding = candidate_embeddings[next_idx:next_idx+1]
            distances_to_next = np.linalg.norm(
                candidate_embeddings - next_embedding, axis=1
            )
            min_distances = np.minimum(min_distances, distances_to_next)

        # Return original indices of selected samples
        selected_local_indices = np.where(selected_mask)[0]
        return candidate_indices[selected_local_indices].tolist()

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None, density = False) -> Tuple[List[int], np.ndarray]:
        """
        Select samples with smallest margins and high diversity in embedding space

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Embeddings array (N x embedding_dim) - REQUIRED
            model: Not used

        Returns:
            Tuple of (selected_indices, uncertainties):
                - selected_indices: List of selected indices with small margins and high diversity
                - uncertainties: Normalized uncertainty scores (1 - margin) for all unlabeled samples [0, 1]
        """
        if predictions is None:
            raise ValueError("MarginDiversitySampling requires predictions")
        if embeddings is None:
            raise ValueError("MarginDiversitySampling requires embeddings")

        import logging
        logger = logging.getLogger(__name__)

        unlabeled_preds = predictions[unlabeled_indices]
        unlabeled_embeddings = embeddings[unlabeled_indices]

        # Step 1: Compute margin-based uncertainties
        sorted_preds = np.sort(unlabeled_preds, axis=1)
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]
        uncertainties = 1.0 - margins

        if density:
            uncertainties = uncertainties * densityEstimation(unlabeled_embeddings)

        logger.info(f"MarginDiversitySampling - margins min: {margins.min():.4f}, max: {margins.max():.4f}")
        logger.info(f"MarginDiversitySampling - uncertainties min: {uncertainties.min():.4f}, max: {uncertainties.max():.4f}")

        # Step 2: Select candidate pool based on margin uncertainty
        n_samples = min(self.n_samples, len(unlabeled_indices))
        candidate_pool_size = min(
            int(n_samples * self.candidate_multiplier),
            len(unlabeled_indices)
        )

        # Get indices of top uncertain samples (smallest margins)
        candidate_local_indices = np.argsort(margins)[:candidate_pool_size]
        candidate_embeddings = unlabeled_embeddings[candidate_local_indices]

        logger.info(f"MarginDiversitySampling - candidate pool size: {candidate_pool_size}, target samples: {n_samples}")

        # Step 3: Apply diversity selection on candidate pool
        selected_local_indices = self._greedy_diversity_selection(
            candidate_embeddings,
            candidate_local_indices,
            n_samples
        )

        # Convert local indices (within unlabeled set) to global indices
        selected = np.array(unlabeled_indices)[selected_local_indices].tolist()

        logger.info(f"MarginDiversitySampling - selected {len(selected)} diverse samples from {candidate_pool_size} candidates")

        return selected, uncertainties
