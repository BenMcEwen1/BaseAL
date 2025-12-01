"""
Sampling strategies for active learning
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import logging

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


class SamplingStrategy:
    """
    Unified sampling strategy class that handles all sampling methods.

    This class contains the selection logic and various sampling methods.
    Data is stored as instance attributes and accessed by sampling methods.
    """

    def __init__(self, method: str = "random", n_samples: int = 20):
        """
        Initialize sampling strategy

        Args:
            method: Sampling method to use ('random', 'margin', 'custom')
            n_samples: Number of samples to select per iteration
        """
        self.method = method
        self.n_samples = n_samples

        # Available sampling methods
        available_methods = ['random', 'margin', 'custom']

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
        }

        # Data attributes (see selct)
        self.unlabeled_indices = None
        self.predictions = None
        self.embeddings = None
        self.model = None
        self.annotations = None

        logger.info(f"Initialized SamplingStrategy with method='{method}' and n_samples={n_samples}")

    def select(self,
               unlabeled_indices: List[int],
               predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None,
               model = None,
               annotations: Optional[pd.DataFrame] = None) -> Tuple[List[int], np.ndarray]:
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

        # Call the appropriate sampling method to get uncertainties
        sampling_func = self._method_map[self.method]
        uncertainties = sampling_func()

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
        # All samples have equal uncertainty for random sampling
        utility = np.ones(len(self.unlabeled_indices))
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

        2. The uncertainty scores should be normalized to [0, 1] where:
           - 1.0 = maximum uncertainty (highest priority for annotation)
           - 0.0 = complete certainty (lowest priority for annotation)

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
