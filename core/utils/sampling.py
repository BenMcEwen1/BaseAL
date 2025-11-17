"""
Sampling strategies for active learning
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional


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
               embeddings: Optional[np.ndarray] = None, model = None) -> List[int]:
        """
        Select samples for annotation

        Args:
            unlabeled_indices: List/array of unlabeled sample indices
            predictions: Optional numpy array of model predictions (N x num_classes)
            embeddings: Optional numpy array of embeddings
            model: Optional reference to the model itself

        Returns:
            List of selected indices
        """
        pass


class RandomSampling(SamplingStrategy):
    """Random sampling strategy - only needs indices"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> List[int]:
        """
        Randomly select samples from unlabeled pool

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Not used
            embeddings: Not used
            model: Not used

        Returns:
            List of randomly selected indices
        """
        n_samples = min(self.n_samples, len(unlabeled_indices))
        return np.random.choice(unlabeled_indices, size=n_samples, replace=False).tolist()


class EntropySampling(SamplingStrategy):
    """Entropy-based sampling - selects samples with highest prediction entropy"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> List[int]:
        """
        Select samples with highest prediction entropy

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            List of selected indices with highest entropy
        """
        if predictions is None:
            raise ValueError("EntropySampling requires predictions")

        # Calculate entropy for unlabeled samples only
        unlabeled_preds = predictions[unlabeled_indices]

        # Compute entropy: -sum(p * log(p))
        epsilon = 1e-10  # Avoid log(0)
        entropy = -np.sum(unlabeled_preds * np.log(unlabeled_preds + epsilon), axis=1)

        # Select samples with highest entropy
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(entropy)[-n_samples:]

        return np.array(unlabeled_indices)[top_indices].tolist()


class UncertaintySampling(SamplingStrategy):
    """Uncertainty sampling - selects samples with lowest maximum probability (least confident)"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> List[int]:
        """
        Select samples with lowest confidence (lowest max probability)

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            List of selected indices with lowest confidence
        """
        if predictions is None:
            raise ValueError("UncertaintySampling requires predictions")

        unlabeled_preds = predictions[unlabeled_indices]
        max_probs = np.max(unlabeled_preds, axis=1)

        # Select samples with lowest maximum probability (most uncertain)
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(max_probs)[:n_samples]  # Lowest confidence first

        return np.array(unlabeled_indices)[top_indices].tolist()


class MarginSampling(SamplingStrategy):
    """Margin sampling - selects samples with smallest margin between top two predictions"""

    def select(self, unlabeled_indices: List[int], predictions: Optional[np.ndarray] = None,
               embeddings: Optional[np.ndarray] = None, model = None) -> List[int]:
        """
        Select samples with smallest margin between top two class probabilities

        Args:
            unlabeled_indices: List of unlabeled sample indices
            predictions: Model predictions (N x num_classes) - REQUIRED
            embeddings: Not used
            model: Not used

        Returns:
            List of selected indices with smallest margins
        """
        if predictions is None:
            raise ValueError("MarginSampling requires predictions")

        unlabeled_preds = predictions[unlabeled_indices]

        # Sort predictions for each sample to get top 2
        sorted_preds = np.sort(unlabeled_preds, axis=1)

        # Calculate margin: difference between top two predictions
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]

        # Select samples with smallest margins (most ambiguous)
        n_samples = min(self.n_samples, len(unlabeled_indices))
        top_indices = np.argsort(margins)[:n_samples]  # Smallest margins first

        return np.array(unlabeled_indices)[top_indices].tolist()
