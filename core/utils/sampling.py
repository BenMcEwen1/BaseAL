"""
Sampling strategies for active learning
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Tuple


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
