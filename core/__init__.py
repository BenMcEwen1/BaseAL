"""
Core active learning module
"""
from .utils.model import EmbeddingClassifier
from .active_learner import ActiveLearner

__all__ = ["EmbeddingClassifier", "ActiveLearner"]
