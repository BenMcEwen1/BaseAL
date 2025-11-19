"""
Simple active learning model for embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingClassifier(nn.Module):
    """
    Two-layer classifier: linear projection + classification head

    Args:
        input_dim: Dimension of input embeddings (e.g., 1024)
        hidden_dim: Dimension of intermediate embedding (default: same as input)
        num_classes: Number of output classes
    """

    def __init__(self, input_dim=1024, hidden_dim=None, num_classes=23):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Linear projection layer (generates new embedding)
        self.projection = nn.Linear(input_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_embedding=False):
        """
        Forward pass

        Args:
            x: Input embeddings of shape (batch_size, input_dim)
            return_embedding: If True, return both logits and intermediate embedding

        Returns:
            logits: Class logits of shape (batch_size, num_classes)
            embedding: (optional) Intermediate embedding of shape (batch_size, hidden_dim)
        """
        # Project to intermediate embedding space
        embedding = self.projection(x)
        embedding = F.relu(embedding)

        # Classification
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        return logits

    def get_embedding(self, x):
        """
        Get only the intermediate embedding

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            embedding: Intermediate embedding of shape (batch_size, hidden_dim)
        """
        with torch.no_grad():
            embedding = self.projection(x)
            embedding = F.relu(embedding)
        return embedding
