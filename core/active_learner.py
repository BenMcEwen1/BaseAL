"""
Active learning pipeline for embeddings
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

from .model import EmbeddingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearner:
    """
    Active learning pipeline for embedding classification
    """

    def __init__(
        self,
        embeddings_dir: Path,
        annotations_path: Path,
        model_name: str = "birdnet",
        dataset_name: str = "FewShot",
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        """
        Initialize active learner

        Args:
            embeddings_dir: Path to embeddings directory
            annotations_path: Path to annotations CSV
            model_name: Name of the model (e.g., 'birdnet')
            dataset_name: Name of the dataset (e.g., 'FewShot')
            hidden_dim: Dimension of intermediate embedding
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu' or 'cuda')
        """
        self.embeddings_dir = embeddings_dir
        self.annotations_path = annotations_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.device = device

        # Load data
        self.embeddings, self.labels, self.label_to_idx, self.idx_to_label = self._load_data()

        # Initialize model
        input_dim = self.embeddings.shape[1]
        num_classes = len(self.label_to_idx)
        self.model = EmbeddingClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Active learning state
        self.labeled_indices = []
        self.unlabeled_indices = list(range(len(self.embeddings)))
        self.training_history = []

        logger.info(f"Initialized ActiveLearner with {len(self.embeddings)} samples and {num_classes} classes")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load embeddings and annotations

        Returns:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Array of shape (n_samples,) with integer labels
            label_to_idx: Dictionary mapping label names to indices
            idx_to_label: Dictionary mapping indices to label names
        """
        # Load annotations
        df = pd.read_csv(self.annotations_path)

        # Extract unique labels and create mappings
        unique_labels = sorted(df['label:default_classifier'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Load embeddings and match with annotations
        embeddings_list = []
        labels_list = []

        # Map audio filenames to embeddings
        for _, row in df.iterrows():
            audio_filename = row['audiofilename']
            label = row['label:default_classifier']

            # Construct embedding filename
            # Convert audio\FewShot\CHE_01_20190101_163410.wav -> CHE_01_20190101_163410_birdnet.npy
            filename_parts = Path(audio_filename).stem
            embedding_filename = f"{filename_parts}_{self.model_name}.npy"

            embedding_path = self.embeddings_dir / embedding_filename

            if embedding_path.exists():
                # Load embedding for this segment
                emb = np.load(embedding_path)

                # Calculate which segment this row corresponds to
                start_time = row['start']
                segment_duration = 3.0  # Assuming 3-second segments
                segment_idx = int(start_time / segment_duration)

                if segment_idx < len(emb):
                    embeddings_list.append(emb[segment_idx])
                    labels_list.append(label_to_idx[label])

        embeddings = np.array(embeddings_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int64)

        logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")

        return embeddings, labels, label_to_idx, idx_to_label

    def sample_random(self, n_samples: int = 5) -> List[int]:
        """
        Random sampling strategy

        Args:
            n_samples: Number of samples to select

        Returns:
            List of selected indices
        """
        if len(self.unlabeled_indices) == 0:
            logger.warning("No unlabeled samples remaining")
            return []

        n_samples = min(n_samples, len(self.unlabeled_indices))
        selected = np.random.choice(self.unlabeled_indices, size=n_samples, replace=False)
        return selected.tolist()

    def add_samples(self, indices: List[int]):
        """
        Add samples to the labeled set

        Args:
            indices: List of indices to add to labeled set
        """
        for idx in indices:
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
                self.labeled_indices.append(idx)

        logger.info(f"Added {len(indices)} samples. Labeled: {len(self.labeled_indices)}, Unlabeled: {len(self.unlabeled_indices)}")

    def train_step(self, epochs: int = 10, batch_size: int = 8) -> Dict:
        """
        Train the model on the current labeled set

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Dictionary with training metrics
        """
        if len(self.labeled_indices) == 0:
            logger.warning("No labeled samples to train on")
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.train()

        # Prepare labeled data
        X_train = torch.from_numpy(self.embeddings[self.labeled_indices]).to(self.device)
        y_train = torch.from_numpy(self.labels[self.labeled_indices]).to(self.device)

        # Training loop
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            # Shuffle data
            perm = torch.randperm(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            total_loss = epoch_loss

        # Calculate final accuracy on labeled set
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_train)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_train).sum().item()
            total = len(y_train)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(X_train) // batch_size)

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "n_labeled": len(self.labeled_indices),
            "n_unlabeled": len(self.unlabeled_indices)
        }

        self.training_history.append(metrics)
        logger.info(f"Training step complete: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return metrics

    def get_embeddings_3d(self, reduction_method: str = "pca") -> np.ndarray:
        """
        Get 3D embeddings from the intermediate layer

        Args:
            reduction_method: Method for dimension reduction ('pca')

        Returns:
            Array of shape (n_samples, 3) with 3D coordinates
        """
        self.model.eval()

        with torch.no_grad():
            X = torch.from_numpy(self.embeddings).to(self.device)
            embeddings = self.model.get_embedding(X).cpu().numpy()

        # Reduce to 3D using PCA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings_scaled)

        return embeddings_3d

    def get_state(self) -> Dict:
        """
        Get current state of the active learner

        Returns:
            Dictionary with current state
        """
        return {
            "n_labeled": len(self.labeled_indices),
            "n_unlabeled": len(self.unlabeled_indices),
            "labeled_indices": self.labeled_indices,
            "unlabeled_indices": self.unlabeled_indices,
            "training_history": self.training_history,
            "num_classes": len(self.label_to_idx),
            "labels": list(self.label_to_idx.keys())
        }
