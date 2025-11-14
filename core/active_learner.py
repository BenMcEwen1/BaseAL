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
import warnings
import os
import umap

# Suppress numba warnings and debug output
warnings.filterwarnings('ignore', module='numba')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

# Set numba logging to WARNING to avoid verbose compilation details
logging.getLogger('numba').setLevel(logging.WARNING)

from .model import EmbeddingClassifier

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pre-warm UMAP at module import time to trigger numba JIT compilation
def _prewarm_umap_module():
    """
    Pre-warm UMAP by running a small dummy fit to trigger numba JIT compilation.
    This runs once at module import time, not during ActiveLearner initialization.
    """
    try:
        logger.info("Pre-warming UMAP (triggering numba JIT compilation)...")

        # Create small dummy dataset
        dummy_data = np.random.randn(100, 10).astype(np.float32)

        # Run a quick UMAP fit with parameters similar to what we'll use
        dummy_reducer = umap.UMAP(
            n_neighbors=10,
            n_components=3,
            metric="euclidean",
            low_memory=True,
            verbose=False
        )
        _ = dummy_reducer.fit_transform(dummy_data)

        logger.info("UMAP pre-warming complete")
    except Exception as e:
        logger.warning(f"UMAP pre-warming failed (non-critical): {e}")


# Execute pre-warming immediately at module import
_prewarm_umap_module()


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
        learning_rate: float = 0.0001,
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
        import sys
        print("="*50, file=sys.stderr)
        print("ACTIVE LEARNER INIT CALLED", file=sys.stderr)
        print("="*50, file=sys.stderr)
        sys.stderr.flush()
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

        # PCA transformation (fitted once and reused)
        self.pca = None
        self.scaler = None

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
        import sys
        print(f"Loading annotations from: {self.annotations_path}", file=sys.stderr)
        sys.stderr.flush()
        df = pd.read_csv(self.annotations_path)
        print(f"Loaded {len(df)} annotations", file=sys.stderr)
        sys.stderr.flush()

        # Extract unique labels and create mappings
        # Convert all labels to strings to ensure JSON serialization compatibility
        unique_labels = sorted(df['label:default_classifier'].unique())
        unique_labels = [str(label) for label in unique_labels]
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Load embeddings and match with annotations
        embeddings_list = []
        labels_list = []

        # Map audio filenames to embeddings
        for _, row in df.iterrows():
            audio_filename = row['audiofilename']
            label = str(row['label:default_classifier'])

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

        # Prepare labeled data (keep original for evaluation)
        X_train_orig = torch.from_numpy(self.embeddings[self.labeled_indices]).to(self.device)
        y_train_orig = torch.from_numpy(self.labels[self.labeled_indices]).to(self.device)

        # Training loop
        total_loss = 0.0

        for epoch in range(epochs):
            # Shuffle data for this epoch
            perm = torch.randperm(len(X_train_orig))
            X_train_shuffled = X_train_orig[perm]
            y_train_shuffled = y_train_orig[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for i in range(0, len(X_train_orig), batch_size):
                batch_X = X_train_shuffled[i:i + batch_size]
                batch_y = y_train_shuffled[i:i + batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            total_loss = epoch_loss

        # Calculate final accuracy on labeled set (using ORIGINAL unshuffled order)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.from_numpy(self.embeddings).to(self.device))
            
            _, predicted = torch.max(outputs, 1)
            # print("Model predicted", predicted.shape, predicted)

            labels = torch.from_numpy(self.labels).to(self.device)
            # print("label", labels.shape, labels)

            correct = (predicted == labels).sum().item()
            total = len(labels)

        print(correct, total)
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(X_train_orig) // batch_size)

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
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.model.eval()

        with torch.no_grad():
            X = torch.from_numpy(self.embeddings).to(self.device)
            embeddings = self.model.get_embedding(X).cpu().numpy()

        # Fit PCA transformation on first call, then reuse it
        if self.pca is None or self.scaler is None:
            logger.info("Fitting PCA transformation (will be reused for all subsequent calls)")
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)

            self.pca = PCA(n_components=3)
            embeddings_3d = self.pca.fit_transform(embeddings_scaled)
        else:
            print("PCA applied here")
            # Reuse the fitted transformation
            # embeddings_scaled = self.scaler.transform(embeddings)

            umap_config = {
                "n_neighbors": 10,
                "min_dist": 0.1,
                "n_components": 3,
                "metric": "euclidean",
                "random_state": 42,
                "low_memory": True,
                "n_epochs": 100,
                "init": "spectral",
            }

            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)

            n_subset = 1000
            subset_indices = np.random.choice(len(embeddings), n_subset, replace=False)
            embeddings_subset = embeddings_scaled[subset_indices]

            reducer = umap.UMAP(**umap_config)
            reducer.fit(embeddings_subset)
            embeddings_3d = reducer.transform(embeddings_scaled)
            print(embeddings_3d.shape)

            # self.pca = PCA(n_components=3)
            # embeddings_3d = self.pca.fit_transform(embeddings_scaled)
            # embeddings_3d = self.pca.transform(embeddings_scaled)
            # print(embeddings_3d.shape)

        return embeddings_3d

    def get_state(self) -> Dict:
        """
        Get current state of the active learner

        Returns:
            Dictionary with current state
        """
        return {
            "n_labeled": int(len(self.labeled_indices)),
            "n_unlabeled": int(len(self.unlabeled_indices)),
            "labeled_indices": [int(idx) for idx in self.labeled_indices],
            "unlabeled_indices": [int(idx) for idx in self.unlabeled_indices],
            "training_history": self.training_history,
            "num_classes": int(len(self.label_to_idx)),
            "labels": list(self.label_to_idx.keys())  # Already strings from initialization
        }
