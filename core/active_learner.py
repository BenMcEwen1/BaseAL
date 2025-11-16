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
import time
import yaml
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        start = time.time()
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
        end = time.time()

        logger.info(f"UMAP pre-warming completed in {end - start}")
    except Exception as e:
        logger.warning(f"UMAP pre-warming failed (non-critical): {e}")


# Execute pre-warming immediately at module import
_prewarm_umap_module()


class Manager:
    """
    Manages multiple parallel Active Learning experiments
    """

    def __init__(self, config_path: Path):
        """
        Initialize Manager with experiments from config file

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.configs = self._load_configs(self.config_path)
        self.experiments = []
        self.experiment_names = []
        self.__initialize_experiments()
        logger.info(f"Manager initialized with {len(self.experiments)} experiments")

    def _load_configs(self, path: Path) -> List[Dict]:
        """
        Load experiment configurations from YAML file

        Args:
            path: Path to YAML config file

        Returns:
            List of configuration dictionaries
        """
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)

        if 'experiments' not in config_data:
            raise ValueError("Config file must contain 'experiments' key")

        experiments = config_data['experiments']

        # Convert string paths to Path objects
        for exp in experiments:
            if 'embeddings_dir' in exp:
                exp['embeddings_dir'] = Path(exp['embeddings_dir'])
            if 'annotations_path' in exp:
                exp['annotations_path'] = Path(exp['annotations_path'])

        logger.info(f"Loaded {len(experiments)} experiment configurations from {path}")
        return experiments

    def __initialize_experiments(self):
        """Initialize ActiveLearner instances for each experiment config"""
        for i, config in enumerate(self.configs):
            # Extract experiment name if provided, otherwise use index
            exp_name = config.pop('name', f'experiment_{i}')
            self.experiment_names.append(exp_name)

            logger.info(f"Initializing experiment: {exp_name}")
            learner = ActiveLearner(**config)
            self.experiments.append(learner)

    def run(self,
            n_samples: int = 5,
            epochs: int = 5,
            batch_size: int = 8,
            parallel: bool = False) -> Dict[str, Dict]:
        """
        Run one complete AL cycle for all experiments

        Args:
            n_samples: Number of samples to select per experiment
            epochs: Number of training epochs
            batch_size: Training batch size
            parallel: Whether to run experiments in parallel

        Returns:
            Dictionary mapping experiment names to their training metrics
        """
        logger.info(f"Starting AL cycle: {n_samples} samples, {epochs} epochs, parallel={parallel}")

        results = {}

        if parallel:
            # Run experiments in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(self.experiments)) as executor:
                future_to_name = {
                    executor.submit(self._run_single_experiment, learner, n_samples, epochs, batch_size): name
                    for learner, name in zip(self.experiments, self.experiment_names)
                }

                for future in as_completed(future_to_name):
                    exp_name = future_to_name[future]
                    try:
                        metrics = future.result()
                        results[exp_name] = metrics
                        logger.info(f"Experiment '{exp_name}' completed: {metrics}")
                    except Exception as e:
                        logger.error(f"Experiment '{exp_name}' failed: {e}")
                        results[exp_name] = {"error": str(e)}
        else:
            # Run experiments sequentially
            for learner, exp_name in zip(self.experiments, self.experiment_names):
                try:
                    metrics = self._run_single_experiment(learner, n_samples, epochs, batch_size)
                    results[exp_name] = metrics
                    logger.info(f"Experiment '{exp_name}' completed: {metrics}")
                except Exception as e:
                    logger.error(f"Experiment '{exp_name}' failed: {e}")
                    results[exp_name] = {"error": str(e)}

        return results

    def _run_single_experiment(self,
                               learner: 'ActiveLearner',
                               n_samples: int,
                               epochs: int,
                               batch_size: int) -> Dict:
        """
        Run one AL cycle for a single experiment

        Args:
            learner: ActiveLearner instance
            n_samples: Number of samples to select
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics dictionary
        """
        # Sample new data points
        selected_indices = learner.sample_random(n_samples)

        if len(selected_indices) > 0:
            # Add selected samples to labeled set
            learner.add_samples(selected_indices)

            # Train on updated labeled set
            metrics = learner.train_step(epochs=epochs, batch_size=batch_size)
        else:
            # No samples to add, just return current state
            metrics = {
                "loss": 0.0,
                "accuracy": 0.0,
                "n_labeled": len(learner.labeled_indices),
                "n_unlabeled": len(learner.unlabeled_indices)
            }

        return metrics

    def add(self, new_config: Dict, name: Optional[str] = None):
        """
        Add a new experiment dynamically

        Args:
            new_config: Configuration dictionary for new experiment
            name: Optional name for the experiment
        """
        # Convert string paths to Path objects
        if 'embeddings_dir' in new_config:
            new_config['embeddings_dir'] = Path(new_config['embeddings_dir'])
        if 'annotations_path' in new_config:
            new_config['annotations_path'] = Path(new_config['annotations_path'])

        exp_name = name or f'experiment_{len(self.experiments)}'
        self.experiment_names.append(exp_name)

        logger.info(f"Adding new experiment: {exp_name}")
        learner = ActiveLearner(**new_config)
        self.experiments.append(learner)

    def save(self, output_dir: Optional[Path] = None):
        """
        Save training histories and experiment states to JSON files

        Args:
            output_dir: Directory to save results (defaults to './results')
        """
        if output_dir is None:
            output_dir = Path('./results')

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save each experiment's history
        for learner, exp_name in zip(self.experiments, self.experiment_names):
            # Create experiment-specific results
            results = {
                'experiment_name': exp_name,
                'timestamp': timestamp,
                'config': {
                    'model_name': learner.model_name,
                    'dataset_name': learner.dataset_name,
                    'learning_rate': learner.learning_rate,
                    'device': learner.device,
                },
                'final_state': learner.get_state(),
                'training_history': learner.training_history
            }

            # Save to JSON file
            output_file = output_dir / f'{exp_name}_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved results for '{exp_name}' to {output_file}")

        # Save combined summary
        summary = {
            'timestamp': timestamp,
            'num_experiments': len(self.experiments),
            'experiment_names': self.experiment_names,
            'experiments': [
                {
                    'name': name,
                    'n_labeled': len(learner.labeled_indices),
                    'n_unlabeled': len(learner.unlabeled_indices),
                    'final_accuracy': learner.training_history[-1]['accuracy'] if learner.training_history else 0.0,
                    'num_iterations': len(learner.training_history)
                }
                for learner, name in zip(self.experiments, self.experiment_names)
            ]
        }

        summary_file = output_dir / f'summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved experiment summary to {summary_file}")

    def get_summary(self) -> Dict:
        """
        Get current status of all experiments

        Returns:
            Dictionary with summary information for all experiments
        """
        summary = {
            'num_experiments': len(self.experiments),
            'experiments': []
        }

        for learner, name in zip(self.experiments, self.experiment_names):
            exp_summary = {
                'name': name,
                'n_labeled': len(learner.labeled_indices),
                'n_unlabeled': len(learner.unlabeled_indices),
                'num_iterations': len(learner.training_history),
                'current_accuracy': learner.training_history[-1]['accuracy'] if learner.training_history else 0.0,
                'learning_rate': learner.learning_rate,
                'model_name': learner.model_name
            }
            summary['experiments'].append(exp_summary)

        return summary

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

        self.dim_reduction_method = "UMAP"
        self.umap_transform_batch_size = 500
        self.idx = None

        self.umap_config = {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 3,
                "n_epochs": 200,
                "init": "spectral",
                "n_jobs": 1
            }

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

        # Dimensionality reduction (fitted once and reused)
        self.reducer = None
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

    def train_step(self, epochs: int = 5, batch_size: int = 8) -> Dict:
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

            train_length = X_train_shuffled.shape[0]

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

            total_loss = epoch_loss / train_length

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

        # print(correct, total)
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
    
    def _transform_batched(self, embeddings_scaled: np.ndarray) -> np.ndarray:
        """
        Transform embeddings in batches to improve performance for large datasets.
        Args:
            embeddings_scaled: Scaled embeddings to transform

        Returns:
            Transformed 3D embeddings
        """
        n_samples = len(embeddings_scaled)

        # If dataset is small, no need for batching
        if n_samples <= self.umap_transform_batch_size:
            return self.reducer.transform(embeddings_scaled)

        # Split into batches and transform
        logger.info(f"Transforming {n_samples} samples in batches of {self.umap_transform_batch_size}")

        batches = []
        for start_idx in range(0, n_samples, self.umap_transform_batch_size):
            start = time.time()
            end_idx = min(start_idx + self.umap_transform_batch_size, n_samples)
            batch = embeddings_scaled[start_idx:end_idx]

            # Transform this batch
            batch_transformed = self.reducer.transform(batch)
            batches.append(batch_transformed)
            end = time.time()

            logger.info(f"Transformed batch {start_idx//self.umap_transform_batch_size + 1}/{(n_samples-1)//self.umap_transform_batch_size + 1} ({end_idx}/{n_samples} samples in {end - start}s)")

        # Combine all batches
        embeddings_3d = np.vstack(batches)
        logger.info(f"Batch transformation complete: {embeddings_3d.shape}")

        return embeddings_3d

    def get_embeddings_3d(self, reduction_method: str = "pca", max_embeddings: int = 1000) -> np.ndarray:
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

        # Subsampling and plot embeddings
        if embeddings.shape[0] > max_embeddings:
            if self.idx is None:
                print("Generate subset...")
                self.idx = np.random.choice(embeddings.shape[0], size=max_embeddings, replace=False)
            
            embeddings = embeddings[self.idx]
            print(f"Embeddings subsampled, new shape {embeddings.shape}")

        # Fit transformation on first call, then reuse
        if self.reducer is None or self.scaler is None:
            logger.info(f"Fitting {self.dim_reduction_method} (will be reused for subsequent calls)")
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)

            if self.dim_reduction_method == "PCA":
                self.reducer = PCA(n_components=3)
                embeddings_3d = self.reducer.fit_transform(embeddings_scaled)
            elif self.dim_reduction_method == "UMAP":
                self.reducer = umap.UMAP(**self.umap_config)
                start = time.time()
                embeddings_3d = self.reducer.fit_transform(embeddings_scaled)
                end = time.time()
                logger.info(f"UMAP fit completed in {end - start:.1f}s")
        else:
            # Transform using fitted transformation
            embeddings_scaled = self.scaler.transform(embeddings)
            start = time.time()
            embeddings_3d = self.reducer.transform(embeddings_scaled)
            end = time.time()
            logger.info(f"Transformed {len(embeddings)} samples using {self.dim_reduction_method} in {end - start:.3f}s")

        # Center the embeddings at the origin for better camera rotation
        embeddings_3d = embeddings_3d - embeddings_3d.mean(axis=0)
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
