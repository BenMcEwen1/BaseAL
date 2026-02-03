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
import copy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score

# Initialize sampling strategy
from .utils.sampling import SamplingStrategy

# Suppress numba warnings and debug output
warnings.filterwarnings('ignore', module='numba')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

# Set numba logging to WARNING to avoid verbose compilation details
logging.getLogger('numba').setLevel(logging.WARNING)

from .utils.model import EmbeddingClassifier

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

    def __init__(self, config_path: Path, base_dir: Optional[Path] = None):
        """
        Initialize Manager with experiments from config file

        Args:
            config_path: Path to YAML configuration file
            base_dir: Base directory for resolving relative paths in config (defaults to config file's parent)
        """
        self.config_path = Path(config_path)
        self.base_dir = Path(base_dir) if base_dir else self.config_path.parent
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

        # Convert string paths to Path objects and resolve relative to base_dir
        for exp in experiments:
            if 'embeddings_dir' in exp:
                emb_path = Path(exp['embeddings_dir'])
                # If relative path, resolve relative to base_dir
                if not emb_path.is_absolute():
                    exp['embeddings_dir'] = self.base_dir / emb_path
                else:
                    exp['embeddings_dir'] = emb_path

            if 'annotations_path' in exp:
                ann_path = Path(exp['annotations_path'])
                # If relative path, resolve relative to base_dir
                if not ann_path.is_absolute():
                    exp['annotations_path'] = self.base_dir / ann_path
                else:
                    exp['annotations_path'] = ann_path

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
        selected_indices = learner.sample(n_samples)

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
        repeats: int = 1,
        device: str = "cpu",
        sampling_strategy: str = "random",
        n_samples_per_iteration: int = 5,
        pretrain_samples: Optional[int] = None
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
            repeats: Number of training repeats for computing mean/std metrics
            device: Device to use ('cpu' or 'cuda')
            sampling_strategy: Sampling method to use ('random', 'margin', 'custom')
            n_samples_per_iteration: Default number of samples to select per iteration
            pretrain_samples: Number of high-density samples to pre-select for warm-up training (optional)
        """
        self.embeddings_dir = embeddings_dir
        self.annotations_path = annotations_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.device = device
        self.repeats = repeats

        # Audio directory for media retrieval (new format: {dataset}/data/{model_name}/)
        self.audio_dir = Path(annotations_path).parent / "data" / model_name

        self.dim_reduction_method = "UMAP"
        self.umap_transform_batch_size = 500
        self.idx = None

        self.umap_config = {
                "n_neighbors": 30,
                "min_dist": 0.1,
                "n_components": 3,
                "n_epochs": 200,
                "init": "spectral",
                "n_jobs": 1
            }

        # Load data
        # import sys
        # print("="*50, file=sys.stderr)
        # print("ACTIVE LEARNER INIT CALLED", file=sys.stderr)
        # print("="*50, file=sys.stderr)
        # sys.stderr.flush()
        self.embeddings, self.labels, self.label_to_idx, self.idx_to_label, self.annotations_df = self._load_data()

        # Detect if dataset is multilabel based on label shape
        # Single-label: (n_samples,) with dtype int64
        # Multilabel: (n_samples, num_classes) with dtype float32
        self.is_multilabel = len(self.labels.shape) == 2
        logger.info(f"Dataset mode: {'MULTILABEL' if self.is_multilabel else 'SINGLE-LABEL'}")

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

        # Loss function - select based on single-label vs multilabel
        if self.is_multilabel:
            # Multilabel: Binary Cross-Entropy with Logits
            # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss for multilabel classification")
        else:
            # Single-label: Cross-Entropy Loss
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using CrossEntropyLoss for single-label classification")

        self.sampling_strategy = SamplingStrategy(method=sampling_strategy, n_samples=n_samples_per_iteration)
        logger.info(f"Initialized '{sampling_strategy}' sampling strategy with n_samples={n_samples_per_iteration}")

        # Active learning state
        self.labeled_indices = []
        self.unlabeled_indices = list(range(len(self.embeddings)))
        self.training_history = []

        # Pre-training warm-up: select high-density samples if specified
        if pretrain_samples is not None and pretrain_samples > 0:
            self._pretrain_warmup(pretrain_samples)
            logger.info(f"Pre-training warm-up: selected {len(self.labeled_indices)} high-density samples")

        # Per-sample uncertainties (updated after each sampling step)
        # Initialize with zeros for all samples
        self.uncertainties = np.zeros(len(self.embeddings))

        # Dimensionality reduction (fitted once and reused)
        self.reducer = None
        self.scaler = None

        logger.info(f"Initialised ActiveLearner with {len(self.embeddings)} samples and {num_classes} classes")

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, pd.DataFrame]:
        """
        Load embeddings and annotations

        Supports both single-label and multilabel formats:
        - Single-label: '5' (integer)
        - Multilabel: '5;12;18' (semicolon-separated integers)

        Returns:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: Array of shape (n_samples, num_classes) for multilabel or (n_samples,) for single-label
            label_to_idx: Dictionary mapping label names to indices
            idx_to_label: Dictionary mapping indices to label names
            annotations_df: DataFrame containing the matched annotations with metadata
        """
        # Load annotations
        import sys
        print(f"Loading annotations from: {self.annotations_path}", file=sys.stderr)
        sys.stderr.flush()
        df = pd.read_csv(self.annotations_path)
        print(f"Loaded {len(df)} annotations", file=sys.stderr)
        sys.stderr.flush()

        # Detect if data is multilabel by checking for semicolons in labels
        label_column = 'label'
        sample_labels = df[label_column].astype(str) #.head(100)
        is_multilabel = sample_labels.str.contains(';').any()

        if is_multilabel:
            logger.info("Detected MULTILABEL format (semicolon-separated labels)")
        else:
            logger.info("Detected SINGLE-LABEL format")

        # Extract all unique labels across dataset
        all_labels = set()
        for label_str in df[label_column].astype(str):
            if ';' in label_str:
                # Multilabel case: split by semicolon
                labels_in_row = label_str.split(';')
                all_labels.update(labels_in_row)
            else:
                # Single label case
                all_labels.add(label_str)

        # Create mappings
        unique_labels = sorted(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        num_classes = len(unique_labels)

        logger.info(f"Found {num_classes} unique classes: {unique_labels}")

        # Load embeddings and match with annotations
        embeddings_list = []
        labels_list = []
        annotations_list = []  # Track matched annotation rows

        # Map audio filenames to embeddings
        for _, row in df.iterrows():
            audio_filename = row['filename']
            label_str = str(row[label_column])

            # Construct embedding filename
            # Convert audio\FewShot\CHE_01_20190101_163410.wav -> CHE_01_20190101_163410_birdnet.npy
            filename_parts = Path(audio_filename).stem
            embedding_filename = f"{filename_parts}_{self.model_name}.npy"

            embedding_path = self.embeddings_dir / embedding_filename

            if embedding_path.exists():
                # Load embedding - new format: each file is a single segment embedding
                emb = np.load(embedding_path)

                # Handle both 1D (single embedding) and 2D (batch of 1) shapes
                if emb.ndim == 1:
                    embeddings_list.append(emb)
                else:
                    embeddings_list.append(emb[0] if len(emb) == 1 else emb.flatten())

                # Parse labels based on format
                if ';' in label_str:
                    # Multilabel: create binary vector
                    label_vector = np.zeros(num_classes, dtype=np.float32)
                    for lbl in label_str.split(';'):
                        lbl = lbl.strip()
                        if lbl in label_to_idx:
                            label_vector[label_to_idx[lbl]] = 1.0
                    labels_list.append(label_vector)
                else:
                    # Single-label: use integer index
                    if is_multilabel:
                        # If dataset is multilabel but this sample has single label,
                        # still use binary vector format for consistency
                        label_vector = np.zeros(num_classes, dtype=np.float32)
                        label_vector[label_to_idx[label_str]] = 1.0
                        labels_list.append(label_vector)
                    else:
                        # Pure single-label dataset
                        labels_list.append(label_to_idx[label_str])

                # Store the annotation row for this sample
                annotations_list.append(row)

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Create DataFrame from matched annotations
        annotations_df = pd.DataFrame(annotations_list).reset_index(drop=True)

        if is_multilabel:
            # Multilabel: shape (n_samples, num_classes)
            labels = np.array(labels_list, dtype=np.float32)
            logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
            logger.info(f"Labels shape: {labels.shape} (multilabel binary vectors)")
        else:
            # Single-label: shape (n_samples,)
            labels = np.array(labels_list, dtype=np.int64)
            logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
            logger.info(f"Labels shape: {labels.shape} (single-label indices)")

        logger.info(f"Annotations DataFrame shape: {annotations_df.shape}")

        return embeddings, labels, label_to_idx, idx_to_label, annotations_df

    def _pretrain_warmup(self, n_samples: int):
        """
        Pre-training warm-up: select high-density samples for initial training

        This method selects samples with lots of neighbors (high density) for
        initial annotation and training, without requiring model predictions.

        Args:
            n_samples: Number of high-density samples to pre-select
        """
        from .utils.sampling import densityEstimation

        # Ensure we don't request more samples than available
        n_samples = min(n_samples, len(self.embeddings))

        logger.info(f"Computing density estimation for {len(self.embeddings)} samples...")

        # Compute density using KNN method (samples with more neighbors = higher density)
        # Using k=20 as a reasonable default for neighbor count
        density_scores = densityEstimation(
            embeddings=self.embeddings,
            method='knn',
            k=min(20, len(self.embeddings) - 1),  # Ensure k < n_samples
            beta=1
        )

        logger.info(f"Density scores - min: {density_scores.min():.4f}, max: {density_scores.max():.4f}, mean: {density_scores.mean():.4f}")

        # Select samples with highest density
        # Higher density = more neighbors = more representative samples
        top_density_indices = np.argsort(density_scores)[-n_samples:]

        # Add these samples to labeled set
        self.labeled_indices = top_density_indices.tolist()

        # Remove from unlabeled set
        self.unlabeled_indices = [idx for idx in range(len(self.embeddings))
                                  if idx not in self.labeled_indices]

        logger.info(f"Pre-training warm-up complete: selected {len(self.labeled_indices)} high-density samples")
        logger.info(f"Remaining unlabeled samples: {len(self.unlabeled_indices)}")

    def sample(self, n_samples: Optional[int] = None) -> List[int]:
        """
        Sample unlabeled data points using the configured sampling strategy
        Also updates per-sample uncertainty scores for visualization

        Args:
            n_samples: Number of samples to select (overrides default if provided)

        Returns:
            List of selected indices
        """
        if len(self.unlabeled_indices) == 0:
            logger.warning("No unlabeled samples remaining")
            return []

        # Override n_samples if provided
        if n_samples is not None:
            original_n = self.sampling_strategy.n_samples
            self.sampling_strategy.n_samples = n_samples

        # Get predictions if model has been trained
        predictions = None
        # if len(self.labeled_indices) > 0:  # Only get predictions if model is trained
        self.model.eval()
        with torch.no_grad():
            X_all = torch.from_numpy(self.embeddings).to(self.device)
            outputs = self.model(X_all)

            # Use appropriate activation based on single-label vs multilabel
            if self.is_multilabel:
                # Multilabel: use sigmoid for independent class probabilities
                predictions = torch.sigmoid(outputs).cpu().numpy()
            else:
                # Single-label: use softmax for mutually exclusive classes
                predictions = torch.softmax(outputs, dim=1).cpu().numpy()

        # Call the sampling strategy with all available data
        # Now returns both selected indices and uncertainties
        selected, unlabeled_uncertainties = self.sampling_strategy.select(
            unlabeled_indices=self.unlabeled_indices,
            predictions=predictions,
            embeddings=self.embeddings,
            model=self.model,
            annotations=self.annotations_df
        )

        # Debug: Check uncertainty values from sampling strategy
        logger.info(f"Unlabeled uncertainties - min: {unlabeled_uncertainties.min():.4f}, max: {unlabeled_uncertainties.max():.4f}, mean: {unlabeled_uncertainties.mean():.4f}")
        logger.info(f"Unlabeled uncertainties shape: {unlabeled_uncertainties.shape}, expected: {len(self.unlabeled_indices)}")

        # Update uncertainties array for all samples
        # Labeled samples have uncertainty = 0
        self.uncertainties = np.zeros(len(self.embeddings))
        # Unlabeled samples have computed uncertainty
        self.uncertainties[self.unlabeled_indices] = unlabeled_uncertainties

        # Debug: Check final uncertainties
        logger.info(f"Final uncertainties - min: {self.uncertainties.min():.4f}, max: {self.uncertainties.max():.4f}")
        logger.info(f"Non-zero uncertainties: {np.count_nonzero(self.uncertainties)} out of {len(self.uncertainties)}")

        # Restore original n_samples if it was overridden
        if n_samples is not None:
            self.sampling_strategy.n_samples = original_n

        logger.info(f"Selected {len(selected)} samples using {self.sampling_strategy.__class__.__name__}")
        return selected

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

    def _calculate_calibration_metrics(self, probabilities: np.ndarray, predicted: np.ndarray,
                                       labels: np.ndarray, n_bins: int = 10) -> Dict:
        """
        Calculate calibration metrics for reliability plot

        Args:
            probabilities: Predicted probabilities for all classes (n_samples, n_classes)
            predicted: Predicted class labels (n_samples,)
            labels: True labels (n_samples,)
            n_bins: Number of bins for calibration plot

        Returns:
            Dictionary with calibration data for plotting
        """
        # Get confidence (max probability) for each prediction
        confidences = np.max(probabilities, axis=1)
        correct = (predicted == labels).astype(int)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Calculate accuracy and confidence per bin
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        ece = 0.0  # Expected Calibration Error

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                count_in_bin = in_bin.sum()

                bin_accuracies.append(float(accuracy_in_bin))
                bin_confidences.append(float(avg_confidence_in_bin))
                bin_counts.append(int(count_in_bin))

                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            else:
                bin_accuracies.append(None)
                bin_confidences.append(float((bin_lower + bin_upper) / 2))
                bin_counts.append(0)

        return {
            'bin_confidences': bin_confidences,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
            'ece': float(ece),
            'n_bins': n_bins
        }

    def train_step(self, epochs: int = 5, batch_size: int = 32) -> Dict:
        """
        Train the model on the current labeled set

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            repeats: Number of times to repeat training cycle for aggregating statistics

        Returns:
            Dictionary with training metrics including mean and SD
        """
        if len(self.labeled_indices) == 0:
            logger.warning("No labeled samples to train on")
            return {
                "loss": 0.0, "accuracy": 0.0, "f1_score": 0.0, "mAP": 0.0,
                "loss_sd": 0.0, "accuracy_sd": 0.0, "f1_score_sd": 0.0, "mAP_sd": 0.0
            }

        # Store original model state for repeats
        original_model_state = copy.deepcopy(self.model.state_dict())
        original_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # Collect metrics across repeats
        all_losses = []
        all_accuracies = []
        all_f1_scores = []
        all_mAPs = []

        for repeat_idx in range(self.repeats):
            if repeat_idx > 0:
                # Reset model and optimizer to original state for each repeat
                self.model.load_state_dict(copy.deepcopy(original_model_state))
                self.optimizer.load_state_dict(copy.deepcopy(original_optimizer_state))
                logger.info(f"Starting repeat {repeat_idx + 1}/{self.repeats}")

            self.model.train()

            # Prepare labeled data
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

                    # Calculate loss based on single-label vs multilabel
                    if self.is_multilabel:
                        # For BCEWithLogitsLoss, targets should be float32
                        loss = self.criterion(outputs, batch_y)
                    else:
                        # For CrossEntropyLoss, targets should be int64
                        loss = self.criterion(outputs, batch_y)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                total_loss = epoch_loss / train_length

            # Evaluation on all data
            self.model.eval()
            with torch.no_grad():
                # Get predictions for all data
                X_all = torch.from_numpy(self.embeddings).to(self.device)
                outputs = self.model(X_all)
                num_classes = len(self.label_to_idx)

                if self.is_multilabel:
                    # Multilabel: use sigmoid to get probabilities for each class
                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                    # Use threshold of 0.5 for predictions
                    predicted_np = (probabilities > 0.5).astype(int)
                    labels_np = self.labels  # Already in binary vector format

                    # Calculate accuracy (exact match ratio - all labels must match)
                    exact_match = np.all(predicted_np == labels_np, axis=1)
                    accuracy = exact_match.mean()

                    # Calculate F1 score (samples average for multilabel)
                    # This calculates F1 for each sample and averages
                    f1 = f1_score(labels_np, predicted_np, average='samples', zero_division=0)

                    # Calculate mAP (mean Average Precision) for multilabel
                    try:
                        aps = []
                        for class_idx in range(num_classes):
                            if labels_np[:, class_idx].sum() > 0:  # Only if class has samples
                                ap = average_precision_score(
                                    labels_np[:, class_idx],
                                    probabilities[:, class_idx]
                                )
                                aps.append(ap)
                        mAP = np.mean(aps) if len(aps) > 0 else 0.0
                    except:
                        mAP = 0.0
                        logger.warning("mAP calculation failed, using 0.0")

                    # For calibration in multilabel, use the maximum probability
                    max_probs = np.max(probabilities, axis=1)
                    # Create pseudo single-label for calibration visualization
                    pseudo_predicted = np.argmax(predicted_np, axis=1)
                    pseudo_labels = np.argmax(labels_np, axis=1)
                    calibration_data = self._calculate_calibration_metrics(
                        probabilities, pseudo_predicted, pseudo_labels
                    )

                else:
                    # Single-label: use softmax to get probabilities
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                    _, predicted = torch.max(outputs, 1)
                    predicted_np = predicted.cpu().numpy()
                    labels_np = self.labels

                    # Calculate accuracy
                    correct = (predicted_np == labels_np).sum().item()
                    accuracy = correct / len(labels_np) if len(labels_np) > 0 else 0.0

                    # Calculate F1 score (macro average)
                    if num_classes > 2:
                        f1 = f1_score(labels_np, predicted_np, average='macro', zero_division=0)
                    else:
                        f1 = f1_score(labels_np, predicted_np, average='binary', zero_division=0)

                    # Calculate mAP (mean Average Precision)
                    # For multiclass, we use one-vs-rest approach
                    try:
                        # Create one-hot encoding for true labels
                        labels_onehot = np.zeros((len(labels_np), num_classes))
                        labels_onehot[np.arange(len(labels_np)), labels_np] = 1

                        # Calculate average precision for each class
                        aps = []
                        for class_idx in range(num_classes):
                            if labels_onehot[:, class_idx].sum() > 0:  # Only if class has samples
                                ap = average_precision_score(
                                    labels_onehot[:, class_idx],
                                    probabilities[:, class_idx]
                                )
                                aps.append(ap)

                        mAP = np.mean(aps) if len(aps) > 0 else 0.0
                    except:
                        # Fallback if mAP calculation fails
                        mAP = 0.0
                        logger.warning("mAP calculation failed, using 0.0")

                    # Calculate calibration metrics
                    calibration_data = self._calculate_calibration_metrics(
                        probabilities, predicted_np, labels_np
                    )

            # Store metrics for this repeat
            avg_loss = total_loss / max(1, len(X_train_orig) // batch_size)
            all_losses.append(avg_loss)
            all_accuracies.append(accuracy)
            all_f1_scores.append(f1)
            all_mAPs.append(mAP)

            logger.info(f"Repeat {repeat_idx + 1}/{self.repeats} - Loss: {avg_loss:.4f}, "
                       f"Acc: {accuracy:.4f}, F1: {f1:.4f}, mAP: {mAP:.4f}")

        # After all repeats, restore the model from the last training run
        # (the model is already in the state from the last repeat)

        # Calculate mean and standard deviation across repeats
        metrics = {
            "loss": float(np.mean(all_losses)),
            "accuracy": float(np.mean(all_accuracies)),
            "f1_score": float(np.mean(all_f1_scores)),
            "mAP": float(np.mean(all_mAPs)),
            "loss_sd": float(np.std(all_losses)) if self.repeats > 1 else 0.0,
            "accuracy_sd": float(np.std(all_accuracies)) if self.repeats > 1 else 0.0,
            "f1_score_sd": float(np.std(all_f1_scores)) if self.repeats > 1 else 0.0,
            "mAP_sd": float(np.std(all_mAPs)) if self.repeats > 1 else 0.0,
            "n_labeled": len(self.labeled_indices),
            "n_unlabeled": len(self.unlabeled_indices),
            "repeats": self.repeats,
            "calibration": calibration_data  # Add calibration data from last repeat
        }

        self.training_history.append(metrics)
        logger.info(f"Training step complete: Loss={metrics['loss']:.4f}±{metrics['loss_sd']:.4f}, "
                   f"Acc={metrics['accuracy']:.4f}±{metrics['accuracy_sd']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}±{metrics['f1_score_sd']:.4f}, "
                   f"mAP={metrics['mAP']:.4f}±{metrics['mAP_sd']:.4f}")

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

    def _project_euclidean(self, embeddings_3d: np.ndarray) -> np.ndarray:
        """
        Euclidean space projection (identity - just centered)

        Args:
            embeddings_3d: Input 3D coordinates

        Returns:
            Centered 3D coordinates
        """
        return embeddings_3d

    def _project_spherical(self, embeddings_3d: np.ndarray) -> np.ndarray:
        """
        Project points onto unit sphere (S²)

        Args:
            embeddings_3d: Input 3D coordinates

        Returns:
            3D coordinates normalized to unit sphere
        """
        norms = np.linalg.norm(embeddings_3d, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings_3d / norms

    def _project_torus(self, embeddings_3d: np.ndarray, R: float = 3.0, r: float = 1.0) -> np.ndarray:
        """
        Project points onto a torus surface

        Uses the first two coordinates to determine toroidal angles (theta, phi),
        and the third coordinate to modulate the minor radius.

        Args:
            embeddings_3d: Input 3D coordinates
            R: Major radius (distance from center of torus to center of tube)
            r: Minor radius (radius of the tube)

        Returns:
            3D coordinates on torus surface
        """
        # Normalize input to [-π, π] range for angles
        x_norm = embeddings_3d[:, 0]
        y_norm = embeddings_3d[:, 1]
        z_norm = embeddings_3d[:, 2]

        # Map to toroidal coordinates
        # theta: angle around the major circle
        theta = np.arctan2(y_norm, x_norm)

        # phi: angle around the tube
        # Use the radial distance and z to determine phi
        radial_dist = np.sqrt(x_norm**2 + y_norm**2)
        phi = np.arctan2(z_norm, radial_dist - R)

        # Convert toroidal coordinates to Cartesian
        x_torus = (R + r * np.cos(phi)) * np.cos(theta)
        y_torus = (R + r * np.cos(phi)) * np.sin(theta)
        z_torus = r * np.sin(phi)

        return np.column_stack([x_torus, y_torus, z_torus])

    def _project_hyperbolic(self, embeddings_3d: np.ndarray, scale: float = 0.9) -> np.ndarray:
        """
        Project points into Poincaré ball model of hyperbolic space

        The Poincaré ball is the unit ball with hyperbolic metric.
        Points are mapped so they lie within the ball, with distance from
        origin representing hyperbolic distance.

        Args:
            embeddings_3d: Input 3D coordinates
            scale: Scaling factor to control how close points get to boundary (< 1)

        Returns:
            3D coordinates in Poincaré ball (within unit sphere)
        """
        # Normalize to get direction
        norms = np.linalg.norm(embeddings_3d, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        directions = embeddings_3d / norms

        # Map norms to (0, 1) using tanh for smooth compression
        # tanh naturally maps R -> (-1, 1), and we want (0, 1)
        radii = np.tanh(norms / norms.max() * 2) * scale

        return directions * radii

    def get_embeddings_3d(self,
                         reduction_method: str = "pca",
                         max_embeddings: int = 1000,
                         projection: str = "euclidean") -> np.ndarray:
        """
        Get 3D embeddings from the intermediate layer with geometric projection

        Args:
            reduction_method: Method for dimension reduction ('pca')
            max_embeddings: Maximum number of embeddings to compute
            projection: Geometric space projection ('euclidean', 'spherical', 'torus', 'hyperbolic')

        Returns:
            Array of shape (n_samples, 3) with 3D coordinates in the specified space
        """
        # Validate projection type
        valid_projections = ['euclidean', 'spherical', 'torus', 'hyperbolic']
        if projection not in valid_projections:
            raise ValueError(f"projection must be one of {valid_projections}, got '{projection}'")

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
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            start = time.time()
            embeddings_3d = self.reducer.fit_transform(embeddings_scaled)
            end = time.time()
            logger.info(f"Transformed {len(embeddings)} samples using {self.dim_reduction_method} in {end - start:.3f}s")

        # Center the embeddings at the origin for better camera rotation
        embeddings_3d = embeddings_3d - embeddings_3d.mean(axis=0)

        # Apply geometric projection
        projection_funcs = {
            'euclidean': self._project_euclidean,
            'spherical': self._project_spherical,
            'torus': self._project_torus,
            'hyperbolic': self._project_hyperbolic
        }

        embeddings_3d = projection_funcs[projection](embeddings_3d)
        logger.info(f"Applied {projection} projection to embeddings")

        return embeddings_3d

    def get_state(self) -> Dict:
        """
        Get current state of the active learner

        Returns:
            Dictionary with current state including per-sample uncertainties
        """
        return {
            "n_labeled": int(len(self.labeled_indices)),
            "n_unlabeled": int(len(self.unlabeled_indices)),
            "labeled_indices": [int(idx) for idx in self.labeled_indices],
            "unlabeled_indices": [int(idx) for idx in self.unlabeled_indices],
            "training_history": self.training_history,
            "num_classes": int(len(self.label_to_idx)),
            "labels": list(self.label_to_idx.keys()),  # Already strings from initialization
            "uncertainties": self.uncertainties.tolist(),  # Per-sample uncertainty scores [0, 1]
            "is_multilabel": self.is_multilabel  # Whether dataset is multilabel or single-label
        }
