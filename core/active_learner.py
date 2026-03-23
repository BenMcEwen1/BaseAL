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

    def __init__(self, config_path: Path, base_dir: Optional[Path] = None, verbose: bool = True):
        """
        Initialize Manager with experiments from config file

        Args:
            config_path: Path to YAML configuration file
            base_dir: Base directory for resolving relative paths in config (defaults to config file's parent)
            verbose: Whether to enable INFO-level logging (default True). Set False to suppress logs.
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)
            logging.getLogger("core.utils.sampling").setLevel(logging.WARNING)

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

            if 'metadata_path' in exp:
                meta_path = Path(exp['metadata_path'])
                if not meta_path.is_absolute():
                    exp['metadata_path'] = self.base_dir / meta_path
                else:
                    exp['metadata_path'] = meta_path

        logger.info(f"Loaded {len(experiments)} experiment configurations from {path}")
        return experiments

    def __initialize_experiments(self):
        """Initialize ActiveLearner instances for each experiment config"""
        for i, config in enumerate(self.configs):
            # Extract experiment name if provided, otherwise use index
            exp_name = config.pop('name', f'experiment_{i}')
            self.experiment_names.append(exp_name)

            # Propagate verbose setting to each ActiveLearner
            config.setdefault('verbose', self.verbose)

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
        if 'metadata_path' in new_config:
            new_config['metadata_path'] = Path(new_config['metadata_path'])

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
                'aulc_mAP':      learner.compute_aulc('mAP'),
                'aulc_accuracy': learner.compute_aulc('accuracy'),
                'aulc_f1_score': learner.compute_aulc('f1_score'),
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
                    'num_iterations': len(learner.training_history),
                    'aulc_mAP':      learner.compute_aulc('mAP'),
                    'aulc_accuracy': learner.compute_aulc('accuracy'),
                    'aulc_f1_score': learner.compute_aulc('f1_score'),
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
                'aulc_mAP':      learner.compute_aulc('mAP'),
                'aulc_accuracy': learner.compute_aulc('accuracy'),
                'aulc_f1_score': learner.compute_aulc('f1_score'),
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
        dataset_name: str = "ESC10",
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.001,
        repeats: int = 1,
        device: str = "cpu",
        sampling_strategy: str = "random",
        n_samples_per_iteration: int = 5,
        pretrain_samples: Optional[int] = None,
        dropout_rate: float = 0.0,
        mc_dropout_passes: int = 1,
        verbose: bool = True,
        metadata_path: Optional[Path] = None,
    ):
        """
        Initialize active learner

        Args:
            embeddings_dir: Path to embeddings directory
            annotations_path: Path to annotations CSV (labels.csv) — used internally
                              for training. Must contain a 'filename' and 'label' column,
                              and optionally a 'validation' column.
            model_name: Name of the model (e.g., 'birdnet')
            dataset_name: Name of the dataset (e.g., 'FewShot')
            hidden_dim: Dimension of intermediate embedding
            learning_rate: Learning rate for optimizer
            repeats: Number of training repeats for computing mean/std metrics
            device: Device to use ('cpu' or 'cuda')
            sampling_strategy: Sampling method to use
                              (e.g., 'random', 'margin', 'margin_multilabel',
                              'coreset_farthest', 'nn_disagreement', 'custom')
            n_samples_per_iteration: Default number of samples to select per iteration
            pretrain_samples: Number of high-density samples to pre-select for warm-up training (optional)
            verbose: Whether to enable INFO-level logging (default True). Set False to suppress logs.
            metadata_path: Optional path to a label-free metadata CSV (metadata.csv).
                           When provided, this DataFrame (aligned to the same rows as
                           annotations_path) is passed to sampling strategies instead of
                           the labels CSV, preventing participants from accessing
                           ground-truth labels for unlabeled samples.
        """
        if not verbose:
            logger.setLevel(logging.WARNING)
            logging.getLogger("core.utils.sampling").setLevel(logging.WARNING)

        self.embeddings_dir = embeddings_dir
        self.annotations_path = annotations_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.device = device
        self.repeats = repeats # TODO: potentially adjust for MC
        self.pretrain_samples = pretrain_samples or 0

        # Model dropout for MC
        self.dropout_rate = dropout_rate
        self.mc_dropout_passes = mc_dropout_passes

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
        self.embeddings, self.labels, self.label_to_idx, self.idx_to_label, self.annotations_df, _val_mask = self._load_data()

        # Load label-free metadata for the sampling interface (optional).
        # Aligned to self.annotations_df on 'filename' so row i in metadata_df
        # corresponds to the same sample as row i in annotations_df / embeddings.
        if metadata_path is not None:
            meta_df = pd.read_csv(metadata_path)
            self.metadata_df = (
                self.annotations_df[['filename']]
                .merge(meta_df, on='filename', how='left')
                .reset_index(drop=True)
            )
            logger.info(f"Loaded metadata from {metadata_path} ({len(self.metadata_df)} rows, "
                        f"columns: {list(self.metadata_df.columns)})")
        else:
            self.metadata_df = None

        # Build validation index set (these are never sampled or trained on)
        self.validation_indices: set = set(np.where(_val_mask.values)[0].tolist())
        if self.validation_indices:
            logger.info(f"Holding out {len(self.validation_indices)} validation samples from the AL pool")

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
            num_classes=num_classes,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function - select based on single-label vs multilabel
        if self.is_multilabel:
            # Multilabel: Binary Cross-Entropy with Logits
            self.criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss for multilabel classification")
        else:
            # Single-label: Cross-Entropy Loss
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using CrossEntropyLoss for single-label classification")

        self.sampling_strategy = SamplingStrategy(method=sampling_strategy, n_samples=n_samples_per_iteration)
        logger.info(f"Initialized '{sampling_strategy}' sampling strategy with n_samples={n_samples_per_iteration}")

        # Active learning state (validation indices are excluded from both pools)
        self.labeled_indices = set()
        self.unlabeled_indices = set(range(len(self.embeddings))) - self.validation_indices
        self.training_history = []

        # Pending per-cycle supplementary metrics (populated by sample() / add_samples())
        self._pending_sampling_time: float = 0.0
        self._pending_annotation_cost: int = 0

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

        # Batch size for full-dataset inference (predictions / evaluation)
        self.inference_batch_size = 4096

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
        logger.info(f"Loading annotations from: {self.annotations_path}")
        df = pd.read_csv(self.annotations_path)
        logger.info(f"Loaded {len(df)} annotations")

        # --- Detect multilabel and build label mappings (vectorized) -----------
        label_column = 'label'
        label_strings = df[label_column].astype(str)
        is_multilabel = label_strings.str.contains(';').any()

        if is_multilabel:
            logger.info("Detected MULTILABEL format (semicolon-separated labels)")
        else:
            logger.info("Detected SINGLE-LABEL format")

        # Extract all unique labels using vectorized split
        all_labels: set = set()
        for lbl in label_strings.str.split(';').explode().str.strip().unique():
            all_labels.add(lbl)

        unique_labels = sorted(all_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        num_classes = len(unique_labels)

        logger.info(f"Found {num_classes} unique classes: {unique_labels}")

        # --- Build embedding filenames and filter to existing files -----------
        # Construct expected embedding path for every annotation row
        stems = df['filename'].apply(lambda f: Path(f).stem)
        embedding_names = stems + f".npy" # NOTE: _{self.model_name}

        # Single directory listing instead of N individual Path.exists() calls
        existing_files = set(p.name for p in self.embeddings_dir.iterdir())
        exists_mask = embedding_names.isin(existing_files)

        # If no exact matches, try suffix matching (embedding files may have a deployment prefix
        # separated by '__', e.g. "deployment__<stem>_<model>.npy")
        if not exists_mask.any():
            suffix_map = {f.split('__', 1)[-1]: f for f in existing_files}
            exists_mask = embedding_names.isin(suffix_map)
            embedding_names = embedding_names.map(lambda n: suffix_map.get(n, n))

        n_missing = (~exists_mask).sum()
        if n_missing > 0:
            logger.warning(f"{n_missing} annotation rows have no matching embedding file — skipped")

        # Filter to matched rows only
        df_matched = df[exists_mask].reset_index(drop=True)
        embedding_names_matched = embedding_names[exists_mask].values
        label_strings_matched = label_strings[exists_mask].values

        n_matched = len(df_matched)
        logger.info(f"Matched {n_matched} annotations to embedding files")

        # --- Load embeddings (with cache) -------------------------------------
        cache_dir = self.embeddings_dir / ".cache"
        manifest_path = cache_dir / "manifest.json"
        cache_path = cache_dir / "embeddings.npy"

        # Check if a valid cache exists: manifest filenames must match current matched set
        filenames_list = list(embedding_names_matched)
        cache_hit = False
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if manifest.get("filenames") == filenames_list:
                embeddings = np.load(str(cache_path), mmap_mode='r')
                cache_hit = True
                logger.info(f"Loaded cached embeddings: {embeddings.shape}")

        if not cache_hit:
            # Load individual files (slow for large datasets)
            embedding_paths = [self.embeddings_dir / name for name in embedding_names_matched]

            sample_emb = np.load(embedding_paths[0])
            if sample_emb.ndim == 1:
                emb_dim = sample_emb.shape[0]
            else:
                emb_dim = sample_emb.shape[1] if len(sample_emb) == 1 else sample_emb.size

            embeddings = np.empty((n_matched, emb_dim), dtype=np.float32)

            def _load_one(idx: int) -> None:
                emb = np.load(embedding_paths[idx])
                if emb.ndim == 1:
                    embeddings[idx] = emb
                else:
                    embeddings[idx] = emb[0] if len(emb) == 1 else emb.flatten()

            if sample_emb.ndim == 1:
                embeddings[0] = sample_emb
            else:
                embeddings[0] = sample_emb[0] if len(sample_emb) == 1 else sample_emb.flatten()

            if n_matched > 1:
                max_workers = min(8, n_matched - 1)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    list(pool.map(_load_one, range(1, n_matched)))

            # Save consolidated cache for fast future loads
            cache_dir.mkdir(exist_ok=True)
            np.save(str(cache_path), embeddings)
            with open(manifest_path, 'w') as f:
                json.dump({"filenames": filenames_list}, f)
            logger.info(f"Saved embedding cache to {cache_dir}")

        # --- Build label arrays (vectorized) ----------------------------------
        if is_multilabel:
            # Multilabel: build binary matrix
            labels = np.zeros((n_matched, num_classes), dtype=np.float32)
            for i, lbl_str in enumerate(label_strings_matched):
                for lbl in lbl_str.split(';'):
                    lbl = lbl.strip()
                    if lbl in label_to_idx:
                        labels[i, label_to_idx[lbl]] = 1.0
            logger.info(f"Loaded {n_matched} embeddings with shape {embeddings.shape}")
            logger.info(f"Labels shape: {labels.shape} (multilabel binary vectors)")
        else:
            # Single-label: vectorized map to integer indices
            labels = np.array([label_to_idx[s] for s in label_strings_matched], dtype=np.int64)
            logger.info(f"Loaded {n_matched} embeddings with shape {embeddings.shape}")
            logger.info(f"Labels shape: {labels.shape} (single-label indices)")

        annotations_df = df_matched

        # --- Build validation mask --------------------------------------------
        if 'validation' in annotations_df.columns:
            validation_mask = annotations_df['validation'].astype(str).str.lower().isin(['true', '1', 'yes'])
            n_val = validation_mask.sum()
            logger.info(f"Validation column detected: {n_val} validation samples, "
                        f"{len(annotations_df) - n_val} training/unlabeled samples")
        else:
            validation_mask = pd.Series(False, index=annotations_df.index)
            logger.info("No validation column found — all samples treated as unlabeled")

        logger.info(f"Annotations DataFrame shape: {annotations_df.shape}")

        return embeddings, labels, label_to_idx, idx_to_label, annotations_df, validation_mask

    def _pretrain_warmup(self, n_samples: int):
        """
        Pre-training warm-up: select high-density samples for initial training

        This method selects samples with lots of neighbors (high density) for
        initial annotation and training, without requiring model predictions.

        Args:
            n_samples: Number of high-density samples to pre-select
        """
        from .utils.sampling import densityEstimation # TODO: generalise to other warmup methods

        # Candidate pool: all non-validation samples
        candidate_indices = np.array(sorted(set(range(len(self.embeddings))) - self.validation_indices))
        n_samples = min(n_samples, len(candidate_indices))

        logger.info(f"Computing density estimation for {len(candidate_indices)} candidate samples...")

        # Compute density using KNN method (samples with more neighbors = higher density)
        # Using k=20 as a reasonable default for neighbor count
        density_scores = densityEstimation(
            embeddings=self.embeddings[candidate_indices],
            method='knn',
            k=min(20, len(candidate_indices) - 1),  # Ensure k < n_samples
            beta=1
        )

        logger.info(f"Density scores - min: {density_scores.min():.4f}, max: {density_scores.max():.4f}, mean: {density_scores.mean():.4f}")

        # Select samples with highest density and map back to global indices
        top_local = np.argsort(density_scores)[-n_samples:]
        top_density_indices = candidate_indices[top_local]

        # Add these samples to labeled set
        self.labeled_indices = set(top_density_indices.tolist())

        # Remove from unlabeled set (validation indices already excluded)
        self.unlabeled_indices = set(candidate_indices.tolist()) - self.labeled_indices

        logger.info(f"Pre-training warm-up complete: selected {len(self.labeled_indices)} high-density samples")
        logger.info(f"Remaining unlabeled samples: {len(self.unlabeled_indices)}")

    def _predict_all(self) -> np.ndarray:
        """
        Run batched inference on the full embedding matrix.

        Returns:
            probabilities: Array of shape (n_samples, num_classes) with class
                           probabilities (softmax for single-label, sigmoid for multilabel).
        """
        n = len(self.embeddings)
        bs = self.inference_batch_size

        if self.mc_dropout_passes <= 1:
            self.model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, n, bs):
                    batch = torch.from_numpy(self.embeddings[start:start + bs]).to(self.device)
                    out = self.model(batch)
                    probs = torch.sigmoid(out) if self.is_multilabel else torch.softmax(out, dim=1)
                    chunks.append(probs.cpu().numpy())
            return np.concatenate(chunks, axis=0)
        
        else:
            # mc_dropout_passes > 1 then compute multiple forward passes
            all_passes = []
            self.model.train()
            with torch.no_grad():
                for _ in range(self.mc_dropout_passes):
                    chunks = []
                    for start in range(0, n, bs):
                        batch = torch.from_numpy(self.embeddings[start:start + bs]).to(self.device)
                        out = self.model(batch)
                        probs = torch.sigmoid(out) if self.is_multilabel else torch.softmax(out, dim=1)
                        chunks.append(probs.cpu().numpy())
                    all_passes.append(np.concatenate(chunks, axis=0))
            self.model.eval()
            return np.stack(all_passes, axis=0) # shape: (mc_dropout_passes, n_samples, n_classes)

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

        # Get predictions for all samples (batched to avoid OOM on large datasets)
        predictions = self._predict_all()

        if self.mc_dropout_passes > 1:
            mean_predictions = predictions.mean(axis=0) # (n_samples, n_classes)
        else:
            mean_predictions = predictions

        # Convert sets to sorted lists for numpy indexing in sampling strategies
        unlabeled_list = sorted(self.unlabeled_indices)
        labeled_list = sorted(self.labeled_indices)

        # Pass only the labels for the labeled pool (aligned with labeled_list by position).
        # This prevents sampling strategies from accessing ground-truth labels for
        # unlabeled or validation samples. Participants can recover the embedding for
        # labels[i] via embeddings[labeled_indices[i]] (same sorted order).
        labeled_labels = self.labels[labeled_list] if len(labeled_list) > 0 else None

        # Call the sampling strategy with all available data
        # Now returns both selected indices and uncertainties
        _t0 = time.perf_counter()
        selected, unlabeled_uncertainties = self.sampling_strategy.select(
            unlabeled_indices=unlabeled_list,
            predictions=mean_predictions,
            embeddings=self.embeddings,
            model=self.model,
            metadata=self.metadata_df,
            labeled_indices=labeled_list,
            labels=labeled_labels,
            mc_predictions=predictions if self.mc_dropout_passes > 1 else None
        )
        self._pending_sampling_time += time.perf_counter() - _t0

        # Debug: Check uncertainty values from sampling strategy
        logger.info(f"Unlabeled uncertainties - min: {unlabeled_uncertainties.min():.4f}, max: {unlabeled_uncertainties.max():.4f}, mean: {unlabeled_uncertainties.mean():.4f}")
        logger.info(f"Unlabeled uncertainties shape: {unlabeled_uncertainties.shape}, expected: {len(self.unlabeled_indices)}")

        # Update uncertainties array for all samples
        # Labeled samples have uncertainty = 0
        self.uncertainties = np.zeros(len(self.embeddings))
        # Unlabeled samples have computed uncertainty
        self.uncertainties[unlabeled_list] = unlabeled_uncertainties

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
        moved = self.unlabeled_indices.intersection(indices)
        self.unlabeled_indices -= moved
        self.labeled_indices |= moved

        # Annotation cost = Σ events per selected sample.
        # For multilabel: events(i) = number of positive labels.
        # For single-label: every sample has exactly 1 event.
        if self.is_multilabel:
            self._pending_annotation_cost += int(self.labels[list(moved)].sum())
        else:
            self._pending_annotation_cost += len(moved)

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

            # Prepare labeled data (convert set to sorted list for numpy indexing)
            labeled_list = sorted(self.labeled_indices)
            X_train_orig = torch.from_numpy(self.embeddings[labeled_list]).to(self.device)
            y_train_orig = torch.from_numpy(self.labels[labeled_list]).to(self.device)

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

            # Evaluation — use validation set if available, else all samples
            probabilities = self._predict_all()
            num_classes = len(self.label_to_idx)

            if self.mc_dropout_passes > 1:
                probabilities = probabilities.mean(axis=0) # (n_samples, n_classes)

            if self.validation_indices:
                eval_indices = sorted(self.validation_indices)
                logger.info(f"Evaluating on {len(eval_indices)} validation samples")
            else:
                eval_indices = list(range(len(self.embeddings)))

            eval_probs = probabilities[eval_indices]
            eval_labels = self.labels[eval_indices]

            if self.is_multilabel:
                # Use threshold of 0.5 for predictions
                predicted_np = (eval_probs > 0.5).astype(int)
                labels_np = eval_labels  # Already in binary vector format

                # Calculate accuracy (exact match ratio - all labels must match)
                exact_match = np.all(predicted_np == labels_np, axis=1)
                accuracy = exact_match.mean()

                # Calculate F1 score (samples average for multilabel)
                f1 = f1_score(labels_np, predicted_np, average='macro', zero_division=0)

                # Calculate mAP (mean Average Precision) for multilabel
                try:
                    aps = []
                    for class_idx in range(num_classes):
                        if labels_np[:, class_idx].sum() > 0:  # Only if class has samples
                            ap = average_precision_score(
                                labels_np[:, class_idx],
                                eval_probs[:, class_idx]
                            )
                            aps.append(ap)
                    mAP = np.mean(aps) if len(aps) > 0 else 0.0
                except:
                    mAP = 0.0
                    logger.warning("mAP calculation failed, using 0.0")

                # For calibration in multilabel, use the maximum probability
                pseudo_predicted = np.argmax(predicted_np, axis=1)
                pseudo_labels = np.argmax(labels_np, axis=1)
                calibration_data = self._calculate_calibration_metrics(
                    eval_probs, pseudo_predicted, pseudo_labels
                )

            else:
                # Single-label: probabilities already from _predict_all()
                predicted_np = np.argmax(eval_probs, axis=1)
                labels_np = eval_labels

                # Calculate accuracy
                correct = (predicted_np == labels_np).sum()
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
                                eval_probs[:, class_idx]
                            )
                            aps.append(ap)

                    mAP = np.mean(aps) if len(aps) > 0 else 0.0
                except:
                    mAP = 0.0
                    logger.warning("mAP calculation failed, using 0.0")

                # Calculate calibration metrics
                calibration_data = self._calculate_calibration_metrics(
                    eval_probs, predicted_np, labels_np
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
            "epochs": epochs,
            "batch_size": batch_size,
            "sampling_time_s": round(self._pending_sampling_time, 6),
            "annotation_cost": self._pending_annotation_cost,
            "calibration": calibration_data  # Add calibration data from last repeat
        }

        # Reset pending supplementary metrics for next cycle
        self._pending_sampling_time = 0.0
        self._pending_annotation_cost = 0

        self.training_history.append(metrics)

        # Compute running AULC for all three performance metrics
        aulc_mAP      = self.compute_aulc('mAP')
        aulc_accuracy = self.compute_aulc('accuracy')
        aulc_f1       = self.compute_aulc('f1_score')
        metrics['aulc_mAP']      = aulc_mAP
        metrics['aulc_accuracy'] = aulc_accuracy
        metrics['aulc_f1_score'] = aulc_f1
        self.training_history[-1].update({
            'aulc_mAP':      aulc_mAP,
            'aulc_accuracy': aulc_accuracy,
            'aulc_f1_score': aulc_f1,
        })

        logger.info(f"Training step complete: Loss={metrics['loss']:.4f}±{metrics['loss_sd']:.4f}, "
                   f"Acc={metrics['accuracy']:.4f}±{metrics['accuracy_sd']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}±{metrics['f1_score_sd']:.4f}, "
                   f"mAP={metrics['mAP']:.4f}±{metrics['mAP_sd']:.4f}, "
                   f"AULC(mAP)={aulc_mAP:.4f}")

        return metrics

    def compute_aulc(self, metric: str = 'mAP') -> float:
        """
        Compute the Area Under the Learning Curve (AULC) from the current training
        history using the trapezoidal rule, normalised by the x-axis range so the
        result is in [0, 1].

        Args:
            metric: Which metric to use as the y-axis. One of 'mAP', 'accuracy',
                    or 'f1_score'. Defaults to 'mAP'.

        Returns:
            Normalised AULC in [0, 1], or 0.0 if no training steps have been
            recorded or all steps have the same n_labeled value.
        """
        if len(self.training_history) < 1:
            return 0.0
        # Prepend a (0, 0) anchor so cycle 1 yields a non-zero area.
        n_labeled = [0] + [entry['n_labeled'] for entry in self.training_history]
        values    = [0.0] + [entry[metric]    for entry in self.training_history]
        x_range   = n_labeled[-1] - n_labeled[0]
        if x_range == 0:
            return 0.0
        return float(np.trapz(values, n_labeled) / x_range)

    def export(self, output_path: str,
               author_lastname: Optional[str] = None,
               institute_abbreviation: Optional[str] = None,
               max_budget: Optional[int] = None) -> None:
        """
        Export the submission YAML file for the BioDCASE challenge.

        The file contains:
        - Configuration (hyperparameters, model size)
        - Learning curve: per-cycle performance metrics and AULC
        - Supplementary metrics: sampling wall-time, annotation cost,
          computational cost (training)

        Internal bookkeeping fields (calibration, SD values, n_unlabeled,
        repeats) are excluded from the output.

        Args:
            output_path: Destination path for the YAML file.
                         Recommended naming: {method}_{dataset}_{lastname}.yaml
            author_lastname: Participant's last name (included in the YAML header).
            institute_abbreviation: Short institute identifier (included in the YAML header).
            max_budget: Total labelling budget used in the run (including warm-up).
                        Used to derive the baseline n_cycles as
                        max_budget // BASELINE_BATCH_SIZE (32), giving a fair
                        cost comparison. Falls back to actual n_cycles if None.
        """
        _STRIP = {'calibration', 'loss', 'loss_sd', 'n_unlabeled', 'repeats'}

        model_parameters = int(sum(p.numel() for p in self.model.parameters()))

        # Build per-cycle learning curve (stripped)
        learning_curve = []
        for i, entry in enumerate(self.training_history):
            row = {'cycle': i + 1}
            for k, v in entry.items():
                if k not in _STRIP:
                    row[k] = round(v, 6) if isinstance(v, float) else v
            learning_curve.append(row)

        # Supplementary aggregates
        total_sampling_time  = round(sum(e.get('sampling_time_s', 0.0) for e in self.training_history), 6)
        total_annotation_cost = int(sum(e.get('annotation_cost', 0) for e in self.training_history))
        n_cycles = len(self.training_history)
        # Use epochs from the last cycle (consistent across cycles for the baseline)
        last = self.training_history[-1] if self.training_history else {}
        epochs_per_cycle = last.get('epochs', None)

        # Baseline config (fixed): batch_size=32, epochs=10, same model_parameters.
        # n_cycles for baseline = max_budget // 32 (if budget provided) else actual n_cycles.
        _BASELINE_EPOCHS     = 10
        _BASELINE_BATCH_SIZE = 32
        baseline_n_cycles = (max_budget // _BASELINE_BATCH_SIZE
                             if max_budget is not None else n_cycles)
        cost_method   = (model_parameters * epochs_per_cycle * n_cycles
                         if epochs_per_cycle is not None else None)
        baseline_cost = model_parameters * _BASELINE_EPOCHS * baseline_n_cycles
        relative_cost = (round(cost_method / baseline_cost, 4)
                         if (cost_method is not None and baseline_cost > 0) else None)

        submission = {
            'submission_timestamp': datetime.now().isoformat(timespec='seconds'),
            'author_lastname': author_lastname,
            'institute_abbreviation': institute_abbreviation,
            'sampling_strategy': self.sampling_strategy.method
                if hasattr(self.sampling_strategy, 'method') else str(self.sampling_strategy),
            'dataset': self.dataset_name,
            'model': self.model_name,
            'config': {
                'learning_rate': self.learning_rate,
                'model_parameters': model_parameters,
                'repeats': self.repeats,
                'pretrain_samples': self.pretrain_samples,
            },
            'learning_curve': learning_curve,
            'supplementary': {
                'n_cycles': n_cycles,
                'total_sampling_time_s': total_sampling_time,
                'total_annotation_cost': total_annotation_cost,
                'computational_cost': {
                    'model_parameters': model_parameters,
                    'epochs_per_cycle': epochs_per_cycle,
                    'n_cycles': n_cycles,
                    'cost_method': cost_method,
                    'baseline_n_cycles': baseline_n_cycles,
                    'baseline_cost': baseline_cost,
                    'relative_cost': relative_cost,
                },
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(submission, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        logger.info(f"Submission exported to {output_path}")

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
            "labeled_indices": sorted(self.labeled_indices),
            "unlabeled_indices": sorted(self.unlabeled_indices),
            "training_history": self.training_history,
            "num_classes": int(len(self.label_to_idx)),
            "labels": list(self.label_to_idx.keys()),  # Already strings from initialization
            "uncertainties": self.uncertainties.tolist(),  # Per-sample uncertainty scores [0, 1]
            "is_multilabel": self.is_multilabel  # Whether dataset is multilabel or single-label
        }
