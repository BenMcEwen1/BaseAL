"""
FastAPI application for serving embeddings data and active learning
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import bacpipe
import umap

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.active_learner import ActiveLearner, Manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BaseAL API")

# Configure CORS to allow requests from the React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base paths
BASE_DIR = Path(__file__).parent.parent
EMBEDDINGS_BASE_PATH = BASE_DIR / "results" / "test_data" / "embeddings"
ANNOTATIONS_BASE_PATH = BASE_DIR / "results" / "test_data" / "evaluations"

# Global active learner instance (for backward compatibility with single experiment)
active_learner: Optional[ActiveLearner] = None

# Global manager instance (for managing multiple experiments)
manager: Optional[Manager] = None
selected_experiment_index: int = 0  # Which experiment to use for single-experiment endpoints


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Reduce embeddings from high dimensions (1024) to 3D using PCA

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Number of dimensions to reduce to (default: 3)

    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    # Standardize the features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_scaled)

    # Log variance explained
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA variance explained: {variance_explained:.2%}")

    return reduced


def load_embeddings_from_folder(folder_path: Path) -> List[Dict[str, Any]]:
    """
    Load all .npy embeddings from a folder

    Args:
        folder_path: Path to folder containing .npy files

    Returns:
        List of dictionaries containing filename and embeddings
    """
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    embeddings_data = []
    npy_files = sorted(folder_path.glob("*.npy"))

    if not npy_files:
        raise ValueError(f"No .npy files found in {folder_path}")

    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            embeddings_data.append({
                "filename": npy_file.name,
                "embeddings": data,
                "shape": data.shape
            })
            logger.info(f"Loaded {npy_file.name}: shape {data.shape}")
        except Exception as e:
            logger.error(f"Error loading {npy_file}: {e}")
            continue

    return embeddings_data


@app.get("/")
def info():
    return {"message": "BaseAL Embeddings API", "version": "1.0.0"}

@app.get("/api/generate")
def generate_embeddings():
    try:
        bacpipe.play(save_logs=False)
    except:
        raise HTTPException(status_code=404, detail=f"Embedding generation failed")
    return {"status": "complete"}



@app.get("/api/models")
def list_models():
    """List available embedding models"""
    models = []

    if EMBEDDINGS_BASE_PATH.exists():
        for model_dir in EMBEDDINGS_BASE_PATH.iterdir():
            if model_dir.is_dir():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir.relative_to(EMBEDDINGS_BASE_PATH))
                })
    else:
        raise Exception(status_code=404, detail=f"Embedding base path {EMBEDDINGS_BASE_PATH} doesn't exist")

    return {"models": models}


@app.get("/api/embeddings/{model_name}/datasets")
def list_datasets(model_name: str):
    """List available datasets for a given model"""
    model_path = EMBEDDINGS_BASE_PATH / model_name / "audio"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    datasets = []
    for dataset_dir in model_path.iterdir():
        if dataset_dir.is_dir():
            npy_files = list(dataset_dir.glob("*.npy"))
            datasets.append({
                "name": dataset_dir.name,
                "file_count": len(npy_files)
            })

    return {"model": model_name, "datasets": datasets}


@app.get("/api/embeddings/{model_name}/{dataset_name}/3d")
def get_embeddings_3d(model_name: str, dataset_name: str):
    """
    Get embeddings reduced to 3D coordinates

    Args:
        model_name: Name of the model (e.g., '2025-11-09_10-27___birdnet-test_data')
        dataset_name: Name of the dataset folder (e.g., 'FewShot')

    Returns:
        JSON with 3D coordinates for each file's embeddings
    """
    folder_path = EMBEDDINGS_BASE_PATH / model_name / "audio" / dataset_name

    if not folder_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {model_name}/{dataset_name}"
        )

    try:
        # Load all embeddings
        embeddings_data = load_embeddings_from_folder(folder_path)

        # Concatenate all embeddings for PCA
        all_embeddings = []
        file_info = []

        for item in embeddings_data:
            embeddings = item["embeddings"]
            n_samples = embeddings.shape[0]

            all_embeddings.append(embeddings)
            file_info.append({
                "filename": item["filename"],
                "n_samples": n_samples,
                "original_shape": item["shape"]
            })

        # Concatenate all embeddings
        all_embeddings_array = np.vstack(all_embeddings)
        logger.info(f"Total embeddings shape: {all_embeddings_array.shape}")

        # Reduce to 3D
        reduced_3d = reduce_dimensions(all_embeddings_array, n_components=3)

        # Split back into original files
        result = []
        start_idx = 0

        for i, info in enumerate(file_info):
            n_samples = info["n_samples"]
            end_idx = start_idx + n_samples

            # Extract coordinates for this file
            coords = reduced_3d[start_idx:end_idx].tolist()

            result.append({
                "filename": info["filename"],
                "file_index": i,
                "n_samples": n_samples,
                "coordinates": coords
            })

            start_idx = end_idx

        return {
            "model": model_name,
            "dataset": dataset_name,
            "total_samples": len(all_embeddings_array),
            "files": result
        }

    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/api/embeddings/{model_name}/{dataset_name}/steps")
# def get_embeddings_steps(model_name: str, dataset_name: str, n_steps: int = 4):
#     """
#     Get embeddings as progressive steps showing clustering over different PCA dimensions
#     This creates an animation effect similar to the example

#     Args:
#         model_name: Name of the model
#         dataset_name: Name of the dataset folder
#         n_steps: Number of progressive steps to generate

#     Returns:
#         JSON with coordinates for each step
#     """
#     folder_path = EMBEDDINGS_BASE_PATH / model_name / "audio" / dataset_name

#     if not folder_path.exists():
#         raise HTTPException(
#             status_code=404,
#             detail=f"Dataset not found: {model_name}/{dataset_name}"
#         )

#     try:
#         # Load all embeddings
#         embeddings_data = load_embeddings_from_folder(folder_path)

#         # Concatenate all embeddings
#         all_embeddings = np.vstack([item["embeddings"] for item in embeddings_data])
#         total_samples = len(all_embeddings)

#         logger.info(f"Total embeddings shape: {all_embeddings.shape}")

#         # Standardize once
#         scaler = StandardScaler()
#         embeddings_scaled = scaler.fit_transform(all_embeddings)

#         # Generate steps with increasing PCA variance (simulating clustering)
#         steps = []

#         # Determine max components we can use (must be <= min(n_samples, n_features))
#         max_components = min(total_samples, all_embeddings.shape[1])

#         for step in range(n_steps):
#             # Vary the amount of dimension reduction to simulate clustering progression
#             if step == 0:
#                 # First step: More random (use more components, then project to 3D)
#                 # Use at most 50 components, but respect the max_components limit
#                 n_components_first = min(50, max_components)

#                 if n_components_first > 3:
#                     pca = PCA(n_components=n_components_first)
#                     intermediate = pca.fit_transform(embeddings_scaled)
#                     # Add random noise to simulate initial disorder
#                     noise_scale = 2.0
#                     intermediate += np.random.randn(*intermediate.shape) * noise_scale
#                     # Project to 3D
#                     pca_3d = PCA(n_components=3)
#                     coords = pca_3d.fit_transform(intermediate)
#                 else:
#                     # If we have very few samples, just use 3 components directly
#                     pca = PCA(n_components=3)
#                     coords = pca.fit_transform(embeddings_scaled)
#                     # Add noise
#                     coords += np.random.randn(*coords.shape) * 2.0
#             else:
#                 # Progressive clustering: reduce dimensions more aggressively
#                 n_components_intermediate = max(3, min(30 - step * 7, max_components))
#                 pca = PCA(n_components=n_components_intermediate)
#                 intermediate = pca.fit_transform(embeddings_scaled)

#                 if n_components_intermediate > 3:
#                     # Further reduce to 3D
#                     pca_3d = PCA(n_components=3)
#                     coords = pca_3d.fit_transform(intermediate)
#                 else:
#                     coords = intermediate

#                 # Scale down coordinates progressively to show clustering
#                 scale_factor = 1.0 - (step * 0.15)
#                 coords = coords * scale_factor

#             # Normalize the coordinates to a reasonable range
#             coords = coords * 3.0 / np.std(coords)

#             steps.append(coords.tolist())

#         return {
#             "model": model_name,
#             "dataset": dataset_name,
#             "total_samples": total_samples,
#             "n_steps": n_steps,
#             "steps": steps
#         }

#     except Exception as e:
#         logger.error(f"Error processing embeddings: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ==================== Manager Endpoints ====================

@app.post("/api/manager/initialize")
def initialize_manager(config_path: str = "core/config.yml"):
    """
    Initialize the Manager with a config file

    Args:
        config_path: Path to the YAML config file (relative to project root)

    Returns:
        Status and initial summary
    """
    global manager, selected_experiment_index

    try:
        config_file = BASE_DIR / config_path

        if not config_file.exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_file}")

        logger.info(f"Initializing Manager with config: {config_file}")
        logger.info(f"Base directory: {BASE_DIR}")

        # Initialize Manager with BASE_DIR so all paths in config are resolved relative to project root
        manager = Manager(config_file, base_dir=BASE_DIR)
        selected_experiment_index = 0  # Reset to first experiment

        return {
            "status": "initialized",
            "summary": manager.get_summary()
        }

    except Exception as e:
        logger.error(f"Error initializing manager: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manager/add-experiment")
def add_experiment_to_manager(
    name: str,
    model_name: str = "2025-11-13_21-42___birdnet-test_data",
    dataset_name: str = "esc50",
    learning_rate: float = 0.0001,
    hidden_dim: Optional[int] = None,
    device: str = "cpu"
):
    """
    Add a new experiment to the manager

    Args:
        name: Name for the new experiment
        model_name: Model directory name
        dataset_name: Dataset name
        learning_rate: Learning rate
        hidden_dim: Hidden dimension (optional)
        device: Device to use

    Returns:
        Updated summary
    """
    global manager

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    try:
        # Extract base model name
        base_model_name = "birdnet" if "birdnet" in model_name.lower() else "perch_bird"

        # Create config
        new_config = {
            'embeddings_dir': str(EMBEDDINGS_BASE_PATH / model_name / "audio" / dataset_name),
            'annotations_path': str(ANNOTATIONS_BASE_PATH / base_model_name / "classification" / "default_classifier_annotations.csv"),
            'model_name': base_model_name,
            'dataset_name': dataset_name,
            'learning_rate': learning_rate,
            'hidden_dim': hidden_dim,
            'device': device
        }

        manager.add(new_config, name=name)

        return {
            "status": "added",
            "summary": manager.get_summary()
        }

    except Exception as e:
        logger.error(f"Error adding experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manager/run")
def run_manager_cycle(
    n_samples: int = 5,
    epochs: int = 5,
    batch_size: int = 8,
    parallel: bool = False
):
    """
    Run one AL cycle across all experiments

    Args:
        n_samples: Number of samples per experiment
        epochs: Training epochs
        batch_size: Batch size
        parallel: Run in parallel

    Returns:
        Results for each experiment
    """
    global manager

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    try:
        results = manager.run(
            n_samples=n_samples,
            epochs=epochs,
            batch_size=batch_size,
            parallel=parallel
        )

        return {
            "results": results,
            "summary": manager.get_summary()
        }

    except Exception as e:
        logger.error(f"Error running manager cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/manager/experiments")
def list_manager_experiments():
    """
    Get list of all experiments in the manager

    Returns:
        List of experiment names and basic info
    """
    global manager

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    return {
        "experiments": [
            {
                "index": i,
                "name": name,
                "learning_rate": learner.learning_rate,
                "model_name": learner.model_name,
                "n_labeled": len(learner.labeled_indices),
                "n_unlabeled": len(learner.unlabeled_indices),
                "training_history": learner.training_history
            }
            for i, (learner, name) in enumerate(zip(manager.experiments, manager.experiment_names))
        ]
    }


@app.get("/api/manager/summary")
def get_manager_summary():
    """
    Get current summary of all experiments

    Returns:
        Summary of all experiments
    """
    global manager

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    return manager.get_summary()


@app.post("/api/manager/save")
def save_manager_results(output_dir: str = "results/manager_experiments"):
    """
    Save all experiment results

    Args:
        output_dir: Directory to save results (relative to project root)

    Returns:
        Status
    """
    global manager

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    try:
        save_path = BASE_DIR / output_dir
        manager.save(output_dir=save_path)

        return {
            "status": "saved",
            "output_dir": str(save_path)
        }

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manager/select-experiment")
def select_experiment(experiment_index: int):
    """
    Select which experiment to use for single-experiment endpoints

    Args:
        experiment_index: Index of experiment to select

    Returns:
        Selected experiment info
    """
    global manager, selected_experiment_index, active_learner

    if manager is None:
        raise HTTPException(status_code=400, detail="Manager not initialized. Call /api/manager/initialize first.")

    if experiment_index < 0 or experiment_index >= len(manager.experiments):
        raise HTTPException(status_code=400, detail=f"Invalid experiment index: {experiment_index}")

    selected_experiment_index = experiment_index
    # Also set the active_learner to point to the selected experiment for backward compatibility
    active_learner = manager.experiments[selected_experiment_index]

    return {
        "status": "selected",
        "experiment_index": selected_experiment_index,
        "experiment_name": manager.experiment_names[selected_experiment_index],
        "state": active_learner.get_state()
    }


# ==================== Active Learning Endpoints ====================

@app.post("/api/active-learning/initialize")
def initialize_active_learner(model_name: str = "2025-11-13_21-42___birdnet-test_data", dataset_name: str = "esc50"):
    """
    Initialize the active learning pipeline

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset

    Returns:
        Status and initial state
    """
    global active_learner

    try:
        # Extract base model name (birdnet or perch_bird)
        base_model_name = "birdnet" if "birdnet" in model_name.lower() else "perch_bird"

        # Paths
        embeddings_dir = EMBEDDINGS_BASE_PATH / model_name / "audio" / dataset_name
        annotations_path = ANNOTATIONS_BASE_PATH / base_model_name / "classification" / "default_classifier_annotations.csv"

        logger.info(f"Initializing active learner:")
        logger.info(f"  Embeddings dir: {embeddings_dir}")
        logger.info(f"  Annotations path: {annotations_path}")
        logger.info(f"  Embeddings exists: {embeddings_dir.exists()}")
        logger.info(f"  Annotations exists: {annotations_path.exists()}")

        if not embeddings_dir.exists():
            raise HTTPException(status_code=404, detail=f"Embeddings directory not found: {embeddings_dir}")

        if not annotations_path.exists():
            raise HTTPException(status_code=404, detail=f"Annotations file not found: {annotations_path}")

        # Initialize active learner
        active_learner = ActiveLearner(
            embeddings_dir=embeddings_dir,
            annotations_path=annotations_path,
            model_name=base_model_name,
            dataset_name=dataset_name,
            hidden_dim=1024,
            learning_rate=0.001,
            device="cpu"
        )

        return {
            "status": "initialized",
            "state": active_learner.get_state()
        }

    except Exception as e:
        logger.error(f"Error initializing active learner: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/active-learning/sample")
def sample_next_batch(n_samples: int = 200):
    """
    Sample next batch using active learning strategy

    Args:
        n_samples: Number of samples to select

    Returns:
        Selected indices and updated state
    """
    global active_learner

    if active_learner is None:
        raise HTTPException(status_code=400, detail="Active learner not initialized. Call /initialize first.")

    try:
        # Sample using random strategy
        selected_indices = active_learner.sample(n_samples=n_samples)

        # Add to labeled set
        active_learner.add_samples(selected_indices)

        return {
            "selected_indices": selected_indices,
            "state": active_learner.get_state()
        }

    except Exception as e:
        logger.error(f"Error sampling: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/active-learning/train")
def train_model(epochs: int = 5, batch_size: int = 8):
    """
    Train the model on the current labeled set

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Training metrics and updated state
    """
    global active_learner

    if active_learner is None:
        raise HTTPException(status_code=400, detail="Active learner not initialized. Call /initialize first.")

    try:
        metrics = active_learner.train_step(epochs=epochs, batch_size=batch_size)

        # print(active_learner.get_state())

        return {
            "metrics": metrics,
            "state": active_learner.get_state()
        }

    except Exception as e:
        logger.error(f"Error training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/active-learning/embeddings-3d")
def get_active_learning_embeddings(dimension_reduction: str = 'UMAP', projection: str = 'euclidean'):
    """
    Get 3D embeddings from the trained model

    Returns:
        3D coordinates, labels, and per-sample uncertainties
    """
    global active_learner

    if active_learner is None:
        raise HTTPException(status_code=400, detail="Active learner not initialized. Call /initialize first.")

    try:
        # Get 3D embeddings from model
        embeddings_3d = active_learner.get_embeddings_3d(reduction_method=dimension_reduction, projection=projection)

        # Get labels and uncertainties - apply the same subsampling as embeddings
        if active_learner.idx is not None:
            # Use the same subsampling indices
            labels = active_learner.labels[active_learner.idx].tolist()
            # Get labeled/unlabeled status for subsampled indices
            labeled_mask = [i in active_learner.labeled_indices for i in active_learner.idx]
            # Get uncertainties for subsampled indices
            uncertainties = active_learner.uncertainties[active_learner.idx].tolist()
        else:
            # No subsampling applied
            labels = active_learner.labels.tolist()
            labeled_mask = [i in active_learner.labeled_indices for i in range(len(embeddings_3d))]
            uncertainties = active_learner.uncertainties.tolist()

        # Debug: Check uncertainty range
        import numpy as np
        unc_array = np.array(uncertainties)
        logger.info(f"Uncertainties - min: {unc_array.min():.4f}, max: {unc_array.max():.4f}, mean: {unc_array.mean():.4f}")
        logger.info(f"Uncertainties shape: {unc_array.shape}, unique values: {len(np.unique(unc_array))}")
        print(f"Sample uncertainties: {uncertainties[:10]}")

        # Sanity check: clip to [0, 1] if values are out of range
        if unc_array.min() < 0 or unc_array.max() > 1:
            logger.warning(f"Uncertainties out of range [0, 1]! Clipping values.")
            uncertainties = np.clip(unc_array, 0, 1).tolist()

        return {
            "coordinates": embeddings_3d.tolist(),
            "labels": labels,
            "label_names": [active_learner.idx_to_label[label] for label in labels],
            "labeled_mask": labeled_mask,
            "uncertainties": uncertainties,  # Normalized uncertainty scores [0, 1]
            "state": active_learner.get_state()
        }

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/active-learning/state")
def get_active_learning_state():
    """
    Get current state of active learning pipeline

    Returns:
        Current state
    """
    global active_learner

    if active_learner is None:
        raise HTTPException(status_code=400, detail="Active learner not initialized. Call /initialize first.")

    return active_learner.get_state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
