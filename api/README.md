# BaseAL Embeddings API

FastAPI backend for loading and serving embedding data from the bacpipe results.

## Features

- Load embeddings from .npy files
- Automatic dimension reduction from 1024D to 3D using PCA
- Progressive clustering steps visualization
- CORS enabled for React frontend

## Installation

From the root directory of the project:

```bash
uv sync
```

This will install all Python dependencies from the root [pyproject.toml](../pyproject.toml).

## Running the API

From the root directory:

```bash
uv run python api/main.py
```

Or using uvicorn directly:

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `GET /`
Root endpoint, returns API info.

### `GET /api/models`
List all available embedding models.

**Response:**
```json
{
  "models": [
    {
      "name": "2025-11-09_10-27___birdnet-test_data",
      "path": "2025-11-09_10-27___birdnet-test_data"
    }
  ]
}
```

### `GET /api/embeddings/{model_name}/datasets`
List all datasets for a specific model.

**Response:**
```json
{
  "model": "2025-11-09_10-27___birdnet-test_data",
  "datasets": [
    {
      "name": "FewShot",
      "file_count": 4
    }
  ]
}
```

### `GET /api/embeddings/{model_name}/{dataset_name}/3d`
Get embeddings reduced to 3D coordinates.

**Response:**
```json
{
  "model": "2025-11-09_10-27___birdnet-test_data",
  "dataset": "FewShot",
  "total_samples": 88,
  "files": [
    {
      "filename": "CHE_01_20190101_163410_birdnet.npy",
      "file_index": 0,
      "n_samples": 22,
      "coordinates": [[x, y, z], ...]
    }
  ]
}
```

### `GET /api/embeddings/{model_name}/{dataset_name}/steps?n_steps=4`
Get embeddings as progressive clustering steps.

**Parameters:**
- `n_steps`: Number of steps to generate (default: 4)

**Response:**
```json
{
  "model": "2025-11-09_10-27___birdnet-test_data",
  "dataset": "FewShot",
  "total_samples": 88,
  "n_steps": 4,
  "steps": [
    [[x1, y1, z1], [x2, y2, z2], ...],  // Step 1
    [[x1, y1, z1], [x2, y2, z2], ...],  // Step 2
    ...
  ]
}
```

## Data Structure

The API expects embeddings to be stored in the following structure:

```
bacpipe_results/
  test_data/
    embeddings/
      {model_name}/
        audio/
          {dataset_name}/
            *.npy
```

Each `.npy` file should contain a numpy array of shape `(n_samples, 1024)` with float32 dtype.

## Dimension Reduction

The API uses PCA (Principal Component Analysis) to reduce the 1024-dimensional embeddings to 3D for visualization. The process:

1. Load all embeddings from the specified dataset
2. Concatenate into a single array
3. Standardize features using StandardScaler
4. Apply PCA to reduce to 3 dimensions
5. Return the 3D coordinates

For the `/steps` endpoint, multiple PCA reductions are performed with varying parameters to simulate a clustering progression from random to well-separated clusters.