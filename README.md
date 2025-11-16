# BaseAL - Active Learning Baseline

<div align="center">
  <img src="app/public/baseAL_logo.png" alt="BaseAL Logo" width="200"/>
</div>

## Overview

BaseAL is an interactive web-based framework for developing and evaluating active learning methods, with a focus on audio and bioacoustic data. The tool provides a complete pipeline for embedding generation, intelligent sampling strategies, 3D visualization, and annotation workflows.

Built with a modular architecture, BaseAL serves as both a testing ground for researchers exploring active learning methodologies and a practical tool for efficient data annotation campaigns.

### Key Features

- **Interactive 3D Visualization**: Explore high-dimensional embeddings using UMAP reduction with an interactive Three.js interface
- **Multiple Sampling Strategies**: Compare random sampling, entropy-based selection, and cluster-based diversification
- **Model Integration**: Built on [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) for seamless integration with bioacoustic models
- **Flexible Data Support**: Works with .mp3 and .wav files of any length or sample rate
- **Real-time Updates**: FastAPI backend with live embedding computation and cluster progression
- **Experiment Management**: Track and compare different active learning configurations

## Technology Stack

- **Frontend**: React 19 + Vite, Three.js for 3D visualization, Chart.js for metrics
- **Backend**: FastAPI with PyTorch for model inference
- **ML Pipeline**: UMAP for dimensionality reduction, scikit-learn for clustering
- **Audio Processing**: Bacpipe integration for bioacoustic models

## Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **Git**: For cloning the repository

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BaseAL.git
cd BaseAL
```

### 2. Set Up the Backend (API Server)

```bash
cd api
pip install -r requirements.txt
```

**Requirements**: FastAPI, PyTorch, scikit-learn, numpy, pandas, UMAP

### 3. Set Up the Frontend (Web Interface)

```bash
cd app
npm install
```

## Quick Start

### 1. Start the API Server

```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

You can verify it's running by visiting `http://localhost:8000/docs` for the interactive API documentation.

### 2. Start the Web Interface

```bash
cd app
npm run dev
```

The web interface will be available at `http://localhost:5173`

### 3. Explore Embeddings

1. Open `http://localhost:5173` in your browser
2. Check **"Use Real Embeddings (API)"** to connect to the backend
3. Select a **model** and **dataset** from the dropdown menus
4. Click **"Load Embeddings"** to generate and visualize embeddings
5. Use **Run/Run All** buttons to step through AL cycles
6. Interact with the 3D scatter plot to explore your data

## Usage Workflow

### Basic Active Learning Session

1. **Prepare Your Data**: Place audio files in a directory (flat structure, no subdirectories)
2. **Configure Model**: Select from available bioacoustic models via the interface
3. **Choose Sampling Strategy**:
   - **Random**: Baseline random selection
   - **Entropy**: Uncertainty-based sampling for multilabel classification
   - **Cluster-based**: Diversity-promoting selection
4. **Generate Embeddings**: Click "Load Embeddings" (initial processing may take time)
5. **Annotate Samples**: Use the interface to label samples suggested by the active learner
6. **Iterate**: Repeat the process to progressively improve model performance

### Sampling Methods

BaseAL provides several sampling strategies:

- **Random**: Baseline sampling for comparison
- **Entropy-based**: Selects samples where the model is most uncertain
- **Cluster-based**: Promotes diversity by sampling across embedding space clusters
- **Stratified**: Leverages metadata for balanced sampling (custom implementation)

## Model Integration

BaseAL uses [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) for embedding and prediction generation. Bacpipe provides wrappers for common bioacoustic models including:

- BirdNET
- Perch
- Custom PyTorch models

Refer to the [Bacpipe documentation](https://github.com/bioacoustic-ai/bacpipe) for available models and configuration options.

## Dataset Requirements

- **Format**: .wav or .mp3 files
- **Structure**: Flat directory (no subdirectories)
- **Metadata**: Optional CSV with additional sample information
- **Labels**: Optional ground truth labels for baseline performance metrics
- **Processing**: Audio is automatically segmented and resampled based on model requirements

## Development

### Running in Development Mode

**Backend with auto-reload**:
```bash
cd api
uvicorn main:app --reload
```

**Frontend with HMR**:
```bash
cd app
npm run dev
```

### Building for Production

**Frontend build**:
```bash
cd app
npm run build
```

The optimized static files will be in `app/dist/`.

## Contributing

Contributions are welcome! This framework is designed to be modular and extensible. Areas for contribution include:

- Additional sampling strategies
- New model integrations
- Visualization improvements
- Performance optimizations

## Roadmap

- [ ] Audio playback integration (Wavesurfer.js)
- [ ] Export annotations in multiple formats
- [ ] Support for image/vision models
- [ ] Enhanced experiment comparison tools
- [ ] Batch annotation workflows

## Acknowledgments

- Built on [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) for bioacoustic model integration
- Uses UMAP for dimensionality reduction
- Inspired by ModAL for active learning functionality
