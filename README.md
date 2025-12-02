<h1>
  <img src="app/public/baseAL_logo.png" alt="Logo" width="35" height="30">
  BaseAL - Active Learning Baseline
</h1>

## Overview

BaseAL is an framework for developing and evaluating active learning methods. Built upon [Bacpipe](https://github.com/bioacoustic-ai/bacpipe), BaseAL (v1.0.0) focuses on audio and bioacoustic data. 

The tool provides a complete pipeline for embedding generation, evaluating sampling strategies and 3D visualisation.

![Demo](demo.gif)
*Demo of 3D visualisation.*

### Key Features

- **Experiment Management**: Track and compare different active learning configurations
- **Interactive 3D Visualisation**: Explore high-dimensional embeddings using PCA/UMAP reduction with an interactive Three.js interface
- **Multiple Sampling Strategies**: Compare different sampling and diversification strategies
- **Model Integration**: Built on [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) for seamless integration with bioacoustic models


# Setup

## Prerequisites

- **Python**: 3.11 or 3.12
- **uv**: Package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/BaseAL.git
cd BaseAL
```

### 2. Set Up Python Dependencies (Backend)

```bash
uv sync
```

This will install all Python dependencies including the workspace member `bacpipe`.

**Dependencies**: FastAPI, PyTorch, scikit-learn, numpy, pandas, UMAP, librosa, matplotlib, and more (see [pyproject.toml](pyproject.toml))

### 3. Set Up the Frontend (Web Interface)

```bash
cd app
npm install
cd ..
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

Open a new terminal window and run:

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

## Acknowledgments

BaseAL is built on [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) developed by [Vincent Kather](https://github.com/vskode).
