# BaseAL - Active Learning Baseline (Audio)

## TODO
- [ ] Verify setup of active learning pipeline, create separate script which prints output
- [ ] Review bacpipe embedding and prediction generation, annotation format etc

## Quick Start

### 1. Start the API Server

```bash
cd api
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Web Interface

```bash
cd app
npm install
npm run dev
```

The web interface will be available at `http://localhost:5173`

### 3. Visualize Your Embeddings

1. Open the web interface in your browser
2. Check "Use Real Embeddings (API)"
3. Select a model and dataset from the dropdowns
4. Click "Load Embeddings"
5. Use the Previous/Next buttons to step through the clustering progression

## Planning
ModAL - basic active learning functionality
Bacpipe - Embedding generation and bioacoustics model integration
Equivalent for image feature extractors?

Interface - TBD, wavesurfer? I think provide a super simple python (streamlit) interface then setup the main functionality as an API wrapper so that a JS interface could optionally be built, in which case BaseAL functions as a backend server.


## Aim
Provide a framework for the development of active learning methods and tools. This framework is designed to be simple and modular, perfect as a testing ground for building and evaluating active sampling and annotation methods.

# Installation

# Getting Started
To run the annotator you need three things - a **model**, **dataset**, and **sampling method**. Defaults are already provided so you can get started using the tool straight away.

The only two **bacpipe** files that need to be modified are the config.yaml and settings.yaml files. The config.yaml is used for the standard configurations:

Below the defaults are specified, as well as information about how to customise each ->

## Model Integration
BaseAL is built upon [Bacpipe](https://github.com/bioacoustic-ai/bacpipe) which is used for generation of embeddings, predictions and clusters. Bacpipe is a wrapper over most common (bio)acoustics [models](https://github.com/bioacoustic-ai/bacpipe?tab=readme-ov-file#available-models) (you can provide your own).  

### Custom
TBD

## Datasets
You can specify the path to your dataset. The data should be provided simply as a directory containing audio files (No sub-directories). Both .mp3 and .wav are supported and audio of any length and sample rate can be provided but note that audio will be segmented and resampled according to the selected model requirements. Additional metadata can be provided with audio data which can then be used for sampling. If no audio is provided, ESC50 is provided as a starting point. If audio labels are provided, baseline performance (e.g. random, performance ceiling etc) will be pre-computed.

### Custom
TBD

## Sampling Methods
By default, three baseline sampling methods are provide:
- Random - Baseline
- *Entropy (binary in the case of multilabel)

Diversification:
- None
- *Clustering
- Stratified - Exampling of using metadata

*Default*

### Custom
TBD




# Workflow
1. Select audio directory
2. Select model from drop down
3. Select sampling method and parameters
4. **Generate** model embeddings and predictions (Note - this can take some time)
5. Start annotating.
