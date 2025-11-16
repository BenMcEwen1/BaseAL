/**
 * API Client for BaseAL Embeddings API
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Fetch available models from the API
 * @returns {Promise<Array>} List of available models
 */
export async function fetchModels() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/models`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.models;
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
}

/**
 * Fetch available datasets for a given model
 * @param {string} modelName - Name of the model
 * @returns {Promise<Array>} List of available datasets
 */
export async function fetchDatasets(modelName) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/embeddings/${encodeURIComponent(modelName)}/datasets`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.datasets;
  } catch (error) {
    console.error('Error fetching datasets:', error);
    throw error;
  }
}

/**
 * Fetch 3D embeddings for a given model and dataset
 * @param {string} modelName - Name of the model
 * @param {string} datasetName - Name of the dataset
 * @returns {Promise<Object>} 3D embeddings data
 */
export async function fetch3DEmbeddings(modelName, datasetName) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/embeddings/${encodeURIComponent(modelName)}/${encodeURIComponent(datasetName)}/3d`
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching 3D embeddings:', error);
    throw error;
  }
}

/**
 * Fetch embedding steps showing progressive clustering
 * @param {string} modelName - Name of the model
 * @param {string} datasetName - Name of the dataset
 * @param {number} nSteps - Number of steps to generate (default: 4)
 * @returns {Promise<Object>} Embedding steps data
 */
export async function fetchEmbeddingSteps(modelName, datasetName, nSteps = 4) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/embeddings/${encodeURIComponent(modelName)}/${encodeURIComponent(datasetName)}/steps?n_steps=${nSteps}`
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching embedding steps:', error);
    throw error;
  }
}

/**
 * Convert embedding steps API response to the format expected by PointCluster
 * @param {Object} stepsData - API response from fetchEmbeddingSteps
 * @returns {Array<Array<[number, number, number]>>} Array of steps with 3D coordinates
 */
export function convertStepsToPointFormat(stepsData) {
  if (!stepsData || !stepsData.steps) {
    throw new Error('Invalid steps data');
  }

  // Convert each step from array of [x, y, z] arrays to the expected format
  return stepsData.steps.map(step => {
    return step.map(coord => [coord[0], coord[1], coord[2]]);
  });
}

// ==================== Active Learning API ====================

/**
 * Initialize active learning pipeline
 * @param {string} modelName - Name of the model
 * @param {string} datasetName - Name of the dataset
 * @returns {Promise<Object>} Initialization status and state
 */
export async function initializeActiveLearning(modelName, datasetName) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/active-learning/initialize?model_name=${encodeURIComponent(modelName)}&dataset_name=${encodeURIComponent(datasetName)}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error initializing active learning:', error);
    throw error;
  }
}

/**
 * Sample next batch for active learning
 * @param {number} nSamples - Number of samples to select (default: 5)
 * @returns {Promise<Object>} Selected indices and updated state
 */
export async function sampleNextBatch(nSamples = 200) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/active-learning/sample?n_samples=${nSamples}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error sampling:', error);
    throw error;
  }
}

/**
 * Train model on current labeled set
 * @param {number} epochs - Number of training epochs (default: 10)
 * @param {number} batchSize - Batch size (default: 8)
 * @returns {Promise<Object>} Training metrics and updated state
 */
export async function trainModel(epochs = 5, batchSize = 8) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/active-learning/train?epochs=${epochs}&batch_size=${batchSize}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error training:', error);
    throw error;
  }
}

/**
 * Get 3D embeddings from trained model
 * @returns {Promise<Object>} 3D coordinates and labels
 */
export async function getActiveLearningEmbeddings() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/active-learning/embeddings-3d`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error getting embeddings:', error);
    throw error;
  }
}

/**
 * Get current state of active learning pipeline
 * @returns {Promise<Object>} Current state
 */
export async function getActiveLearningState() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/active-learning/state`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error getting state:', error);
    throw error;
  }
}

// ==================== Manager API ====================

/**
 * Initialize Manager with config file
 * @param {string} configPath - Path to config file (default: "core/config.yml")
 * @returns {Promise<Object>} Status and initial summary
 */
export async function initializeManager(configPath = "core/config.yml") {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/manager/initialize?config_path=${encodeURIComponent(configPath)}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error initializing manager:', error);
    throw error;
  }
}

/**
 * Add new experiment to manager
 * @param {Object} experimentConfig - Experiment configuration
 * @returns {Promise<Object>} Updated summary
 */
export async function addExperimentToManager(experimentConfig) {
  try {
    const params = new URLSearchParams(experimentConfig);
    const response = await fetch(
      `${API_BASE_URL}/api/manager/add-experiment?${params.toString()}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error adding experiment:', error);
    throw error;
  }
}

/**
 * Run one AL cycle across all experiments
 * @param {number} nSamples - Number of samples per experiment
 * @param {number} epochs - Training epochs
 * @param {number} batchSize - Batch size
 * @param {boolean} parallel - Run in parallel
 * @returns {Promise<Object>} Results for each experiment
 */
export async function runManagerCycle(nSamples = 5, epochs = 5, batchSize = 8, parallel = false) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/manager/run?n_samples=${nSamples}&epochs=${epochs}&batch_size=${batchSize}&parallel=${parallel}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error running manager cycle:', error);
    throw error;
  }
}

/**
 * Get list of all experiments
 * @returns {Promise<Object>} List of experiments
 */
export async function getManagerExperiments() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/manager/experiments`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error getting experiments:', error);
    throw error;
  }
}

/**
 * Get manager summary
 * @returns {Promise<Object>} Summary of all experiments
 */
export async function getManagerSummary() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/manager/summary`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error getting summary:', error);
    throw error;
  }
}

/**
 * Save manager results
 * @param {string} outputDir - Output directory path
 * @returns {Promise<Object>} Save status
 */
export async function saveManagerResults(outputDir = "results/manager_experiments") {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/manager/save?output_dir=${encodeURIComponent(outputDir)}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error saving results:', error);
    throw error;
  }
}

/**
 * Select which experiment to use for single-experiment endpoints
 * @param {number} experimentIndex - Index of experiment to select
 * @returns {Promise<Object>} Selected experiment info
 */
export async function selectExperiment(experimentIndex) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/manager/select-experiment?experiment_index=${experimentIndex}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error selecting experiment:', error);
    throw error;
  }
}
