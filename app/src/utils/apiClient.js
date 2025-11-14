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
