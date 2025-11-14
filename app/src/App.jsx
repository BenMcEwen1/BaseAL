import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import PointCluster from './components/PointCluster';
import { generateEmbeddingSteps } from './utils/generatePoints';
import {
  fetchModels,
  fetchDatasets,
  fetchEmbeddingSteps,
  convertStepsToPointFormat,
  initializeActiveLearning,
  sampleNextBatch,
  trainModel,
  getActiveLearningEmbeddings
} from './utils/apiClient';

export default function App() {
  const [step, setStep] = useState(0);
  const [embeddingSteps, setEmbeddingSteps] = useState(generateEmbeddingSteps(500, 5));
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useAPI, setUseAPI] = useState(true);
  const [useActiveLearning, setUseActiveLearning] = useState(true);
  const [alState, setAlState] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [n_samples, setSamples] = useState(10);
  const [labels, setLabels] = useState(null);
  const [labelNames, setLabelNames] = useState(null);
  const [labeledMask, setLabeledMask] = useState(null);

  // Load models on component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        const modelsList = await fetchModels();
        setModels(modelsList);
        if (modelsList.length > 0) {
          setSelectedModel(modelsList[0].name);
        }
      } catch (err) {
        console.error('Failed to load models:', err);
      }
    };

    if (useAPI) {
      loadModels();
    }
  }, [useAPI]);

  // Load datasets when model changes
  useEffect(() => {
    const loadDatasets = async () => {
      if (!selectedModel) return;

      try {
        const datasetsList = await fetchDatasets(selectedModel);
        setDatasets(datasetsList);
        if (datasetsList.length > 0) {
          setSelectedDataset(datasetsList[0].name);
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };

    if (useAPI && selectedModel) {
      loadDatasets();
    }
  }, [selectedModel, useAPI]);

  // Load embeddings from API
  const loadEmbeddingsFromAPI = async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Please select a model and dataset');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const stepsData = await fetchEmbeddingSteps(selectedModel, selectedDataset, 4);
      const formattedSteps = convertStepsToPointFormat(stepsData);
      setEmbeddingSteps(formattedSteps);
      setStep(0);
    } catch (err) {
      setError(`Failed to load embeddings: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Active Learning functions
  const initializeAL = async () => {
    if (!selectedModel || !selectedDataset) {
      setError('Please select a model and dataset');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await initializeActiveLearning(selectedModel, selectedDataset);
      setAlState(result.state);
      setTrainingMetrics(null);

      // Load initial embeddings
      await updateEmbeddings();
    } catch (err) {
      setError(`Failed to initialize: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const sampleAndTrain = async () => {
    setLoading(true);
    setError(null);

    try {
      // Sample next batch
      const sampleResult = await sampleNextBatch(n_samples);
      setAlState(sampleResult.state);

      // Train model
      const trainResult = await trainModel(10, 8);
      setAlState(trainResult.state);
      setTrainingMetrics(trainResult.metrics);

      // Update embeddings visualization
      await updateEmbeddings();
    } catch (err) {
      setError(`Failed to sample and train: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const updateEmbeddings = async () => {
    try {
      const data = await getActiveLearningEmbeddings();
      // Convert to point format (single step)
      setEmbeddingSteps([data.coordinates]);
      setLabels(data.labels);
      setLabelNames(data.label_names);
      setLabeledMask(data.labeled_mask);
      setStep(0);
    } catch (err) {
      console.error('Failed to update embeddings:', err);
    }
  };

  const nextStep = () => {
    setStep(s => (s + 1) % embeddingSteps.length);
  };

  const prevStep = () => {
    setStep(s => (s - 1 + embeddingSteps.length) % embeddingSteps.length);
  };
  
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0a0a0a' }}>
      <Canvas camera={{ position: [8, 8, 8], fov: 60 }}>
        <ambientLight intensity={0.8} />
        <pointLight position={[10, 10, 10]} intensity={0.5} />
        <pointLight position={[-10, -10, -10]} intensity={0.3} />
        <PointCluster
          embeddingData={embeddingSteps}
          currentStep={step}
          labels={labels}
          labelNames={labelNames}
          labeledMask={labeledMask}
        />
        <OrbitControls enableDamping dampingFactor={0.04} />
      </Canvas>
      
      {/* Controls Panel */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        maxWidth: '400px'
      }}>
        {/* <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          background: '#2a2a2a',
          padding: '12px 20px',
          borderRadius: '8px'
        }}>
          <label style={{ color: 'white', fontWeight: 'bold', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useAPI}
              onChange={(e) => setUseAPI(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Use Real Embeddings (API)
          </label>
          {useAPI && (
            <label style={{ color: 'white', fontWeight: 'bold', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={useActiveLearning}
                onChange={(e) => setUseActiveLearning(e.target.checked)}
                style={{ marginRight: '8px' }}
              />
              Active Learning Mode
            </label>
          )}
        </div> */}

        {/* Model and Dataset Selection */}
        {useAPI && (
          <div style={{
            background: '#2a2a2a',
            padding: '12px 20px',
            borderRadius: '8px',
            color: 'white'
          }}>
            <div style={{ marginBottom: '10px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px' }}>
                Model:
              </label>
              <select
                value={selectedModel || ''}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #444',
                  background: '#1a1a1a',
                  color: 'white'
                }}
              >
                {models.map(model => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div style={{ marginBottom: '10px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px' }}>
                Dataset:
              </label>
              <select
                value={selectedDataset || ''}
                onChange={(e) => setSelectedDataset(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #444',
                  background: '#1a1a1a',
                  color: 'white'
                }}
              >
                {datasets.map(dataset => (
                  <option key={dataset.name} value={dataset.name}>
                    {dataset.name} ({dataset.file_count} files)
                  </option>
                ))}
              </select>

              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px' }}>
                Sample number (n):
              </label>
              <input type='number' onChange={(e) => setSamples(e.target.value)}/>
              <p>{n_samples}</p>
            </div>

            <button
              onClick={useActiveLearning ? initializeAL : loadEmbeddingsFromAPI}
              disabled={loading || !selectedModel || !selectedDataset}
              style={{
                width: '100%',
                padding: '10px',
                fontSize: '14px',
                background: loading ? '#666' : (useActiveLearning ? '#e24a90' : '#4a90e2'),
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: loading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {loading ? 'Loading...' : (useActiveLearning ? 'Initialize Active Learning' : 'Load Embeddings')}
            </button>

            {error && (
              <div style={{
                marginTop: '10px',
                padding: '8px',
                background: '#ff4444',
                borderRadius: '4px',
                fontSize: '12px'
              }}>
                {error}
              </div>
            )}
          </div>
        )}

        {/* Active Learning Controls or Step Navigation */}
        { useActiveLearning && alState && (
          <div style={{
            background: '#2a2a2a',
            padding: '12px 20px',
            borderRadius: '8px',
            color: 'white'
          }}>
            <div style={{ marginBottom: '10px', fontSize: '14px' }}>
              <div>Labeled: {alState.n_labeled} / {alState.n_labeled + alState.n_unlabeled}</div>
              {trainingMetrics && (
                <>
                  <div>Accuracy: {(trainingMetrics.accuracy * 100).toFixed(2)}%</div>
                  <div>Loss: {trainingMetrics.loss.toFixed(4)}</div>
                </>
              )}
            </div>

            <button
              onClick={sampleAndTrain}
              disabled={loading || alState.n_unlabeled === 0}
              style={{
                width: '100%',
                padding: '12px 20px',
                fontSize: '16px',
                background: loading || alState.n_unlabeled === 0 ? '#666' : '#4ae290',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: loading || alState.n_unlabeled === 0 ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {loading ? 'Training...' : 'Sample & Train (5 samples)'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}