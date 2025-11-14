import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import PointCluster from './components/PointCluster';
import { generateEmbeddingSteps } from './utils/generatePoints';
import Analytics from './components/Analytics'
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
  const [activeTab, setActiveTab] = useState('controls'); // 'controls' or 'analytics'

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
    <div style={{
      width: '100vw',
      height: '100vh',
      overflow: 'hidden',
      background: '#0a0a0a',
      display: 'flex'
    }}>
      {/* Left Panel - 1/3 width */}
      <div style={{
        width: '33.33%',
        height: '100vh',
        padding: '15px 20px',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        overflowX: 'hidden'
      }}>
        {/* Tab Navigation */}
        <div style={{
          display: 'flex',
          padding: '15px 20px',
          borderBottom: '1px solid #444',
          background: '#0a0a0a'
        }}>
          <button
            onClick={() => setActiveTab('controls')}
            style={{
              flex: 1,
              padding: '16px',
              background: activeTab === 'controls' ? '#2a2a2a' : 'transparent',
              color: 'white',
              border: 'none',
              borderBottom: activeTab === 'controls' ? '2px solid #4ae290' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            Controls
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            style={{
              flex: 1,
              padding: '16px',
              background: activeTab === 'analytics' ? '#2a2a2a' : 'transparent',
              color: 'white',
              border: 'none',
              borderBottom: activeTab === 'analytics' ? '2px solid #4ae290' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            Analytics
          </button>
        </div>

        {/* Tab Content */}
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px'
        }}>
          {activeTab === 'controls' && (
            <>
              {/* Model and Dataset Selection */}
              {useAPI && (
                <div style={{
                  background: '#2a2a2a',
                  padding: '16px 20px',
                  borderRadius: '8px',
                  color: 'white'
                }}>
                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                      Model:
                    </label>
                    <select
                      value={selectedModel || ''}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '4px',
                        border: '1px solid #444',
                        background: '#1a1a1a',
                        color: 'white',
                        fontSize: '14px'
                      }}
                    >
                      {models.map(model => (
                        <option key={model.name} value={model.name}>
                          {model.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                      Dataset:
                    </label>
                    <select
                      value={selectedDataset || ''}
                      onChange={(e) => setSelectedDataset(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '4px',
                        border: '1px solid #444',
                        background: '#1a1a1a',
                        color: 'white',
                        fontSize: '14px'
                      }}
                    >
                      {datasets.map(dataset => (
                        <option key={dataset.name} value={dataset.name}>
                          {dataset.name} ({dataset.file_count} files)
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ marginBottom: '12px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                      Sample number (n):
                    </label>
                    <input
                      type='number'
                      value={n_samples}
                      onChange={(e) => setSamples(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '10px',
                        borderRadius: '4px',
                        border: '1px solid #444',
                        background: '#1a1a1a',
                        color: 'white',
                        fontSize: '14px'
                      }}
                    />
                  </div>

                  <button
                    onClick={useActiveLearning ? initializeAL : loadEmbeddingsFromAPI}
                    disabled={loading || !selectedModel || !selectedDataset}
                    style={{
                      width: '100%',
                      padding: '12px',
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
                      marginTop: '12px',
                      padding: '10px',
                      background: '#ff4444',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      {error}
                    </div>
                  )}
                </div>
              )}

              {/* Active Learning Controls */}
              {useActiveLearning && alState && (
                <div style={{
                  background: '#2a2a2a',
                  padding: '16px 20px',
                  borderRadius: '8px',
                  color: 'white'
                }}>
                  <div style={{ marginBottom: '16px', fontSize: '14px' }}>
                    <div style={{ marginBottom: '8px', fontSize: '16px', fontWeight: 'bold' }}>
                      Training Status
                    </div>
                    <div style={{ marginBottom: '4px' }}>
                      Labeled: {alState.n_labeled} / {alState.n_labeled + alState.n_unlabeled}
                    </div>
                    {trainingMetrics && (
                      <>
                        <div style={{ marginBottom: '4px' }}>
                          Accuracy: {(trainingMetrics.accuracy * 100).toFixed(2)}%
                        </div>
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
                    {loading ? 'Training...' : `Sample & Train (${n_samples} samples)`}
                  </button>
                </div>
              )}
            </>
          )}

          {activeTab === 'analytics' && (
            <>
              {/* Analytics Charts */}
              {alState && alState.training_history ? (
                <>
                  <Analytics data={alState.training_history} />
                  {/* Placeholder for additional charts */}
                  <Analytics data={alState.training_history} />
                </>
              ) : (
                <div style={{
                  background: '#2a2a2a',
                  padding: '20px',
                  borderRadius: '8px',
                  color: 'white',
                  textAlign: 'center',
                  fontSize: '14px'
                }}>
                  No analytics data available yet. Start training to see charts.
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Right Canvas - 2/3 width */}
      <div style={{ width: '66.67%', height: '100vh' }}>
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
      </div>
    </div>
  );
}