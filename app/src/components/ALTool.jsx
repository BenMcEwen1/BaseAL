import React, { useState, useEffect, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import PointCluster from './PointCluster';
// import Analytics from './Analytics';
import ManagerPanel from './ManagerPanel';
import AnalyticsV2 from './AnalyticsV2';
import Viewer from './Viewer';
import {
  fetchModels,
  fetchDatasets,
  // fetchEmbeddingSteps,
  // convertStepsToPointFormat,
  // initializeActiveLearning,
  // sampleNextBatch,
  // trainModel,
  getActiveLearningEmbeddings,
  getActiveLearningState,
  getManagerExperiments
} from '../utils/apiClient';

export default function ALTool() {
  const [step, setStep] = useState(0);
  const [embeddingSteps, setEmbeddingSteps] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useAPI, setUseAPI] = useState(true);
  // const [useActiveLearning, setUseActiveLearning] = useState(true);
  const [alState, setAlState] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  // const [n_samples, setSamples] = useState(10);
  const [labels, setLabels] = useState(null);
  const [labelNames, setLabelNames] = useState(null);
  const [labelIndicesForColor, setLabelIndicesForColor] = useState(null); // Primary label for color assignment
  const [labeledMask, setLabeledMask] = useState(null);
  const [uncertainties, setUncertainties] = useState(null);
  const [activeTab, setActiveTab] = useState('manager');
  // const [isTrainingAll, setIsTrainingAll] = useState(false);
  // const [isCancelling, setIsCancelling] = useState(false);
  // const cancelTrainingRef = useRef(false);
  const [isAnalyticsPanelOpen, setIsAnalyticsPanelOpen] = useState(false);
  const [experimentsData, setExperimentsData] = useState([]);

  // Visualization settings
  const [dimensionReduction, setDimensionReduction] = useState('UMAP');
  const [projection, setProjection] = useState('euclidean');

  // Data viewer
  const [id, setID] = useState(null)

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

  // Load experiments data when analytics panel is opened
  useEffect(() => {
    if (isAnalyticsPanelOpen) {
      loadExperimentsData();
    }
  }, [isAnalyticsPanelOpen]);

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

  // const loadEmbeddingsFromAPI = async () => {
  //   if (!selectedModel || !selectedDataset) {
  //     setError('Please select a model and dataset');
  //     return;
  //   }

  //   setLoading(true);
  //   setError(null);

  //   try {
  //     const stepsData = await fetchEmbeddingSteps(selectedModel, selectedDataset, 4);
  //     const formattedSteps = convertStepsToPointFormat(stepsData);
  //     setEmbeddingSteps(formattedSteps);
  //     setStep(0);
  //   } catch (err) {
  //     setError(`Failed to load embeddings: ${err.message}`);
  //     console.error(err);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  // const initializeAL = async () => {
  //   if (!selectedModel || !selectedDataset) {
  //     setError('Please select a model and dataset');
  //     return;
  //   }

  //   setLoading(true);
  //   setError(null);

  //   try {
  //     const result = await initializeActiveLearning(selectedModel, selectedDataset);
  //     setAlState(result.state);
  //     setTrainingMetrics(null);
  //     await updateEmbeddings();
  //   } catch (err) {
  //     setError(`Failed to initialize: ${err.message}`);
  //     console.error(err);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  // const sampleAndTrain = async () => {
  //   setLoading(true);
  //   setError(null);

  //   try {
  //     const sampleResult = await sampleNextBatch(n_samples);
  //     setAlState(sampleResult.state);

  //     const trainResult = await trainModel(10, 8);
  //     setAlState(trainResult.state);
  //     setTrainingMetrics(trainResult.metrics);

  //     await updateEmbeddings();
  //   } catch (err) {
  //     setError(`Failed to sample and train: ${err.message}`);
  //     console.error(err);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  // const trainAll = async () => {
  //   setLoading(true);
  //   setIsTrainingAll(true);
  //   setIsCancelling(false);
  //   setError(null);
  //   cancelTrainingRef.current = false;

  //   try {
  //     let currentState = alState;

  //     while (currentState.n_unlabeled > 0) {
  //       if (cancelTrainingRef.current) {
  //         console.log('Training cancelled by user');
  //         break;
  //       }

  //       console.log(`Unlabeled samples remaining: ${currentState.n_unlabeled}`);

  //       const sampleResult = await sampleNextBatch(n_samples);
  //       currentState = sampleResult.state;
  //       setAlState(currentState);

  //       const trainResult = await trainModel(10, 8);
  //       currentState = trainResult.state;
  //       setAlState(currentState);
  //       setTrainingMetrics(trainResult.metrics);

  //       await updateEmbeddings();
  //     }

  //     if (!cancelTrainingRef.current) {
  //       console.log('Training completed - all samples labeled');
  //     }
  //   } catch (err) {
  //     setError(`Failed to sample and train: ${err.message}`);
  //     console.error(err);
  //   } finally {
  //     setLoading(false);
  //     setIsTrainingAll(false);
  //     setIsCancelling(false);
  //     cancelTrainingRef.current = false;
  //   }
  // };

  // const cancelTraining = () => {
  //   cancelTrainingRef.current = true;
  //   setIsCancelling(true);
  // };

  // const updateEmbeddings = async () => {
  //   try {
  //     const data = await getActiveLearningEmbeddings({ dimensionReduction, projection });
  //     setEmbeddingSteps([data.coordinates]);
  //     setLabels(data.labels);
  //     setLabelNames(data.label_names);
  //     setLabeledMask(data.labeled_mask);
  //     setUncertainties(data.uncertainties);
  //     setStep(0);
  //   } catch (err) {
  //     console.error('Failed to update embeddings:', err);
  //   }
  // };

  const loadExperimentsData = async () => {
    try {
      const result = await getManagerExperiments();
      setExperimentsData(result.experiments || []);
    } catch (err) {
      console.error('Failed to load experiments data:', err);
    }
  };

  // Regenerate embeddings with current settings
  const regenerateEmbeddings = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getActiveLearningEmbeddings({ dimensionReduction, projection });
      setEmbeddingSteps([data.coordinates]);
      setLabels(data.labels);
      setLabelNames(data.label_names);
      setLabelIndicesForColor(data.label_indices_for_color || data.labels); // Fallback to labels if not provided
      setLabeledMask(data.labeled_mask);
      setUncertainties(data.uncertainties);
      setStep(0);
    } catch (err) {
      setError(`Failed to regenerate embeddings: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      overflow: 'hidden',
      background: '#060014ff',
      display: 'flex',
      position: 'relative'
    }}>
      <Viewer mediaID={id} setID={setID}/>


      {/* Left Panel */}
      <div style={{
        width: 'min(33.33%, 500px)',
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
          background: '#060014ff'
        }}>
          {/* <button
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
          </button> */}
          <button
            onClick={() => setActiveTab('manager')}
            style={{
              flex: 1,
              padding: '16px',
              background: activeTab === 'manager' ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
              color: 'white',
              border: 'none',
              borderBottom: activeTab === 'manager' ? '2px solid #4ae290' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            Manager
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            style={{
              flex: 1,
              padding: '16px',
              background: activeTab === 'settings' ? 'rgba(255, 255, 255, 0.2)' : 'transparent',
              color: 'white',
              border: 'none',
              borderBottom: activeTab === 'settings' ? '2px solid #4ae290' : '2px solid transparent',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold'
            }}
          >
            Settings
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
          {/* Settings Tab */}
          <div style={{ display: activeTab === 'settings' ? 'block' : 'none' }}>
            <div style={{
              background: 'rgba(255, 255, 255, 0.2)',
              padding: '16px 20px',
              borderRadius: '8px',
              color: 'white'
            }}>
              <div style={{ marginBottom: '20px', fontSize: '16px', fontWeight: 'bold' }}>
                Visualization Settings
              </div>

              {/* Dimension Reduction Method */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                  Dimension Reduction Method:
                </label>
                <select
                  value={dimensionReduction}
                  onChange={(e) => setDimensionReduction(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '4px',
                    border: '1px solid #444444',
                    background: 'rgba(47, 44, 58, 1)',
                    color: 'white',
                    fontSize: '14px'
                  }}
                >
                  <option value="UMAP">UMAP</option>
                  <option value="PCA">PCA</option>
                </select>
              </div>

              {/* Projection Method */}
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', fontWeight: 'bold' }}>
                  Projection Method:
                </label>
                <select
                  value={projection}
                  onChange={(e) => setProjection(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '10px',
                    borderRadius: '4px',
                    border: '1px solid #444',
                    background: 'rgba(47, 44, 58, 1)',
                    color: 'white',
                    fontSize: '14px'
                  }}
                >
                  <option value="euclidean">Euclidean</option>
                  <option value="spherical">Spherical</option>
                  <option value="torus">Torus</option>
                  <option value="hyperbolic">hyperbolic</option>
                </select>
              </div>

              {/* Regenerate Button */}
              <button
                onClick={regenerateEmbeddings}
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '14px',
                  fontSize: '14px',
                  background: loading ? '#666' : '#4ae290',
                  color: loading ? 'white' : '#000',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold'
                }}
              >
                {loading ? 'Regenerating...' : 'Regenerate Cluster'}
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
          </div>

          {/* Manager Tab */}
          <div style={{ display: activeTab === 'manager' ? 'block' : 'none' }}>
            <ManagerPanel
              dimensionReduction={dimensionReduction}
              projection={projection}
              onEmbeddingsUpdate={(data) => {
                setEmbeddingSteps([data.coordinates]);
                setLabels(data.labels);
                setLabelNames(data.label_names);
                setLabelIndicesForColor(data.label_indices_for_color || data.labels); // Fallback to labels if not provided
                setLabeledMask(data.labeled_mask);
                setUncertainties(data.uncertainties);
                setStep(0);
              }}
              onExperimentSelect={async (index) => {
                console.log(`Selected experiment ${index}`);
                // Fetch the state of the selected experiment
                try {
                  const state = await getActiveLearningState();
                  setAlState(state);
                  setTrainingMetrics(state.training_history && state.training_history.length > 0
                    ? state.training_history[state.training_history.length - 1]
                    : null);
                } catch (err) {
                  console.error('Failed to fetch experiment state:', err);
                }
              }}
              onTrainingUpdate={async () => {
                // Update state after each training cycle to refresh analytics
                try {
                  const state = await getActiveLearningState();
                  setAlState(state);
                  setTrainingMetrics(state.training_history && state.training_history.length > 0
                    ? state.training_history[state.training_history.length - 1]
                    : null);
                  // Load all experiments data for comparison chart
                  await loadExperimentsData();
                } catch (err) {
                  console.error('Failed to fetch updated state:', err);
                }
              }}
            />
          </div>
        </div>
      </div>

      {/* Right Canvas */}
      <div style={{ width: '66.67%', height: '100vh', position: 'absolute', right: '0px' }}>
        {/* Toggle Analytics Button */}
        {!isAnalyticsPanelOpen && (
          <button
            onClick={() => setIsAnalyticsPanelOpen(true)}
            style={{
              position: 'absolute',
              top: '20px',
              right: '20px',
              background: '#4ae290',
              color: '#0a0a0a',
              border: 'none',
              borderRadius: '8px',
              padding: '12px 24px',
              fontSize: '14px',
              fontWeight: 'bold',
              cursor: 'pointer',
              zIndex: 5,
              transition: 'background 0.2s',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
            }}
            onMouseEnter={(e) => e.target.style.background = '#5af3a0'}
            onMouseLeave={(e) => e.target.style.background = '#4ae290'}
          >
            Open Analytics
          </button>
        )}

        {/* Analytics Overlay Panel */}
        <AnalyticsV2
          isOpen={isAnalyticsPanelOpen}
          onClose={() => setIsAnalyticsPanelOpen(false)}
          trainingHistory={alState?.training_history}
          experimentsData={experimentsData}
        />

        {/* 3D Canvas */}
        <Canvas camera={{ position: [5, 5, 5], fov: 20 }} raycaster={{ params: { Points: { threshold: 0.03 } } }}>
          <ambientLight intensity={0.8} />
          <pointLight position={[10, 10, 10]} intensity={0.5} />
          <PointCluster
            setID={setID}
            selectedID={id}
            embeddingData={embeddingSteps}
            currentStep={step}
            labels={labels}
            labelNames={labelNames}
            labelIndicesForColor={labelIndicesForColor}
            labeledMask={labeledMask}
            uncertainties={uncertainties}
          />
          <OrbitControls
            enableDamping
            dampingFactor={0.04}
            autoRotate={true}
            autoRotateSpeed={0.1}
            target={[0, 0, 0]}
            minDistance={3}
            maxDistance={20}
          />
        </Canvas>
      </div>
    </div>
  );
}
