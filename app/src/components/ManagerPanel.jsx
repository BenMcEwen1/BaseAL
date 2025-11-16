import React, { useState, useEffect } from 'react';
import {
  initializeManager,
  addExperimentToManager,
  runManagerCycle,
  getManagerExperiments,
  getManagerSummary,
  saveManagerResults,
  selectExperiment,
  getActiveLearningEmbeddings
} from '../utils/apiClient';

export default function ManagerPanel({ onEmbeddingsUpdate, onExperimentSelect, onTrainingUpdate }) {
  const [configPath, setConfigPath] = useState('core/config.yml');
  const [experiments, setExperiments] = useState([]);
  const [selectedExperimentIndex, setSelectedExperimentIndex] = useState(null);
  const [managerSummary, setManagerSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [runResults, setRunResults] = useState(null);

  // Add experiment form state
  const [showAddForm, setShowAddForm] = useState(false);
  const [newExpName, setNewExpName] = useState('');
  const [newExpLearningRate, setNewExpLearningRate] = useState(0.0001);

  // Manager run settings
  const [runSettings, setRunSettings] = useState({
    n_samples: 10,
    epochs: 5,
    batch_size: 8,
    parallel: false
  });

  // Run all and cancel state
  const [isRunningAll, setIsRunningAll] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showRunDropdown, setShowRunDropdown] = useState(false);
  const cancelRunRef = React.useRef(false);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showRunDropdown && !event.target.closest('.run-button-container')) {
        setShowRunDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showRunDropdown]);

  const initManager = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await initializeManager(configPath);
      setManagerSummary(result.summary);
      await loadExperiments();
    } catch (err) {
      setError(`Failed to initialize: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const loadExperiments = async () => {
    try {
      const result = await getManagerExperiments();
      setExperiments(result.experiments);
      if (result.experiments.length > 0 && selectedExperimentIndex === null) {
        // Auto-select the first experiment
        await handleSelectExperiment(0);
      }
    } catch (err) {
      console.error('Failed to load experiments:', err);
    }
  };

  const handleSelectExperiment = async (index) => {
    setLoading(true);
    try {
      await selectExperiment(index);
      setSelectedExperimentIndex(index);

      // Update embeddings for this experiment
      const data = await getActiveLearningEmbeddings();
      onEmbeddingsUpdate(data);
      onExperimentSelect(index);
    } catch (err) {
      setError(`Failed to select experiment: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const addExperiment = async () => {
    if (!newExpName) {
      setError('Please enter experiment name');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await addExperimentToManager({
        name: newExpName,
        learning_rate: newExpLearningRate,
        model_name: '2025-11-13_21-42___birdnet-test_data',
        dataset_name: 'esc50'
      });

      await loadExperiments();
      setShowAddForm(false);
      setNewExpName('');
      setNewExpLearningRate(0.0001);
    } catch (err) {
      setError(`Failed to add experiment: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runManager = async () => {
    setLoading(true);
    setError(null);
    setShowRunDropdown(false);

    try {
      const result = await runManagerCycle(
        runSettings.n_samples,
        runSettings.epochs,
        runSettings.batch_size,
        runSettings.parallel
      );

      setRunResults(result.results);
      setManagerSummary(result.summary);
      await loadExperiments();

      // Update embeddings for currently selected experiment
      if (selectedExperimentIndex !== null) {
        const data = await getActiveLearningEmbeddings();
        onEmbeddingsUpdate(data);
        // Update analytics with latest training data
        if (onTrainingUpdate) {
          await onTrainingUpdate();
        }
      }
    } catch (err) {
      setError(`Failed to run manager: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runManagerAll = async () => {
    setLoading(true);
    setIsRunningAll(true);
    setIsCancelling(false);
    setError(null);
    setShowRunDropdown(false);
    cancelRunRef.current = false;

    try {
      // Keep running cycles until cancelled or all experiments are done
      let cyclesRun = 0;
      const maxCycles = 100; // Safety limit

      while (cyclesRun < maxCycles) {
        if (cancelRunRef.current) {
          console.log('Manager run cancelled by user');
          break;
        }

        // Check if any experiments still have unlabeled data
        const experimentsData = await getManagerExperiments();
        const hasUnlabeled = experimentsData.experiments.some(exp => exp.n_unlabeled > 0);

        if (!hasUnlabeled) {
          console.log('All experiments completed - no unlabeled samples remaining');
          break;
        }

        console.log(`Running manager cycle ${cyclesRun + 1}...`);

        const result = await runManagerCycle(
          runSettings.n_samples,
          runSettings.epochs,
          runSettings.batch_size,
          runSettings.parallel
        );

        setRunResults(result.results);
        setManagerSummary(result.summary);
        await loadExperiments();

        // Update embeddings for currently selected experiment
        if (selectedExperimentIndex !== null) {
          const data = await getActiveLearningEmbeddings();
          onEmbeddingsUpdate(data);
          // Update analytics with latest training data
          if (onTrainingUpdate) {
            await onTrainingUpdate();
          }
        }

        cyclesRun++;
      }

      if (!cancelRunRef.current && cyclesRun >= maxCycles) {
        setError('Reached maximum cycle limit (100). Stopped for safety.');
      }
    } catch (err) {
      setError(`Failed to run all: ${err.message}`);
    } finally {
      setLoading(false);
      setIsRunningAll(false);
      setIsCancelling(false);
      cancelRunRef.current = false;
    }
  };

  const cancelRun = () => {
    cancelRunRef.current = true;
    setIsCancelling(true);
  };

  const saveResults = async () => {
    setLoading(true);
    try {
      await saveManagerResults();
      alert('Results saved successfully!');
    } catch (err) {
      setError(`Failed to save: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Initialize Manager */}
      <div style={{
        background: '#2a2a2a',
        padding: '16px 20px',
        borderRadius: '8px',
        color: 'white'
      }}>
        <div style={{ marginBottom: '12px', fontSize: '16px', fontWeight: 'bold' }}>
          Initialize Manager
        </div>
        <div style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px' }}>
            Config Path:
          </label>
          <input
            type="text"
            value={configPath}
            onChange={(e) => setConfigPath(e.target.value)}
            style={{
              width: '80%',
              padding: '10px 0px 10px 8px',
              borderRadius: '4px',
              border: '1px solid #444',
              background: '#1a1a1a',
              color: 'white',
              fontSize: '14px'
            }}
          />
        </div>
        <button
          onClick={initManager}
          disabled={loading}
          style={{
            width: '100%',
            padding: '12px',
            fontSize: '14px',
            background: loading ? '#666' : '#e24a90',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          {loading ? 'Initializing...' : 'Initialize Manager'}
        </button>
      </div>

      {/* Experiments List */}
      {experiments.length > 0 && (
        <div style={{
          background: '#2a2a2a',
          padding: '16px 20px',
          borderRadius: '8px',
          color: 'white'
        }}>
          <div style={{ marginBottom: '12px', fontSize: '16px', fontWeight: 'bold' }}>
            Experiments ({experiments.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {experiments.map((exp, idx) => (
              <div
                key={idx}
                onClick={() => handleSelectExperiment(idx)}
                style={{
                  padding: '12px',
                  background: selectedExperimentIndex === idx ? '#4ae290' : '#1a1a1a',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  color: selectedExperimentIndex === idx ? '#000' : '#fff'
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{exp.name}</div>
                <div>LR: {exp.learning_rate} | Labeled: {exp.n_labeled}/{exp.n_labeled + exp.n_unlabeled}</div>
              </div>
            ))}
          </div>

          <button
            onClick={() => setShowAddForm(!showAddForm)}
            style={{
              width: '100%',
              marginTop: '12px',
              padding: '10px',
              fontSize: '13px',
              background: '#4a90e2',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {showAddForm ? 'Cancel' : '+ Add Experiment'}
          </button>

          {showAddForm && (
            <div style={{
              marginTop: '12px',
              padding: '12px',
              background: '#1a1a1a',
              borderRadius: '4px'
            }}>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                  Name:
                </label>
                <input
                  type="text"
                  value={newExpName}
                  onChange={(e) => setNewExpName(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '8px',
                    borderRadius: '4px',
                    border: '1px solid #444',
                    background: '#0a0a0a',
                    color: 'white',
                    fontSize: '12px'
                  }}
                />
              </div>
              <div style={{ marginBottom: '8px' }}>
                <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                  Learning Rate:
                </label>
                <input
                  type="number"
                  step="0.0001"
                  value={newExpLearningRate}
                  onChange={(e) => setNewExpLearningRate(parseFloat(e.target.value))}
                  style={{
                    width: '100%',
                    padding: '8px',
                    borderRadius: '4px',
                    border: '1px solid #444',
                    background: '#0a0a0a',
                    color: 'white',
                    fontSize: '12px'
                  }}
                />
              </div>
              <button
                onClick={addExperiment}
                disabled={loading}
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '12px',
                  background: loading ? '#666' : '#4ae290',
                  color: loading ? 'white' : '#000',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: 'bold'
                }}
              >
                Add
              </button>
            </div>
          )}
        </div>
      )}

      {/* Run Manager */}
      {experiments.length > 0 && (
        <div style={{
          background: '#2a2a2a',
          padding: '16px 20px',
          borderRadius: '8px',
          color: 'white'
        }}>
          <div style={{ marginBottom: '12px', fontSize: '16px', fontWeight: 'bold' }}>
            Run Manager Cycle
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                Samples:
              </label>
              <input
                type="number"
                value={runSettings.n_samples}
                onChange={(e) => setRunSettings({ ...runSettings, n_samples: parseInt(e.target.value) })}
                style={{
                  width: '80%',
                  padding: '8px 0px 8px 8px',
                  borderRadius: '4px',
                  border: '1px solid #444',
                  background: '#1a1a1a',
                  color: 'white',
                  fontSize: '12px'
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px' }}>
                Epochs:
              </label>
              <input
                type="number"
                value={runSettings.epochs}
                onChange={(e) => setRunSettings({ ...runSettings, epochs: parseInt(e.target.value) })}
                style={{
                  width: '80%',
                  padding: '8px 0px 8px 8px',
                  borderRadius: '4px',
                  border: '1px solid #444',
                  background: '#1a1a1a',
                  color: 'white',
                  fontSize: '12px'
                }}
              />
            </div>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'flex', alignItems: 'center', fontSize: '13px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={runSettings.parallel}
                onChange={(e) => setRunSettings({ ...runSettings, parallel: e.target.checked })}
                style={{ marginRight: '8px' }}
              />
              Run experiments in parallel
            </label>
          </div>

          {/* Split button for Run Cycle / Run All */}
          {isRunningAll ? (
            <button
              onClick={cancelRun}
              disabled={isCancelling}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '14px',
                background: isCancelling ? '#999' : '#e24a4a',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: isCancelling ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {isCancelling ? 'Wrapping up current cycle...' : 'Cancel Run All'}
            </button>
          ) : (
            <div className="run-button-container" style={{ position: 'relative', width: '100%' }}>
              <div style={{ display: 'flex', gap: '0' }}>
                {/* Main Run Cycle button */}
                <button
                  onClick={runManager}
                  disabled={loading}
                  style={{
                    flex: 1,
                    padding: '12px',
                    fontSize: '14px',
                    background: loading ? '#666' : '#4ae290',
                    color: loading ? 'white' : '#000',
                    border: 'none',
                    borderRadius: '4px 0 0 4px',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontWeight: 'bold'
                  }}
                >
                  {loading ? 'Running...' : 'Run Cycle'}
                </button>

                {/* Dropdown toggle button */}
                <button
                  onClick={() => setShowRunDropdown(!showRunDropdown)}
                  disabled={loading}
                  style={{
                    width: '36px',
                    padding: '12px 8px',
                    fontSize: '14px',
                    background: loading ? '#666' : '#4ae290',
                    color: loading ? 'white' : '#000',
                    border: 'none',
                    borderLeft: loading ? 'none' : '1px solid rgba(0, 0, 0, 0.2)',
                    borderRadius: '0 4px 4px 0',
                    cursor: loading ? 'not-allowed' : 'pointer',
                    fontWeight: 'bold',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  ▼
                </button>
              </div>

              {/* Dropdown menu */}
              {showRunDropdown && !loading && (
                <div
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    marginTop: '4px',
                    background: '#4ae290',
                    color: '#000',
                    border: '1px solid #444',
                    borderRadius: '4px',
                    zIndex: 1000,
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)'
                  }}
                >
                  <button
                    onClick={runManagerAll}
                    style={{
                      width: '100%',
                      padding: '12px',
                      fontSize: '14px',
                      background: 'transparent',
                      color: '#000',
                      border: 'none',
                      cursor: 'pointer',
                      textAlign: 'left',
                      fontWeight: 'normal',
                      transition: 'background 0.2s'
                    }}
                    onMouseEnter={(e) => e.target.style.background = '#2a2a2a'}
                    onMouseLeave={(e) => e.target.style.background = 'transparent'}
                  >
                    Run All Cycles
                  </button>
                </div>
              )}
            </div>
          )}

          {runResults && (
            <div style={{ marginTop: '12px', fontSize: '12px' }}>
              <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Last Cycle Results:</div>
              {Object.entries(runResults).map(([name, metrics]) => (
                <div key={name} style={{ marginBottom: '4px', padding: '6px', background: '#1a1a1a', borderRadius: '4px' }}>
                  <div style={{ fontWeight: 'bold' }}>{name}</div>
                  {metrics.error ? (
                    <div style={{ color: '#ff4444' }}>Error: {metrics.error}</div>
                  ) : (
                    <div>
                      Acc: {(metrics.accuracy * 100).toFixed(2)}% | Loss: {metrics.loss.toFixed(4)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Save Results */}
      {experiments.length > 0 && (
        <button
          onClick={saveResults}
          disabled={loading}
          style={{
            padding: '12px',
            fontSize: '14px',
            background: loading ? '#666' : '#4a90e2',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 'bold'
          }}
        >
          Save All Results
        </button>
      )}

      {/* Error Display */}
      {error && (
        <div style={{
          padding: '10px',
          background: '#ff4444',
          borderRadius: '4px',
          fontSize: '12px',
          color: 'white'
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
