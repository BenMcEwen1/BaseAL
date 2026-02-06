import React, { useState } from 'react';
import Analytics from './Analytics';
import ComparisonChart from './ComparisonChart';
import ReliabilityChart from './ReliabilityChart';

export default function AnalyticsV2({ isOpen, onClose, trainingHistory, experimentsData }) {
    const [selectedMetric, setSelectedMetric] = useState('accuracy');

    if (!isOpen) return null;

    return (
        <div style={{
            position: 'absolute',
            right: 0,
            top: 0,
            width: '60vw',
            height: '100vh',
            backgroundColor: 'rgba(6, 0, 20, 0.6)', 
            backdropFilter: 'blur(10px)',
            zIndex: 10,
            display: 'flex',
            flexDirection: 'column',
            padding: '20px',
            overflowY: 'auto'
        }}>
            {/* Close Button */}
            <button
                onClick={onClose}
                style={{
                    position: 'absolute',
                    top: '20px',
                    right: '20px',
                    background: '#e24a4a',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    padding: '12px 24px',
                    fontSize: '11px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    zIndex: 11,
                    transition: 'background 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.background = '#ff5555'}
                onMouseLeave={(e) => e.target.style.background = '#e24a4a'}
            >
                Close
            </button>

            {/* Title */}
            <div style={{
                fontSize: '19px',
                fontWeight: 'bold',
                color: 'white',
                marginBottom: '30px',
                marginTop: '10px'
            }}>
                Dashboard
            </div>

            {/* Charts Container */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))',
                gap: '20px',
                flex: 1
            }}>
                {trainingHistory && trainingHistory.length > 0 ? (
                    <>
                        {/* Comparison Chart Across All Experiments */}
                        {experimentsData && experimentsData.length > 1 && (
                        <ComparisonChart
                            experimentsData={experimentsData}
                            selectedMetric={selectedMetric}
                            onMetricChange={setSelectedMetric}
                        />
                        )}

                        {/* Reliability/Calibration Chart */}
                        <ReliabilityChart
                            trainingHistory={trainingHistory}
                        />

                        {/* Single Experiment Charts */}
                        <Analytics
                            data={trainingHistory}
                            selectedMetric={selectedMetric}
                            onMetricChange={setSelectedMetric}
                        />
                    </>
                ) : (
                    <div style={{
                        background: 'transparent',
                        padding: '16px 20px',
                        borderRadius: '8px',
                        color: 'white',
                        textAlign: 'center', 
                        fontSize: '13px'
                    }}>
                        No training data available yet. Start training to see analytics.
                    </div>
                )}
            </div>
        </div>
    );
}