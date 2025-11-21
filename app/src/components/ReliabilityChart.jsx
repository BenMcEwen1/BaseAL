import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Scatter, Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function ReliabilityChart({ trainingHistory }) {
    console.log('Reliability Chart - Training History:', trainingHistory);

    // Guard against undefined or null data
    if (!trainingHistory || !Array.isArray(trainingHistory) || trainingHistory.length === 0) {
        return (
            <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px'
            }}>
                No calibration data available yet
            </div>
        );
    }

    // Get calibration data from the most recent training step
    const latestMetrics = trainingHistory[trainingHistory.length - 1];

    if (!latestMetrics.calibration) {
        return (
            <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px'
            }}>
                Calibration data not available for this experiment
            </div>
        );
    }

    const calibrationData = latestMetrics.calibration;
    const ece = calibrationData.ece;

    // Filter out bins with no data (null accuracy)
    const validBins = calibrationData.bin_confidences
        .map((conf, idx) => ({
            confidence: conf,
            accuracy: calibrationData.bin_accuracies[idx],
            count: calibrationData.bin_counts[idx]
        }))
        .filter(bin => bin.accuracy !== null && bin.count > 0);

    // Prepare data for the reliability plot
    const reliabilityData = {
        datasets: [
            // Perfect calibration line (diagonal)
            {
                label: 'Perfect Calibration',
                data: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 }
                ],
                borderColor: 'rgba(255, 255, 255, 0.5)',
                backgroundColor: 'rgba(24, 65, 36, 0.3)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: '+1',
                showLine: true,
                type: 'line'
            },
            // Actual calibration points
            {
                label: 'Model Calibration',
                data: validBins.map(bin => ({
                    x: bin.confidence,
                    y: bin.accuracy
                })),
                backgroundColor: 'rgba(74, 226, 144, 0.8)',
                borderColor: 'rgb(74, 226, 144)',
                borderWidth: 2,
                pointRadius: 6,
                pointHoverRadius: 8,
                showLine: true,
                tension: 0.1,
                type: 'line'
            }
        ]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'nearest',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: `Reliability Diagram (ECE: ${ece.toFixed(4)})`,
                color: 'white',
                font: { size: 16 }
            },
            legend: {
                labels: {
                    color: 'white'
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        if (context.datasetIndex === 0) {
                            // Perfect calibration line
                            return null;
                        }
                        const dataPoint = validBins[context.dataIndex];
                        return [
                            `Confidence: ${dataPoint.confidence.toFixed(3)}`,
                            `Accuracy: ${dataPoint.accuracy.toFixed(3)}`,
                            `Samples: ${dataPoint.count}`
                        ];
                    }
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                display: true,
                min: 0,
                max: 1,
                title: {
                    display: true,
                    text: 'Predicted Confidence',
                    color: 'white',
                    font: { size: 14 }
                },
                ticks: {
                    color: 'white',
                    callback: function(value) {
                        return value.toFixed(1);
                    }
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            y: {
                type: 'linear',
                display: true,
                min: 0,
                max: 1,
                title: {
                    display: true,
                    text: 'Actual Accuracy',
                    color: 'white',
                    font: { size: 14 }
                },
                ticks: {
                    color: 'white',
                    callback: function(value) {
                        return value.toFixed(1);
                    }
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        }
    };

    return (
        <div style={{
            background: 'rgba(255, 255, 255, 0.1)',
            padding: '16px 20px',
            borderRadius: '8px',
            height: '400px',
            display: 'flex',
            flexDirection: 'column'
        }}>
            {/* Info box */}
            <div style={{
                marginBottom: '10px',
                padding: '8px',
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '4px',
                fontSize: '12px',
                color: 'rgba(255, 255, 255, 0.8)'
            }}>
                Expected Calibration Error (ECE): <strong>{ece.toFixed(4)}</strong> |
                A well-calibrated model's points should align with the diagonal line
            </div>

            {/* Chart Container */}
            <div style={{ flex: 1, minHeight: 0 }}>
                <Line options={options} data={reliabilityData} />
            </div>
        </div>
    );
}
