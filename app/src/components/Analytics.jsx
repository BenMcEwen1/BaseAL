import React, { useState } from 'react';
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
import { Line } from 'react-chartjs-2';

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

export default function Analytics({ data: trainingHistory, selectedMetric = 'accuracy', onMetricChange = null }) {
    console.log('Training History:', trainingHistory);

    // Guard against undefined or null data
    if (!trainingHistory || !Array.isArray(trainingHistory) || trainingHistory.length === 0) {
        return (
            <div style={{
                background: '#2a2a2a',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontSize: '11px'
            }}>
                No training data available yet
            </div>
        );
    }

    // Transform training history into Chart.js format
    // Loss data - no initial point, starts from cycle 1
    const lossSteps = trainingHistory.map((_, index) => index + 1);
    const losses = trainingHistory.map(entry => entry.loss || 0);
    const lossesSD = trainingHistory.map(entry => entry.loss_sd || 0);

    // Performance metrics - add initial point at (0, 0) before first AL cycle
    const steps = [0, ...trainingHistory.map((_, index) => index + 1)];
    const accuracies = [0, ...trainingHistory.map(entry => entry.accuracy || 0)];
    const f1Scores = [0, ...trainingHistory.map(entry => entry.f1_score || 0)];
    const mAPs = [0, ...trainingHistory.map(entry => entry.mAP || 0)];

    // Extract standard deviations (0 for initial point)
    const accuraciesSD = [0, ...trainingHistory.map(entry => entry.accuracy_sd || 0)];
    const f1ScoresSD = [0, ...trainingHistory.map(entry => entry.f1_score_sd || 0)];
    const mAPsSD = [0, ...trainingHistory.map(entry => entry.mAP_sd || 0)];

    // Helper function to create upper/lower bounds for shaded regions
    const createBounds = (data, sd) => {
        return {
            upper: data.map((val, idx) => val + sd[idx]),
            lower: data.map((val, idx) => val - sd[idx])
        };
    };

    const lossBounds = createBounds(losses, lossesSD);
    const accuracyBounds = createBounds(accuracies, accuraciesSD);
    const f1Bounds = createBounds(f1Scores, f1ScoresSD);
    const mAPBounds = createBounds(mAPs, mAPsSD);

    // Chart for Loss with shaded region
    const lossChartData = {
        labels: lossSteps,
        datasets: [
            // Upper bound (invisible line for fill)
            {
                label: 'Loss Upper',
                data: lossBounds.upper,
                borderColor: 'transparent',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                pointRadius: 0,
                fill: '+1',
                tension: 0.1
            },
            // Main line
            {
                label: 'Loss',
                data: losses,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: '+1'  // Fill to next dataset (lower bound)
            },
            // Lower bound (invisible line)
            {
                label: 'Loss Lower',
                data: lossBounds.lower,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false,
                tension: 0.1
            }
        ]
    };

    // Get data for selected metric
    const getMetricData = (metric) => {
        switch(metric) {
            case 'accuracy':
                return { data: accuracies, bounds: accuracyBounds, color: 'rgb(53, 162, 235)', label: 'Accuracy' };
            case 'f1_score':
                return { data: f1Scores, bounds: f1Bounds, color: 'rgb(75, 192, 192)', label: 'F1 Score' };
            case 'mAP':
                return { data: mAPs, bounds: mAPBounds, color: 'rgb(255, 205, 86)', label: 'mAP' };
            default:
                return { data: accuracies, bounds: accuracyBounds, color: 'rgb(53, 162, 235)', label: 'Accuracy' };
        }
    };

    const metricInfo = getMetricData(selectedMetric);

    // Chart for selected performance metric with shaded region
    const performanceChartData = {
        labels: steps,
        datasets: [
            // Upper bound
            {
                label: `${metricInfo.label} Upper`,
                data: metricInfo.bounds.upper,
                borderColor: 'transparent',
                backgroundColor: metricInfo.color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                pointRadius: 0,
                fill: '+1',
                tension: 0.1
            },
            // Main line
            {
                label: metricInfo.label,
                data: metricInfo.data,
                borderColor: metricInfo.color,
                backgroundColor: metricInfo.color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                tension: 0.1,
                pointRadius: 4,
                pointHoverRadius: 6,
                fill: '+1'
            },
            // Lower bound
            {
                label: `${metricInfo.label} Lower`,
                data: metricInfo.bounds.lower,
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false,
                tension: 0.1
            }
        ]
    };

    const lossOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: 'Training Loss',
                color: 'white',
                font: { size: 13 }
            },
            legend: {
                labels: {
                    color: 'white',
                    filter: (item) => !item.text.includes('Upper') && !item.text.includes('Lower')
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        // Only show tooltip for main line (index 1)
                        if (context.datasetIndex === 1) {
                            const value = context.parsed.y;
                            const sd = lossesSD[context.dataIndex];
                            if (sd > 0) {
                                return `${context.dataset.label}: ${value.toFixed(4)} ± ${sd.toFixed(4)}`;
                            }
                            return `${context.dataset.label}: ${value.toFixed(4)}`;
                        }
                        return null;
                    }
                },
                filter: (item) => item.datasetIndex === 1
            }
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                min: 0,
                title: {
                    display: true,
                    text: 'Loss',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'AL Cycle',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        }
    };

    const performanceOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: metricInfo.label,
                color: 'white',
                font: { size: 13 }
            },
            legend: {
                labels: {
                    color: 'white',
                    filter: (item) => !item.text.includes('Upper') && !item.text.includes('Lower')
                }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        // Only show tooltip for main line (index 1)
                        if (context.datasetIndex === 1) {
                            const value = context.parsed.y;
                            const sdArray = selectedMetric === 'accuracy' ? accuraciesSD :
                                          selectedMetric === 'f1_score' ? f1ScoresSD : mAPsSD;
                            const sd = sdArray[context.dataIndex];
                            if (sd > 0) {
                                return `${context.dataset.label}: ${value.toFixed(4)} ± ${sd.toFixed(4)}`;
                            }
                            return `${context.dataset.label}: ${value.toFixed(4)}`;
                        }
                        return null;
                    }
                },
                filter: (item) => item.datasetIndex === 1
            }
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                min: 0,
                max: 1,
                title: {
                    display: true,
                    text: 'Score',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'AL Cycle',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.1)'
                }
            }
        }
    };

    return (
        <div style={{
            // display: 'flex',
            // flexDirection: 'row',
            // gap: '20px',
            // width: '100%'
        }}>
            {/* Performance Metric Chart with Dropdown */}
            <div style={{
                display: 'flex',
                // gap: '15px',
                alignItems: 'stretch'
            }}>
                

                {/* Chart */}
                {/* <div style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    padding: '16px 20px',
                    borderRadius: '8px',
                    flex: 1,
                    height: '400px'
                }}>
                    <Line options={performanceOptions} data={performanceChartData} />
                </div> */}
            </div>
            {/* Loss Chart */}
            <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '16px 20px',
                borderRadius: '8px',
                // width: 'min(100%, 25vw)',
                height: '400px'
            }}>
                <Line options={lossOptions} data={lossChartData} />
            </div>

            
        </div>
    );
}