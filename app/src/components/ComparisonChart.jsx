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

export default function ComparisonChart({ experimentsData, selectedMetric }) {
    console.log('Comparison Chart - Experiments Data:', experimentsData);
    console.log('Selected Metric:', selectedMetric);

    // Guard against undefined or null data
    if (!experimentsData || experimentsData.length === 0) {
        return (
            <div style={{
                background: 'rgba(255, 255, 255, 0.1)',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px'
            }}>
                No experiment data available yet
            </div>
        );
    }

    // Color palette for different experiments
    const colors = [
        'rgb(53, 162, 235)',   // Blue
        'rgb(255, 99, 132)',   // Red
        'rgb(75, 192, 192)',   // Teal
        'rgb(255, 205, 86)',   // Yellow
        'rgb(153, 102, 255)',  // Purple
        'rgb(255, 159, 64)',   // Orange
        'rgb(201, 203, 207)',  // Gray
        'rgb(255, 99, 255)',   // Pink
    ];

    // Get metric label and data key
    const getMetricInfo = (metric) => {
        switch(metric) {
            case 'accuracy':
                return { label: 'Accuracy', dataKey: 'accuracy', sdKey: 'accuracy_sd' };
            case 'f1_score':
                return { label: 'F1 Score', dataKey: 'f1_score', sdKey: 'f1_score_sd' };
            case 'mAP':
                return { label: 'mAP', dataKey: 'mAP', sdKey: 'mAP_sd' };
            default:
                return { label: 'Accuracy', dataKey: 'accuracy', sdKey: 'accuracy_sd' };
        }
    };

    const metricInfo = getMetricInfo(selectedMetric);

    // Helper function to create upper/lower bounds for shaded regions
    const createBounds = (data, sd) => {
        return {
            upper: data.map((val, idx) => val + sd[idx]),
            lower: data.map((val, idx) => val - sd[idx])
        };
    };

    // Find the maximum number of steps across all experiments
    const maxSteps = Math.max(...experimentsData.map(exp =>
        exp.training_history ? exp.training_history.length : 0
    ));
    const steps = Array.from({ length: maxSteps }, (_, i) => i + 1);

    // Create datasets for each experiment
    const datasets = [];
    experimentsData.forEach((exp, expIdx) => {
        const color = colors[expIdx % colors.length];
        const trainingHistory = exp.training_history || [];

        if (trainingHistory.length === 0) return;

        // Extract metric data and SD
        const metricData = trainingHistory.map(entry => entry[metricInfo.dataKey] || 0);
        const metricSD = trainingHistory.map(entry => entry[metricInfo.sdKey] || 0);
        const bounds = createBounds(metricData, metricSD);

        // Add upper bound (invisible)
        datasets.push({
            label: `${exp.name} Upper`,
            data: bounds.upper,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.1
        });

        // Add main line
        datasets.push({
            label: exp.name,
            data: metricData,
            borderColor: color,
            backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.15)'),
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6,
            fill: '+1'  // Fill to next dataset (lower bound)
        });

        // Add lower bound (invisible)
        datasets.push({
            label: `${exp.name} Lower`,
            data: bounds.lower,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            pointRadius: 0,
            fill: false,
            tension: 0.1
        });
    });

    const chartData = {
        labels: steps,
        datasets: datasets
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: `${metricInfo.label} Comparison Across Experiments`,
                color: 'white',
                font: { size: 16 }
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
                        // Only show tooltip for main lines (every 3rd dataset starting from index 1)
                        if ((context.datasetIndex - 1) % 3 === 0) {
                            const value = context.parsed.y;
                            const expIdx = Math.floor(context.datasetIndex / 3);
                            const exp = experimentsData[expIdx];
                            const trainingHistory = exp.training_history || [];
                            const entry = trainingHistory[context.dataIndex];

                            if (entry && entry[metricInfo.sdKey]) {
                                const sd = entry[metricInfo.sdKey];
                                if (sd > 0) {
                                    return `${context.dataset.label}: ${value.toFixed(4)} ± ${sd.toFixed(4)}`;
                                }
                            }
                            return `${context.dataset.label}: ${value.toFixed(4)}`;
                        }
                        return null;
                    }
                },
                filter: (item) => (item.datasetIndex - 1) % 3 === 0
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
                    text: metricInfo.label,
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
                    text: 'AL Step',
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
            background: 'rgba(255, 255, 255, 0.2)',
            padding: '16px 20px',
            borderRadius: '8px',
            width: '100%',
            height: '400px'
        }}>
            <Line options={options} data={chartData} />
        </div>
    );
}
