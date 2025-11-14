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
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Analytics({ data: trainingHistory }) {
    console.log('Training History:', trainingHistory);

    // Guard against undefined or null data
    if (!trainingHistory || !Array.isArray(trainingHistory) || trainingHistory.length === 0) {
        return (
            <div style={{
                background: '#2a2a2a',
                padding: '12px 20px',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px'
            }}>
                No training data available yet
            </div>
        );
    }

    // Transform training history into Chart.js format
    const epochs = trainingHistory.map((_, index) => index + 1);
    const losses = trainingHistory.map(entry => entry.loss);
    const accuracies = trainingHistory.map(entry => entry.accuracy);

    const chartData = {
        labels: epochs,
        datasets: [
            {
                label: 'Loss',
                data: losses,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                yAxisID: 'y',
            },
            {
                label: 'Accuracy',
                data: accuracies,
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                yAxisID: 'y1',
            }
        ]
    };

    const options = {
        responsive: true,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            title: {
                display: true,
                text: 'Training Metrics',
                color: 'white'
            },
            legend: {
                labels: {
                    color: 'white'
                }
            }
        },
        scales: {
            y: {
                type: 'linear',
                display: true,
                position: 'left',
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
            y1: {
                type: 'linear',
                display: true,
                position: 'right',
                title: {
                    display: true,
                    text: 'Accuracy',
                    color: 'white'
                },
                ticks: {
                    color: 'white'
                },
                grid: {
                    drawOnChartArea: false,
                },
            },
            x: {
                title: {
                    display: true,
                    text: 'Epoch',
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
            background: '#2a2a2a',
            padding: '16px 20px',
            borderRadius: '8px',
            width: '100%'
        }}>
            <Line options={options} data={chartData} />
        </div>
    );
}