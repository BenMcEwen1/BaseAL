import Analytics from './Analytics';

export default function AnalyticsV2({ isOpen, onClose, trainingHistory }) {
    if (!isOpen) return null;

    return (
        <div style={{
            position: 'absolute',
            right: 0,
            top: 0,
            width: '60vw',
            height: '100vh',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
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
                    fontSize: '14px',
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    zIndex: 11,
                    transition: 'background 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.background = '#ff5555'}
                onMouseLeave={(e) => e.target.style.background = '#e24a4a'}
            >
                Close Analytics
            </button>

            {/* Title */}
            <div style={{
                fontSize: '24px',
                fontWeight: 'bold',
                color: 'white',
                marginBottom: '30px',
                marginTop: '10px'
            }}>
                Analytics Dashboard
            </div>

            {/* Charts Container */}
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '20px',
                flex: 1
            }}>
                {trainingHistory && trainingHistory.length > 0 ? (
                    <>
                        <Analytics data={trainingHistory} />
                        <Analytics data={trainingHistory} />
                        {/* Additional charts can be added here */}
                        <div style={{
                            background: '#2a2a2a',
                            padding: '40px',
                            borderRadius: '8px',
                            color: '#666',
                            textAlign: 'center',
                            fontSize: '14px',
                            border: '2px dashed #444'
                        }}>
                            Additional analytics will be displayed here
                        </div>
                    </>
                ) : (
                    <div style={{
                        background: '#2a2a2a',
                        padding: '40px',
                        borderRadius: '8px',
                        color: 'white',
                        textAlign: 'center',
                        fontSize: '16px'
                    }}>
                        No training data available yet. Start training to see analytics.
                    </div>
                )}
            </div>
        </div>
    );
}