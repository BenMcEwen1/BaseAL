export default function Axes({ opacity = 1 }) {
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 10,
            opacity: opacity,
            transition: 'opacity 0.3s ease-out'
        }}>
            <div style={{
                marginLeft: '33vw',
                width: '8px',
                height: '8px',
                backgroundColor: 'rgba(255, 255, 255, 0.6)',
                borderRadius: '100%',
                boxShadow: '0 0 10px rgba(255, 255, 255, 0.4), 0 0 20px rgba(255, 255, 255, 0.2)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
            }}>
                {/* Vertical Axis */}
                <div style={{
                    position: 'absolute',
                    width: '1.5px',
                    height: '150px',
                    background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.1))',
                    boxShadow: '0 0 4px rgba(255, 255, 255, 0.3)'
                }} />

                {/* 60° Axis */}
                <div style={{
                    position: 'absolute',
                    width: '1.5px',
                    height: '150px',
                    background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.1))',
                    transform: 'rotate(-120deg)',
                    boxShadow: '0 0 4px rgba(255, 255, 255, 0.3)'
                }} />

                {/* -60° Axis */}
                <div style={{
                    position: 'absolute',
                    width: '1.5px',
                    height: '150px',
                    background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0.6), rgba(74, 226, 144, 0.1))',
                    transform: 'rotate(120deg)',
                    boxShadow: '0 0 4px rgba(255, 255, 255, 0.3)'
                }} />
            </div>
        </div>
    );
}