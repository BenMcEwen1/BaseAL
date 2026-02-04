import BioDCASE from './pages/BioDCASE';

// Define available challenge pages
const CHALLENGE_PAGES = [
  {
    id: 'biodcase',
    title: 'BioDCASE - Active Learning for Bioacoustics',
    component: BioDCASE,
  },
  // Add more challenge pages here as needed:
  // { id: 'other-challenge', title: 'Other Challenge', component: OtherChallenge },
];

export default function Challenges({ isOpen, onClose, page }) {
  // Default to first challenge if no page specified or page not found
  const currentPage = CHALLENGE_PAGES.find(p => p.id === page) || CHALLENGE_PAGES[0];

  const handleSelectChallenge = (challengeId) => {
    window.location.hash = `/challenges/${challengeId}`;
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(6, 0, 20, 0.95)',
      backdropFilter: 'blur(10px)',
      zIndex: 1000,
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header with Close Button */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '20px 40px',
        borderBottom: '1px solid #2a2a2a'
      }}>
        <h2 style={{
          color: '#fff',
          margin: 0,
          fontSize: '24px',
          fontWeight: '300',
          letterSpacing: '2px'
        }}>
          Base<b style={{ fontWeight: '500' }}>AL</b> Challenges
        </h2>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            // border: '2px solid #666',
            borderRadius: '8px',
            width: '40px',
            height: '40px',
            cursor: 'pointer',
            display: 'flex',
            margin: '5px',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontSize: '24px',
            transition: 'all 0.2s'
          }}
          onMouseEnter={(e) => {
            e.target.style.borderColor = 'transparent';
            e.target.style.color = '#4ae290';
          }}
          onMouseLeave={(e) => {
            e.target.style.borderColor = 'transparent';
            e.target.style.color = '#fff';
          }}
        >
          ×
        </button>
      </div>

      {/* Main Content: Split Layout */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden'
      }}>
        {/* Left Sidebar: Navigation (33%) */}
        <div style={{
          width: '33%',
          background: '#060014ff',
          borderRight: '1px solid #2a2a2a',
          overflowY: 'auto',
          padding: '20px'
        }}>
          <h3 style={{
            color: '#999',
            fontSize: '12px',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            marginTop: 0,
            padding: '20px',
            marginBottom: '20px',
            fontWeight: '600'
          }}>
            
          </h3>
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}>
            {CHALLENGE_PAGES.map(challenge => (
              <button
                key={challenge.id}
                onClick={() => handleSelectChallenge(challenge.id)}
                style={{
                  padding: '16px 20px',
                  background: currentPage.id === challenge.id ? '#1a1a1a' : 'transparent',
                  border: currentPage.id === challenge.id ? '2px solid #4ae290' : '2px solid transparent',
                  borderRadius: '8px',
                  color: currentPage.id === challenge.id ? '#4ae290' : '#ccc',
                  fontSize: '16px',
                  fontWeight: currentPage.id === challenge.id ? '600' : '400',
                  textAlign: 'left',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  if (currentPage.id !== challenge.id) {
                    e.target.style.background = '#0f0f0f';
                    e.target.style.color = '#fff';
                  }
                }}
                onMouseLeave={(e) => {
                  if (currentPage.id !== challenge.id) {
                    e.target.style.background = 'transparent';
                    e.target.style.color = '#ccc';
                  }
                }}
              >
                {challenge.title}
              </button>
            ))}
          </div>
        </div>

        {/* Right Content Area: Challenge Page (66%) */}
        <div style={{
          width: '67%',
          background: '#060014',
          overflowY: 'auto'
        }}>
          <currentPage.component />
        </div>
      </div>
    </div>
  );
}