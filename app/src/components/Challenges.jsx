import Baselines from './pages/Baselines';
import BioDCASE from './pages/BioDCASE';
import Datasets from './pages/Datasets';
import Setup from './pages/Setup';

// Define available challenge pages
const CHALLENGE_PAGES = [
  {
    id: 'biodcase',
    title: 'BioDCASE - Active Learning for Bioacoustics',
    component: BioDCASE,
    subpages: [
      { id: 'instructions', title: 'Instructions', component: Setup },
      { id: 'datasets', title: 'Datasets', component: Datasets },
      { id: 'baselines', title: 'Baselines', component: Baselines },
      // Add more subpages here as needed
    ]
  },
  // Add more challenge pages here as needed:
  { id: 'other-challenge', title: 'Other Challenge', component: Setup, subpages: [] },
];

export default function Challenges({ isOpen, onClose, page }) {
  // Parse the page to extract challenge and optional subpage (e.g., 'biodcase/setup')
  const [challengeId, subpageId] = (page || '').split('/');

  // Find the current challenge
  const currentChallenge = CHALLENGE_PAGES.find(p => p.id === challengeId) || CHALLENGE_PAGES[0];

  // Find the current subpage if specified
  const currentSubpage = subpageId
    ? currentChallenge.subpages?.find(s => s.id === subpageId)
    : null;

  // Determine which component to render
  const CurrentComponent = currentSubpage?.component || currentChallenge.component;

  const handleSelectChallenge = (challengeId) => {
    window.location.hash = `/challenges/${challengeId}`;
  };

  const handleSelectSubpage = (challengeId, subpageId) => {
    window.location.hash = `/challenges/${challengeId}/${subpageId}`;
  };

  // Check if a challenge's subpages should be visible (when challenge or any of its subpages is selected)
  const isExpanded = (challenge) => {
    return currentChallenge.id === challenge.id;
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
            {CHALLENGE_PAGES.map(challenge => {
              const isChallengeActive = currentChallenge.id === challenge.id && !currentSubpage;
              const isChallengeExpanded = isExpanded(challenge);

              return (
                <div key={challenge.id}>
                  <button
                    onClick={() => handleSelectChallenge(challenge.id)}
                    style={{
                      width: '100%',
                      padding: '16px 20px',
                      background: isChallengeActive ? '#1a1a1a' : 'transparent',
                      border: isChallengeActive ? '2px solid #4ae290' : '2px solid transparent',
                      borderRadius: '8px',
                      color: isChallengeActive ? '#4ae290' : (isChallengeExpanded ? '#fff' : '#ccc'),
                      fontSize: '16px',
                      fontWeight: isChallengeActive || isChallengeExpanded ? '600' : '400',
                      textAlign: 'left',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (!isChallengeActive) {
                        e.target.style.background = '#0f0f0f';
                        e.target.style.color = '#fff';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!isChallengeActive) {
                        e.target.style.background = 'transparent';
                        e.target.style.color = isChallengeExpanded ? '#fff' : '#ccc';
                      }
                    }}
                  >
                    {challenge.title}
                  </button>

                  {/* Subpages - only visible when challenge is expanded */}
                  {isChallengeExpanded && challenge.subpages?.length > 0 && (
                    <div style={{
                      marginLeft: '20px',
                      marginTop: '4px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px'
                    }}>
                      {challenge.subpages.map(subpage => {
                        const isSubpageActive = currentSubpage?.id === subpage.id;

                        return (
                          <button
                            key={subpage.id}
                            onClick={() => handleSelectSubpage(challenge.id, subpage.id)}
                            style={{
                              padding: '12px 16px',
                              background: isSubpageActive ? '#1a1a1a' : 'transparent',
                              border: isSubpageActive ? '2px solid #4ae290' : '2px solid transparent',
                              borderRadius: '6px',
                              color: isSubpageActive ? '#4ae290' : '#aaa',
                              fontSize: '14px',
                              fontWeight: isSubpageActive ? '600' : '400',
                              textAlign: 'left',
                              cursor: 'pointer',
                              transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => {
                              if (!isSubpageActive) {
                                e.target.style.background = '#0f0f0f';
                                e.target.style.color = '#fff';
                              }
                            }}
                            onMouseLeave={(e) => {
                              if (!isSubpageActive) {
                                e.target.style.background = 'transparent';
                                e.target.style.color = '#aaa';
                              }
                            }}
                          >
                            {subpage.title}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Content Area: Challenge Page (66%) */}
        <div style={{
          width: '67%',
          background: '#060014',
          overflowY: 'auto'
        }}>
          <CurrentComponent />
        </div>
      </div>
    </div>
  );
}