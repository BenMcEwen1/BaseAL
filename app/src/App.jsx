import React, { useRef, useState, useEffect } from 'react';
import Home from './components/Home';
import ALTool from './components/ALTool';
import Axes from './components/Axes'
// import AnalyticsV2 from './components/AnalyticsV2';
import Docs from './components/Docs';
import Challenges from './components/Challenges';

// Parse hash to get current route
const parseHash = () => {
  const hash = window.location.hash.slice(1); // Remove leading #
  const parts = hash.split('/').filter(Boolean);
  return {
    section: parts[0] || null,  // e.g., 'challenges', 'docs'
    page: parts[1] || null,     // e.g., 'biodcase'
  };
};

export default function App() {
  const alToolRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const [axesOpacity, setAxesOpacity] = useState(0);
  const [route, setRoute] = useState(parseHash());

  // Derived state from route
  const docsOpen = route.section === 'docs';
  const challengesOpen = route.section === 'challenges';
  const challengePage = route.page;

  const handleGetStarted = () => {
    alToolRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleChallengesOpen = (page) => {
    // If called from onClick, page will be an event object - use default
    const pageName = (typeof page === 'string') ? page : 'biodcase';
    window.location.hash = `/challenges/${pageName}`;
  };

  const handleChallengesClose = () => {
    window.location.hash = '';
  };

  const handleDocsOpen = () => {
    window.location.hash = '/docs';
  };

  const handleDocsClose = () => {
    window.location.hash = '';
  };

  // Listen for hash changes (browser back/forward)
  useEffect(() => {
    const handleHashChange = () => {
      setRoute(parseHash());
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      if (scrollContainerRef.current) {
        const scrollTop = scrollContainerRef.current.scrollTop;
        const windowHeight = window.innerHeight;

        // Calculate opacity: 0 at top, 1 when fully scrolled to second section
        // Fade starts at 20% scroll and completes at 80% scroll
        const fadeStart = windowHeight * 0.2;
        const fadeEnd = windowHeight * 0.8;

        let opacity = 0;
        if (scrollTop <= fadeStart) {
          opacity = 0;
        } else if (scrollTop >= fadeEnd) {
          opacity = 1;
        } else {
          opacity = (scrollTop - fadeStart) / (fadeEnd - fadeStart);
        }

        setAxesOpacity(opacity);
      }
    };

    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll);
      // Initial check
      handleScroll();
    }

    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  return (
    <div
      ref={scrollContainerRef}
      className="hide-scrollbar"
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'auto',
        scrollSnapType: 'y mandatory'
      }}
    >
      
      <Axes opacity={axesOpacity} />

      {/* Home Page Section */}
      <div style={{ scrollSnapAlign: 'start' }}>
        <Home onGetStarted={handleGetStarted} onDocsClick={handleDocsOpen} onChallengesClick={handleChallengesOpen} />
      </div>

      {/* AL Tool Section */}
      <div ref={alToolRef} style={{ scrollSnapAlign: 'start' }}>
        <ALTool />
      </div>

      {/* Docs Overlay */}
      <Docs isOpen={docsOpen} onClose={handleDocsClose} />

      {/* Challenges overlay */}
      <Challenges isOpen={challengesOpen} onClose={handleChallengesClose} page={challengePage} />

    </div>
  );
}