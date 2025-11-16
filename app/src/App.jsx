import React, { useRef, useState, useEffect } from 'react';
import Home from './components/Home';
import ALTool from './components/ALTool';
import Axes from './components/Axes'
import AnalyticsV2 from './components/AnalyticsV2';

export default function App() {
  const alToolRef = useRef(null);
  const scrollContainerRef = useRef(null);
  const [axesOpacity, setAxesOpacity] = useState(0);

  const handleGetStarted = () => {
    alToolRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

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
        <Home onGetStarted={handleGetStarted} />
        
      </div>

      {/* AL Tool Section */}
      <div ref={alToolRef} style={{ scrollSnapAlign: 'start' }}>
        <ALTool />
      </div>
    </div>
  );
}