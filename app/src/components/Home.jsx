import { useState } from "react";
import Promote from "./Promote";

export default function Home({ onGetStarted, onDocsClick, onChallengesClick, isMobile }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      background: '#060014ff',
      display: 'flex',
      flexDirection: 'column',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* <Promote/> */}
      {/* Top Navigation Bar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: isMobile ? '15px 20px' : '20px 40px',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 10
      }}>
        {/* Logo Space - Left */}
        <div style={{
          width: '40px',
          height: '40px',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#666',
          fontSize: '10px'
        }}>
            <img
              src='/baseAL_logo.png'
              alt='BaseAL Logo'
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain'
              }}
            />
        </div>

        {/* Links - Right (desktop) / Burger (mobile) */}
        {isMobile ? (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '5px'
              }}
            >
              <span style={{ display: 'block', width: '24px', height: '2px', background: '#fff', transition: 'all 0.3s', transform: menuOpen ? 'rotate(45deg) translate(5px, 5px)' : 'none' }} />
              <span style={{ display: 'block', width: '24px', height: '2px', background: '#fff', transition: 'all 0.3s', opacity: menuOpen ? 0 : 1 }} />
              <span style={{ display: 'block', width: '24px', height: '2px', background: '#fff', transition: 'all 0.3s', transform: menuOpen ? 'rotate(-45deg) translate(5px, -5px)' : 'none' }} />
            </button>

            {/* Dropdown menu */}
            {menuOpen && (
              <div style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                marginTop: '10px',
                background: 'rgba(20, 10, 40, 0.95)',
                backdropFilter: 'blur(10px)',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                padding: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                minWidth: '180px'
              }}>
                <button
                  onClick={() => { onChallengesClick(); setMenuOpen(false); }}
                  style={{
                    color: '#fff',
                    padding: '12px 16px',
                    background: 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    textAlign: 'left'
                  }}
                >
                  Data Challenges
                </button>
                <button
                  onClick={() => { onDocsClick(); setMenuOpen(false); }}
                  style={{
                    color: '#fff',
                    padding: '12px 16px',
                    background: 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    textAlign: 'left'
                  }}
                >
                  Docs
                </button>
                <a
                  href="https://github.com/benmcewen1/BaseAL"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    color: '#fff',
                    textDecoration: 'none',
                    padding: '12px 16px',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <img
                    src="/GitHub_Logos/GitHub Logos/SVG/GitHub_Invertocat_Light.svg"
                    alt="GitHub"
                    style={{ width: '18px', height: '18px' }}
                  />
                  GitHub
                </a>
              </div>
            )}
          </div>
        ) : (
          <div style={{
            display: 'flex',
            gap: '20px'
          }}>
            <button
              onClick={onChallengesClick}
              style={{
                color: '#fff',
                textDecoration: 'none',
                padding: '10px 20px',
                background: 'rgba(255, 255, 255, 0.2)',
                border: 'none',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '500',
                transition: 'background 0.2s',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.4)'}
              onMouseLeave={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.2)'}
            >
              Data Challenges
            </button>
            <button
              onClick={onDocsClick}
              style={{
                color: '#fff',
                textDecoration: 'none',
                padding: '10px 20px',
                background: 'rgba(255, 255, 255, 0.2)',
                border: 'none',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '500',
                transition: 'background 0.2s',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.4)'}
              onMouseLeave={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.2)'}
            >
              Docs
            </button>
            <a
              href="https://github.com/benmcewen1/BaseAL"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                color: '#fff',
                textDecoration: 'none',
                padding: '10px 20px',
                background: 'rgba(255, 255, 255, 0.2)',
                borderRadius: '6px',
                fontSize: '12px',
                fontWeight: '500',
                transition: 'background 0.2s',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
              onMouseEnter={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.4)'}
              onMouseLeave={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.2)'}
            >
              <img
                src="/GitHub_Logos/GitHub Logos/SVG/GitHub_Invertocat_Light.svg"
                alt="GitHub"
                style={{
                  width: '20px',
                  height: '20px'
                }}
              />
              GitHub
            </a>
          </div>
        )}
      </div>

      {/* Center Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '25px'
      }}>
        {/* Title Section */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: isMobile ? '15px' : '25px'
        }}>
          {/* Logo Space - Next to Title */}
          <div style={{
            width: isMobile ? 'clamp(40px, 12vw, 80px)' : '80px',
            height: isMobile ? 'clamp(40px, 12vw, 80px)' : '80px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
            fontSize: '14px',
            overflow: 'hidden',
            flexShrink: 0
          }}>
            <img
              src='baseAL_logo.png'
              alt='BaseAL Logo'
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain'
              }}
            />
          </div>

          {/* Title */}
          <h1 style={{
            fontSize: isMobile ? 'clamp(24px, 8vw, 48px)' : '48px',
            fontWeight: '200',
            color: '#fff',
            margin: 0,
            letterSpacing: isMobile ? '5px' : '10px'
          }}>
            Base<b style={{fontWeight: '500'}}>AL</b><p style={{fontSize: isMobile ? '10px' : '12px', letterSpacing: '5px', paddingLeft: '5px'}}>v1.1.0</p>
          </h1>
        </div>

        {/* Subtitle */}
        <p style={{
          fontSize: isMobile ? '12px' : '14px',
          color: 'rgba(255, 255, 255, 0.8)',
          margin: 0,
          textAlign: 'center',
          maxWidth: '600px',
          padding: isMobile ? '0 20px' : 0
        }}>
          A Modular Active Learning Evaluation Framework
        </p>

        {/* Get Started Button */}
        <button
          onClick={onGetStarted}
          style={{
            padding: '12px 36px',
            fontSize: '14px',
            fontWeight: '600',
            background: '#4ae290',
            color: '#0a0a0a',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'transform 0.2s, background 0.2s',
            // marginTop: '20px'
          }}
          onMouseEnter={(e) => {
            e.target.style.transform = 'translateY(-2px)';
            e.target.style.background = '#5ef0a0';
          }}
          onMouseLeave={(e) => {
            e.target.style.transform = 'translateY(0)';
            e.target.style.background = '#4ae290';
          }}
        >
          Get Started
        </button>
      </div>

      {/* Scroll Indicator */}
      <div style={{
        position: 'absolute',
        bottom: '40px',
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        color: '#666',
        fontSize: '12px',
        animation: isMobile ? 'none' : 'bounce 2s infinite'
      }}>
        <span>{isMobile ? 'Demo on desktop' : 'Scroll to explore'}</span>
        {!isMobile && (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9"></polyline>
          </svg>
        )}
      </div>

      {/* Add animation */}
      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateX(-50%) translateY(0); }
          50% { transform: translateX(-50%) translateY(10px); }
        }

      `}</style>
    </div>
  );
}
