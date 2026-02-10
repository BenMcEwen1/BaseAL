import { useState } from 'react';
import NotebookRenderer from './NotebookRenderer';

// Import notebooks directly as URLs
import notebook00 from '../../../core/docs/00_introduction.ipynb?url';
import notebook01 from '../../../core/docs/01_understanding_data.ipynb?url';
import notebook02 from '../../../core/docs/02_neural_network_model.ipynb?url';
import notebook03 from '../../../core/docs/03_active_learning_loop.ipynb?url';
import notebook04 from '../../../core/docs/04_visualization_pca.ipynb?url';
import notebook05 from '../../../core/docs/05_complete_pipeline.ipynb?url';
import notebook06 from '../../../core/docs/06_experiment_manager.ipynb?url';

const NOTEBOOKS = [
  {
    id: '00',
    title: 'Welcome',
    path: notebook00
  },
  {
    id: '01',
    title: '1. Understanding the Data',
    path: notebook01
  },
  {
    id: '02',
    title: '2. Neural Network Model',
    path: notebook02
  },
  {
    id: '03',
    title: '3. Active Learning Loop',
    path: notebook03
  },
  {
    id: '04',
    title: '4. Visualization (PCA)',
    path: notebook04
  },
  {
    id: '05',
    title: '5. Complete Pipeline',
    path: notebook05
  },
  {
    id: '06',
    title: '6. Experiment Manager',
    path: notebook06
  }
];

export default function Docs({ isOpen, onClose, isMobile }) {
  const [selectedNotebook, setSelectedNotebook] = useState(NOTEBOOKS[0].path);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
        padding: isMobile ? '15px 20px' : '20px 40px',
        borderBottom: '1px solid #2a2a2a'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {isMobile && (
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
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
              <span style={{ display: 'block', width: '20px', height: '2px', background: '#fff', transition: 'all 0.3s', transform: sidebarOpen ? 'rotate(45deg) translate(5px, 5px)' : 'none' }} />
              <span style={{ display: 'block', width: '20px', height: '2px', background: '#fff', transition: 'all 0.3s', opacity: sidebarOpen ? 0 : 1 }} />
              <span style={{ display: 'block', width: '20px', height: '2px', background: '#fff', transition: 'all 0.3s', transform: sidebarOpen ? 'rotate(-45deg) translate(5px, -5px)' : 'none' }} />
            </button>
          )}
          <h2 style={{
            color: '#fff',
            margin: 0,
            fontSize: isMobile ? '15px' : '19px',
            fontWeight: '300',
            letterSpacing: '2px'
          }}>
            Base<b style={{ fontWeight: '500' }}>AL</b> Documentation
          </h2>
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            borderRadius: '8px',
            width: '40px',
            height: '40px',
            cursor: 'pointer',
            display: 'flex',
            margin: '5px',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff',
            fontSize: '19px',
            transition: 'all 0.2s',
            border: 'none'
          }}
          onMouseEnter={(e) => {
            e.target.style.color = '#4ae290';
          }}
          onMouseLeave={(e) => {
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
        overflow: 'hidden',
        position: 'relative'
      }}>
        {/* Left Sidebar: Navigation */}
        {(!isMobile || sidebarOpen) && (
          <div style={{
            width: isMobile ? '100%' : '33%',
            background: '#060014ff',
            borderRight: isMobile ? 'none' : '1px solid #2a2a2a',
            overflowY: 'auto',
            padding: '20px',
            ...(isMobile ? {
              position: 'absolute',
              top: 0,
              left: 0,
              bottom: 0,
              zIndex: 10
            } : {})
          }}>
            <h3 style={{
              color: '#999',
              fontSize: '10px',
              textTransform: 'uppercase',
              letterSpacing: '2px',
              marginTop: 0,
              padding: '20px',
              marginBottom: '20px',
              fontWeight: '600'
            }}>
              Notebooks
            </h3>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '8px'
            }}>
              {NOTEBOOKS.map(notebook => (
                <button
                  key={notebook.id}
                  onClick={() => {
                    setSelectedNotebook(notebook.path);
                    if (isMobile) setSidebarOpen(false);
                  }}
                  style={{
                    padding: '16px 20px',
                    background: selectedNotebook === notebook.path ? '#1a1a1a' : 'transparent',
                    border: selectedNotebook === notebook.path ? '2px solid #4ae290' : '2px solid transparent',
                    borderRadius: '8px',
                    color: selectedNotebook === notebook.path ? '#4ae290' : '#ccc',
                    fontSize: '13px',
                    fontWeight: selectedNotebook === notebook.path ? '600' : '400',
                    textAlign: 'left',
                    cursor: 'pointer',
                    transition: 'all 0.2s'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedNotebook !== notebook.path) {
                      e.target.style.background = '#0f0f0f';
                      e.target.style.color = '#fff';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedNotebook !== notebook.path) {
                      e.target.style.background = 'transparent';
                      e.target.style.color = '#ccc';
                    }
                  }}
                >
                  {notebook.title}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Right Content Area: Notebook */}
        <div style={{
          width: isMobile ? '100%' : '67%',
          background: '#060014',
          overflowY: 'auto'
        }}>
          <NotebookRenderer notebookPath={selectedNotebook} />
        </div>
      </div>
    </div>
  );
}