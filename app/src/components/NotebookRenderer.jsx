import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function NotebookRenderer({ notebookPath }) {
  const [notebook, setNotebook] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!notebookPath) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    // Fetch the notebook file
    fetch(notebookPath)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load notebook: ${response.statusText}`);
        }
        return response.json();
      })
      .then(data => {
        setNotebook(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [notebookPath]);

  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: '#888'
      }}>
        Loading notebook...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: '#ff6b6b'
      }}>
        Error: {error}
      </div>
    );
  }

  if (!notebook) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: '#888'
      }}>
        Select a notebook from the menu
      </div>
    );
  }

  // Render a single cell
  const renderCell = (cell, index) => {
    const cellId = cell.id || `cell-${index}`;

    // Render markdown cells
    if (cell.cell_type === 'markdown') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;

      return (
        <div
          key={cellId}
          style={{
            marginBottom: '24px',
            padding: '16px',
            background: '#1a1a1a',
            borderRadius: '8px',
            borderLeft: '4px solid #4ae290'
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Style markdown elements
              h1: ({node, ...props}) => (
                <h1 style={{
                  color: '#fff',
                  fontSize: '26px',
                  marginTop: '0',
                  marginBottom: '16px',
                  fontWeight: '600'
                }} {...props} />
              ),
              h2: ({node, ...props}) => (
                <h2 style={{
                  color: '#fff',
                  fontSize: '19px',
                  marginTop: '24px',
                  marginBottom: '12px',
                  fontWeight: '600'
                }} {...props} />
              ),
              h3: ({node, ...props}) => (
                <h3 style={{
                  color: '#fff',
                  fontSize: '16px',
                  marginTop: '20px',
                  marginBottom: '10px',
                  fontWeight: '600'
                }} {...props} />
              ),
              p: ({node, ...props}) => (
                <p style={{
                  color: '#ccc',
                  lineHeight: '1.6',
                  marginBottom: '12px'
                }} {...props} />
              ),
              code: ({node, inline, ...props}) => (
                inline ?
                  <code style={{
                    background: '#2a2a2a',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    color: '#4ae290',
                    fontSize: '11px'
                  }} {...props} /> :
                  <code style={{
                    display: 'block',
                    background: '#0a0a0a',
                    padding: '12px',
                    borderRadius: '6px',
                    color: '#4ae290',
                    fontSize: '11px',
                    overflowX: 'auto',
                    whiteSpace: 'pre'
                  }} {...props} />
              ),
              pre: ({node, ...props}) => (
                <pre style={{
                  background: '#0a0a0a',
                  padding: '12px',
                  borderRadius: '6px',
                  overflowX: 'auto',
                  marginBottom: '12px'
                }} {...props} />
              ),
              ul: ({node, ...props}) => (
                <ul style={{
                  color: '#ccc',
                  paddingLeft: '20px',
                  marginBottom: '12px'
                }} {...props} />
              ),
              ol: ({node, ...props}) => (
                <ol style={{
                  color: '#ccc',
                  paddingLeft: '20px',
                  marginBottom: '12px'
                }} {...props} />
              ),
              li: ({node, ...props}) => (
                <li style={{
                  marginBottom: '6px',
                  lineHeight: '1.6'
                }} {...props} />
              ),
              blockquote: ({node, ...props}) => (
                <blockquote style={{
                  borderLeft: '4px solid #555',
                  paddingLeft: '16px',
                  margin: '12px 0',
                  color: '#999',
                  fontStyle: 'italic'
                }} {...props} />
              ),
              a: ({node, ...props}) => (
                <a style={{
                  color: '#4ae290',
                  textDecoration: 'none'
                }} {...props} />
              ),
            }}
          >
            {source}
          </ReactMarkdown>
        </div>
      );
    }

    // Render code cells
    if (cell.cell_type === 'code') {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      const outputs = cell.outputs || [];

      return (
        <div key={cellId} style={{ marginBottom: '24px' }}>
          {/* Code Input */}
          {source && (
            <div style={{
              background: '#0a0a0a',
              borderRadius: '8px 8px 0 0',
              padding: '16px',
              borderLeft: '4px solid #3a86ff'
            }}>
              <div style={{
                fontSize: '9px',
                color: '#666',
                marginBottom: '8px',
                textTransform: 'uppercase',
                letterSpacing: '1px'
              }}>
                Input
              </div>
              <pre style={{
                margin: 0,
                color: '#4ae290',
                fontSize: '11px',
                whiteSpace: 'pre-wrap',
                wordWrap: 'break-word',
                fontFamily: 'monospace'
              }}>
                {source}
              </pre>
            </div>
          )}

          {/* Code Outputs */}
          {outputs.length > 0 && (
            <div style={{
              background: '#1a1a1a',
              borderRadius: source ? '0 0 8px 8px' : '8px',
              padding: '16px',
              borderLeft: '4px solid #ff6b6b',
              borderTop: source ? '1px solid #333' : 'none'
            }}>
              <div style={{
                fontSize: '9px',
                color: '#666',
                marginBottom: '8px',
                textTransform: 'uppercase',
                letterSpacing: '1px'
              }}>
                Output
              </div>
              {outputs.map((output, outputIndex) => renderOutput(output, outputIndex))}
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  // Render cell outputs (text, images, etc.)
  const renderOutput = (output, index) => {
    const key = `output-${index}`;

    // Stream output (stdout/stderr)
    if (output.output_type === 'stream') {
      const text = Array.isArray(output.text) ? output.text.join('') : output.text;
      return (
        <pre key={key} style={{
          margin: 0,
          color: output.name === 'stderr' ? '#ff6b6b' : '#ccc',
          fontSize: '11px',
          whiteSpace: 'pre-wrap',
          wordWrap: 'break-word',
          fontFamily: 'monospace'
        }}>
          {text}
        </pre>
      );
    }

    // Execute result (return values)
    if (output.output_type === 'execute_result' || output.output_type === 'display_data') {
      const data = output.data;

      // Image output
      if (data && data['image/png']) {
        return (
          <img
            key={key}
            src={`data:image/png;base64,${data['image/png']}`}
            alt="Output"
            style={{
              maxWidth: '100%',
              height: 'auto',
              borderRadius: '4px',
              marginTop: '8px'
            }}
          />
        );
      }

      // SVG output
      if (data && data['image/svg+xml']) {
        const svgContent = Array.isArray(data['image/svg+xml'])
          ? data['image/svg+xml'].join('')
          : data['image/svg+xml'];
        return (
          <div
            key={key}
            dangerouslySetInnerHTML={{ __html: svgContent }}
            style={{ marginTop: '8px' }}
          />
        );
      }

      // Text/plain output
      if (data && data['text/plain']) {
        const text = Array.isArray(data['text/plain'])
          ? data['text/plain'].join('')
          : data['text/plain'];
        return (
          <pre key={key} style={{
            margin: 0,
            color: '#ccc',
            fontSize: '11px',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
            fontFamily: 'monospace'
          }}>
            {text}
          </pre>
        );
      }

      // HTML output
      if (data && data['text/html']) {
        const html = Array.isArray(data['text/html'])
          ? data['text/html'].join('')
          : data['text/html'];
        return (
          <div
            key={key}
            dangerouslySetInnerHTML={{ __html: html }}
            style={{ marginTop: '8px', color: '#ccc' }}
          />
        );
      }
    }

    // Error output
    if (output.output_type === 'error') {
      return (
        <pre key={key} style={{
          margin: 0,
          color: '#ff6b6b',
          fontSize: '11px',
          whiteSpace: 'pre-wrap',
          wordWrap: 'break-word',
          fontFamily: 'monospace'
        }}>
          {output.ename}: {output.evalue}
          {'\n'}
          {Array.isArray(output.traceback) ? output.traceback.join('\n') : output.traceback}
        </pre>
      );
    }

    return null;
  };

  return (
    <div style={{
      padding: '40px',
      maxWidth: '900px',
      margin: '0 auto'
    }}>
      {notebook.cells && notebook.cells.map((cell, index) => renderCell(cell, index))}
    </div>
  );
}
