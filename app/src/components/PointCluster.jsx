import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * PointCluster component - Visualizes 3D point cloud data with animated transitions
 * @param {Object} props
 * @param {Array<Array<[number, number, number]>>} props.embeddingData - Array of steps, each containing point coordinates
 * @param {number} props.currentStep - Current step index to display
 * @param {Array<number>} props.labels - Array of label indices for each point
 * @param {Array<string>} props.labelNames - Array of label names for each point
 * @param {Array<boolean>} props.labeledMask - Array indicating which points are labeled
 */
export default function PointCluster({ embeddingData, currentStep, labels, labelNames, labeledMask }) {
  // console.log('PointCluster props:', { embeddingData, labels, labelNames, labeledMask });
  const pointsRef = useRef();
  const currentStepRef = useRef(currentStep);

  currentStepRef.current = currentStep;

  // Create a unique key based on labels and labeledMask to force recreation when colors change
  const colorKey = useMemo(() => {
    const labelsStr = labels ? labels.join(',') : 'no-labels';
    const maskStr = labeledMask ? labeledMask.map(m => m ? '1' : '0').join('') : 'no-mask';
    return `${labelsStr}-${maskStr}`;
  }, [labels, labeledMask]);

  // Convert embedding data to Float32Arrays and generate colors
  const { positionSteps, colors } = useMemo(() => {
    // Handle empty data case
    if (!embeddingData || embeddingData.length === 0) {
      return { positionSteps: [], colors: new Float32Array(0) };
    }

    const steps = embeddingData.map(stepData => {
      const positions = new Float32Array(stepData.length * 3);
      stepData.forEach((point, i) => {
        positions[i * 3] = point[0];
        positions[i * 3 + 1] = point[1];
        positions[i * 3 + 2] = point[2];
      });
      return positions;
    });

    // Generate colors based on labels or fallback to cluster-based coloring
    const count = embeddingData[0].length;
    const cols = new Float32Array(count * 3);

    if (labels && labels.length === count) {
      // console.log("Setting colours")
      // Find unique labels to determine number of classes
      const uniqueLabels = [...new Set(labels)];
      const numClasses = uniqueLabels.length;
      // console.log('Unique labels:', uniqueLabels, 'Num classes:', numClasses);
      // console.log('First 10 label indices:', labels.slice(0, 10));

      // Generate distinct colors for each label
      for (let i = 0; i < count; i++) {
        const labelIndex = labels[i];
        const hue = (labelIndex / numClasses) * 360;

        // if (i < 5) {
        //   console.log(`Point ${i}: labelIndex=${labelIndex}, hue=${hue}`);
        // }

        // Convert HSL to RGB for vibrant, distinct colors
        const h = hue / 60;
        const c = 0.8; // Saturation
        const x = c * (1 - Math.abs((h % 2) - 1));
        const m = 0.3; // Lightness adjustment

        let r, g, b;
        if (h < 1) { r = c; g = x; b = 0; }
        else if (h < 2) { r = x; g = c; b = 0; }
        else if (h < 3) { r = 0; g = c; b = x; }
        else if (h < 4) { r = 0; g = x; b = c; }
        else if (h < 5) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        // Check if this point is labeled - make unlabeled points dimmer
        const isLabeled = labeledMask ? labeledMask[i] : true;
        const intensity = isLabeled ? 1.0 : 0.3;

        cols[i * 3] = (r + m) * intensity;
        cols[i * 3 + 1] = (g + m) * intensity;
        cols[i * 3 + 2] = (b + m) * intensity;

        // if (i < 5) {
        //   console.log(`  RGB: [${cols[i * 3]}, ${cols[i * 3 + 1]}, ${cols[i * 3 + 2]}], isLabeled=${isLabeled}`);
        // }
      }
      // console.log('Color array sample (first 15 values):', cols.slice(0, 15));
    } else {
      // console.log("Using fallback colours")
      // Fallback: Generate colors based on which cluster each point belongs to
      for (let i = 0; i < count; i++) {
        const cluster = Math.floor(i / 100) / 5;
        cols[i * 3] = 0.5 + Math.sin(cluster * Math.PI * 2) * 0.5;
        cols[i * 3 + 1] = 0.5 + Math.sin(cluster * Math.PI * 2 + 2) * 0.5;
        cols[i * 3 + 2] = 0.5 + Math.sin(cluster * Math.PI * 2 + 4) * 0.5;
      }
    }

    return { positionSteps: steps, colors: cols };
  }, [embeddingData, labels, labeledMask]);

  // Initialize with first step positions (or empty if no data)
  const currentPositions = useRef(
    positionSteps.length > 0 ? new Float32Array(positionSteps[0]) : new Float32Array(0)
  );

  // Update currentPositions when we get new data
  if (positionSteps.length > 0 && currentPositions.current.length === 0) {
    currentPositions.current = new Float32Array(positionSteps[0]);
  }

  // Create circle texture for points with hard edge
  const circleTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Create a hard-edged circle
    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 28);
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.85, 'rgba(255,255,255,1)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    return texture;
  }, []);

  // Animate transition between steps
  useFrame(() => {
    if (!pointsRef.current || positionSteps.length === 0) return;

    const positions = currentPositions.current;
    const target = positionSteps[currentStepRef.current];

    for (let i = 0; i < positions.length; i++) {
      positions[i] += (target[i] - positions[i]) * 0.05;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  // Return null if there's no data to display
  if (positionSteps.length === 0) {
    return null;
  }

  return (
    <points ref={pointsRef} key={colorKey}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={currentPositions.current.length / 3}
          array={currentPositions.current}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.1}
        vertexColors
        sizeAttenuation
        map={circleTexture}
        transparent
        alphaTest={0.01}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
