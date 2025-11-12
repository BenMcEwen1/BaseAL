import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/**
 * PointCluster component - Visualizes 3D point cloud data with animated transitions
 * @param {Object} props
 * @param {Array<Array<[number, number, number]>>} props.embeddingData - Array of steps, each containing point coordinates
 * @param {number} props.currentStep - Current step index to display
 */
export default function PointCluster({ embeddingData, currentStep }) {
  const pointsRef = useRef();
  const currentStepRef = useRef(currentStep);

  currentStepRef.current = currentStep;

  // Convert embedding data to Float32Arrays and generate colors
  const { positionSteps, colors } = useMemo(() => {
    const steps = embeddingData.map(stepData => {
      const positions = new Float32Array(stepData.length * 3);
      stepData.forEach((point, i) => {
        positions[i * 3] = point[0];
        positions[i * 3 + 1] = point[1];
        positions[i * 3 + 2] = point[2];
      });
      return positions;
    });

    // Generate colors based on which cluster each point belongs to
    const count = embeddingData[0].length;
    const cols = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const cluster = Math.floor(i / 100) / 5;
      cols[i * 3] = 0.5 + Math.sin(cluster * Math.PI * 2) * 0.5;
      cols[i * 3 + 1] = 0.5 + Math.sin(cluster * Math.PI * 2 + 2) * 0.5;
      cols[i * 3 + 2] = 0.5 + Math.sin(cluster * Math.PI * 2 + 4) * 0.5;
    }

    return { positionSteps: steps, colors: cols };
  }, [embeddingData]);

  // Initialize with first step positions
  const currentPositions = useRef(new Float32Array(positionSteps[0]));

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
    if (!pointsRef.current) return;

    const positions = currentPositions.current;
    const target = positionSteps[currentStepRef.current];

    for (let i = 0; i < positions.length; i++) {
      positions[i] += (target[i] - positions[i]) * 0.05;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
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
        size={0.35}
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
