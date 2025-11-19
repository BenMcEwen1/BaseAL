import React, { useRef, useMemo, useEffect } from 'react';
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
 * @param {Array<number>} props.uncertainties - Array of uncertainty values [0, 1] for scaling point sizes
 * @param {Function} props.setID - Callback function to set the selected point ID
 * @param {number|null} props.selectedID - Currently selected point index (null if none selected)
 * @param {Function} props.onPointClick - Optional callback function that receives the clicked point index
 */
export default function PointCluster({ setID, selectedID, embeddingData, currentStep, labels, labelNames, labeledMask, uncertainties, onPointClick }) {
  // console.log('PointCluster props:', { embeddingData, labels, labelNames, labeledMask });
  const pointsRef = useRef();
  const currentStepRef = useRef(currentStep);
  const selectedPointRef = useRef(null);
  const originalSizesRef = useRef(null);
  const ringRef = useRef();

  // Track mouse down position and time to detect drag vs click
  const mouseDownRef = useRef({ x: 0, y: 0, time: 0 });
  const isDraggingRef = useRef(false);

  currentStepRef.current = currentStep;

  // Handle deselection when selectedID becomes null
  useEffect(() => {
    if (selectedID === null && selectedPointRef.current !== null) {
      // Restore the previously selected point's size
      if (pointsRef.current && originalSizesRef.current) {
        const sizeAttribute = pointsRef.current.geometry.attributes.size;
        sizeAttribute.array[selectedPointRef.current] = originalSizesRef.current[selectedPointRef.current];
        sizeAttribute.needsUpdate = true;
      }

      // Hide the ring
      if (ringRef.current) {
        ringRef.current.visible = false;
      }

      selectedPointRef.current = null;
    }
  }, [selectedID]);

  // Create a unique key based on labels, labeledMask, and uncertainties to force recreation when they change
  const colorKey = useMemo(() => {
    const labelsStr = labels ? labels.join(',') : 'no-labels';
    const maskStr = labeledMask ? labeledMask.map(m => m ? '1' : '0').join('') : 'no-mask';
    const uncertStr = uncertainties ? uncertainties.map(u => u.toFixed(2)).join(',') : 'no-uncertainties';
    return `${labelsStr}-${maskStr}-${uncertStr}`;
  }, [labels, labeledMask, uncertainties]);

  // Convert embedding data to Float32Arrays and generate colors, sizes, and alphas
  const { positionSteps, colors, sizes, alphas } = useMemo(() => {
    // Handle empty data case
    if (!embeddingData || embeddingData.length === 0) {
      return { positionSteps: [], colors: new Float32Array(0), sizes: new Float32Array(0), alphas: new Float32Array(0) };
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
        const c = 0.2; // Saturation - increased for more vibrant colors
        const x = c * (1 - Math.abs((h % 2) - 1));
        const m = 0.1

        let r, g, b;
        if (h < 1) { r = c; g = x; b = 0; }
        else if (h < 2) { r = x; g = c; b = 0; }
        else if (h < 3) { r = 0; g = c; b = x; }
        else if (h < 4) { r = 0; g = x; b = c; }
        else if (h < 5) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        // Check if this point is labeled - make labeled points grey
        const isLabeled = labeledMask ? labeledMask[i] : true;

        if (isLabeled) {
          // Labeled points: grey color
          cols[i * 3] = 0.4;
          cols[i * 3 + 1] = 0.4;
          cols[i * 3 + 2] = 0.4;
        } else {
          // Unlabeled points: vibrant saturated colors (no lightness adjustment)
          cols[i * 3] = r + m;
          cols[i * 3 + 1] = g + m;
          cols[i * 3 + 2] = b + m;
        }

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

    // Generate sizes based on uncertainties
    const pointSizes = new Float32Array(count);
    const minSize = 0.3;  // Minimum point size (for labeled/certain points)
    const maxSize = 0.4;  // Maximum point size (for uncertain points)

    console.log(uncertainties)
    if (uncertainties && uncertainties.length === count) {
      // Find actual min/max uncertainties in the data for relative scaling
      const minUncertainty = Math.min(...uncertainties);
      const maxUncertainty = Math.max(...uncertainties);
      const uncertaintyRange = maxUncertainty - minUncertainty;

      // Scale points relative to actual uncertainty distribution
      for (let i = 0; i < count; i++) {
        const uncertainty = uncertainties[i];

        if (uncertaintyRange > 0.01) {  // If there's variation in uncertainties
          // Normalize to [0, 1] based on actual data range
          const normalizedUncertainty = (uncertainty - minUncertainty) / uncertaintyRange;
          pointSizes[i] = minSize + normalizedUncertainty * (maxSize - minSize);
        } else {
          // All uncertainties are similar, use average size
          pointSizes[i] = (minSize + maxSize) / 2;
        }
      }
    } else {
      console.log("Fallback point size")
      // Fallback: uniform size
      for (let i = 0; i < count; i++) {
        pointSizes[i] = (minSize + maxSize) / 2; // Default medium size
      }
    }

    // Generate alpha (transparency) values based on labeled status
    const pointAlphas = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const isLabeled = labeledMask ? labeledMask[i] : true;
      pointAlphas[i] = isLabeled ? 0.2 : 1.0;  // Labeled: semi-transparent, Unlabeled: opaque
    }

    // Store original sizes for restoration when selection changes
    originalSizesRef.current = new Float32Array(pointSizes);

    return { positionSteps: steps, colors: cols, sizes: pointSizes, alphas: pointAlphas };
  }, [embeddingData, labels, labeledMask, uncertainties]);

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

  // Shader modification to use per-vertex sizes
  const onBeforeCompile = useMemo(() => (shader) => {
    shader.vertexShader = shader.vertexShader.replace(
      'uniform float size;',
      'attribute float size;'
    );
  }, []);

  // Animate transition between steps
  useFrame(({ camera }) => {
    if (!pointsRef.current || positionSteps.length === 0) return;

    const positions = currentPositions.current;
    const target = positionSteps[currentStepRef.current];

    for (let i = 0; i < positions.length; i++) {
      positions[i] += (target[i] - positions[i]) * 0.05;
    }

    pointsRef.current.geometry.attributes.position.needsUpdate = true;

    // Update ring position to follow the selected point and face the camera
    if (ringRef.current && ringRef.current.visible && selectedPointRef.current !== null) {
      const idx = selectedPointRef.current;
      ringRef.current.position.set(
        positions[idx * 3],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
      );

      // Make the ring always face the camera (billboard effect)
      ringRef.current.quaternion.copy(camera.quaternion);
    }
  });

  // Return null if there's no data to display
  if (positionSteps.length === 0) {
    return null;
  }

  // Handle mouse down - record position and time
  const handlePointerDown = (event) => {
    mouseDownRef.current = {
      x: event.clientX || event.point.x,
      y: event.clientY || event.point.y,
      time: Date.now()
    };
    isDraggingRef.current = false;
  };

  // Handle mouse move - detect dragging
  const handlePointerMove = (event) => {
    if (mouseDownRef.current.time > 0) {
      const dx = (event.clientX || event.point.x) - mouseDownRef.current.x;
      const dy = (event.clientY || event.point.y) - mouseDownRef.current.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      // If moved more than 5 pixels, consider it a drag
      if (distance > 5) {
        isDraggingRef.current = true;
      }
    }
  };

  // Handle point click - check if actually clicking on a point
  const handlePointClick = (event) => {
    console.log(event)
    event.stopPropagation();

    const clickTime = Date.now() - mouseDownRef.current.time;
    const isDrag = isDraggingRef.current;

    // Reset tracking
    mouseDownRef.current = { x: 0, y: 0, time: 0 };
    isDraggingRef.current = false;

    // Only trigger viewer if it's a quick click (< 200ms) and not a drag
    if (isDrag || clickTime > 500) {
      console.log('Ignoring click - was a drag or hold');
      return;
    }

    const clickedIndex = event.index;
    setID(clickedIndex);

    // Update point sizes to highlight the selected point
    if (pointsRef.current && originalSizesRef.current) {
      const sizeAttribute = pointsRef.current.geometry.attributes.size;

      // Restore previous selected point's size
      if (selectedPointRef.current !== null && selectedPointRef.current !== clickedIndex) {
        sizeAttribute.array[selectedPointRef.current] = originalSizesRef.current[selectedPointRef.current];
      }

      // Increase the size of the newly selected point
      const enlargementFactor = 2.0; // Make selected point 3x larger
      sizeAttribute.array[clickedIndex] = originalSizesRef.current[clickedIndex] * enlargementFactor;

      // Mark the attribute as needing update
      sizeAttribute.needsUpdate = true;

      // Update the selected point reference
      selectedPointRef.current = clickedIndex;

      // Position and show the white ring around the selected point
      if (ringRef.current && positionSteps.length > 0 && pointsRef.current) {
        const positions = currentPositions.current;
        ringRef.current.position.set(
          positions[clickedIndex * 3],
          positions[clickedIndex * 3 + 1],
          positions[clickedIndex * 3 + 2]
        );
        ringRef.current.visible = true;
      }
    }

    return;

    // if (event.distanceToRay > 0.05) {
    //   setID(null)
    //   return;
    // } else {
    //   console.log('Selected point: ', event.index)
    //   setID(event.index)
    //   return
    // }
  };

  return (
    <>
      <points
        ref={pointsRef}
        key={colorKey}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onClick={handlePointClick}
      >
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
          <bufferAttribute
            attach="attributes-size"
            count={sizes.length}
            array={sizes}
            itemSize={1}
          />
          <bufferAttribute
            attach="attributes-alpha"
            count={alphas.length}
            array={alphas}
            itemSize={1}
          />
        </bufferGeometry>
        <pointsMaterial
          size={1.0}
          vertexColors
          sizeAttenuation
          map={circleTexture}
          transparent
          // alphaTest={0.01}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          onBeforeCompile={onBeforeCompile}
        />
      </points>

      {/* White ring around selected point */}
      <mesh ref={ringRef} visible={false}>
        <ringGeometry args={[0.05, 0.06, 64]} />
        <meshBasicMaterial color="white" transparent opacity={0.9} side={THREE.DoubleSide} />
      </mesh>
    </>
  );
}
