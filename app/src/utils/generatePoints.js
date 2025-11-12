/**
 * Generates a random cloud of points
 * @param {number} count - Number of points to generate
 * @param {number} spread - Spread of the cloud
 * @returns {Array<[number, number, number]>} Array of 3D coordinates
 */
export function generateRandomCloud(count, spread = 15) {
  return Array.from({ length: count }, () => [
    (Math.random() - 0.5) * spread,
    (Math.random() - 0.5) * spread,
    (Math.random() - 0.5) * spread
  ]);
}

/**
 * Generates cluster centers in 3D space
 * @param {number} clustersCount - Number of clusters
 * @param {number} spread - How spread out the cluster centers are
 * @returns {Array<[number, number, number]>} Array of cluster center coordinates
 */
function generateClusterCenters(clustersCount, spread = 2) {
  const centers = [];
  const goldenRatio = (1 + Math.sqrt(5)) / 2;

  for (let i = 0; i < clustersCount; i++) {
    // Use spherical Fibonacci distribution for even spacing
    const theta = 2 * Math.PI * i / goldenRatio;
    const phi = Math.acos(1 - 2 * (i + 0.5) / clustersCount);

    centers.push([
      spread * Math.cos(theta) * Math.sin(phi),
      spread * Math.sin(theta) * Math.sin(phi),
      spread * Math.cos(phi)
    ]);
  }

  return centers;
}

/**
 * Generates points starting to form clusters
 * @param {number} count - Number of points to generate
 * @param {number} clustersCount - Number of clusters
 * @returns {Array<[number, number, number]>} Array of 3D coordinates
 */
export function generateLooseClusters(count, clustersCount = 5) {
  const pointsPerCluster = Math.floor(count / clustersCount);
  const centers = generateClusterCenters(clustersCount, 3);

  return Array.from({ length: count }, (_, i) => {
    const clusterIdx = Math.floor(i / pointsPerCluster) % clustersCount;
    const center = centers[clusterIdx];

    // Loose 3D gaussian distribution around center
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;
    const r = Math.random() * 1.8;

    return [
      center[0] + r * Math.cos(theta) * Math.sin(phi),
      center[1] + r * Math.sin(theta) * Math.sin(phi),
      center[2] + r * Math.cos(phi)
    ];
  });
}

/**
 * Generates tighter clusters
 * @param {number} count - Number of points to generate
 * @param {number} clustersCount - Number of clusters
 * @returns {Array<[number, number, number]>} Array of 3D coordinates
 */
export function generateTightClusters(count, clustersCount = 5) {
  const pointsPerCluster = Math.floor(count / clustersCount);
  const centers = generateClusterCenters(clustersCount, 3.5);

  return Array.from({ length: count }, (_, i) => {
    const clusterIdx = Math.floor(i / pointsPerCluster) % clustersCount;
    const center = centers[clusterIdx];

    // Tighter 3D gaussian distribution around center
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;
    const r = Math.random() * 1.0;

    return [
      center[0] + r * Math.cos(theta) * Math.sin(phi),
      center[1] + r * Math.sin(theta) * Math.sin(phi),
      center[2] + r * Math.cos(phi)
    ];
  });
}

/**
 * Generates well-separated clusters with some dispersion
 * @param {number} count - Number of points to generate
 * @param {number} clustersCount - Number of clusters
 * @returns {Array<[number, number, number]>} Array of 3D coordinates
 */
export function generateSeparatedClusters(count, clustersCount = 5) {
  const pointsPerCluster = Math.floor(count / clustersCount);
  const centers = generateClusterCenters(clustersCount, 4);

  return Array.from({ length: count }, (_, i) => {
    const clusterIdx = Math.floor(i / pointsPerCluster) % clustersCount;
    const center = centers[clusterIdx];

    // Tight 3D gaussian distribution, still with some spread
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.random() * Math.PI;
    const r = Math.random() * 0.6;

    return [
      center[0] + r * Math.cos(theta) * Math.sin(phi),
      center[1] + r * Math.sin(theta) * Math.sin(phi),
      center[2] + r * Math.cos(phi)
    ];
  });
}

/**
 * Generates a sequence of embedding steps showing clustering progression
 * @param {number} pointCount - Number of points per step
 * @param {number} clustersCount - Number of clusters
 * @returns {Array<Array<[number, number, number]>>} Array of steps, each containing point coordinates
 */
export function generateEmbeddingSteps(pointCount = 500, clustersCount = 5) {
  return [
    generateRandomCloud(pointCount),
    generateLooseClusters(pointCount, clustersCount),
    generateTightClusters(pointCount, clustersCount),
    generateSeparatedClusters(pointCount, clustersCount)
  ];
}
