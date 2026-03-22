# Problem: K-Means Clustering Algorithm from Scratch
# Description: This script implements the core logic of K-Means clustering using NumPy, 
# demonstrating centroid initialization, cluster assignment, and centroid updating.

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize synthetic 2D data points for clustering
X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11], [8,2], [10,2], [9,3]])
k = 2  # Number of clusters

# Step 2: Randomly initialize centroids from the existing data points
np.random.seed(42)
initial_centroids = X[np.random.choice(len(X), k, replace=False)]

def euclidean_dist(x, y):
    """Computes the straight-line distance between two points."""
    return np.sqrt(np.sum((x - y) ** 2))

def assign_clusters(X, centroids):
    """Assigns each data point to the nearest centroid."""
    clusters = []
    for point in X:
        distances = [euclidean_dist(point, center) for center in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)

def update_centroids(X, clusters, k):
    """Recalculates centroids as the mean of points in each cluster."""
    new_centroids = []
    for i in range(k):
        # Calculate mean of all points assigned to cluster 'i'
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            # If a cluster is empty, keep the centroid where it was (or handle appropriately)
            new_centroids.append(X[np.random.choice(len(X))])
    return np.array(new_centroids)

# Step 3: Run the iterative process (Optimization Loop)
current_centroids = initial_centroids
for iteration in range(10):
    clusters = assign_clusters(X, current_centroids)
    new_centroids = update_centroids(X, clusters, k)
    
    # Convergence Check: if centroids don't change, we've found the solution
    if np.all(current_centroids == new_centroids):
        print(f"Converged after {iteration} iterations.")
        break
    current_centroids = new_centroids

# Step 4: Visualization of final clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, label='Data Clusters')
plt.scatter(current_centroids[:, 0], current_centroids[:, 1], c='red', marker='X', s=200, label='Final Centroids')

# Professional Plot Styling
plt.title("K-Means Clustering Implementation (from Scratch)")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Display final centroid coordinates
print("Final Centroids:\n", current_centroids)