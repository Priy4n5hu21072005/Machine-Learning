# Problem: Agglomerative Hierarchical Clustering (Bottom-Up)
# Description: This script demonstrates how to perform Agglomerative Clustering using 
# SciPy (for dendrogram visualization) and Scikit-Learn (for clustering execution).

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Step 1: Input Data - defined as single feature points for simplicity of visualization
X = np.array([[1], [3], [8], [9]])

# Step 2: Create a Linkage Matrix
# Using 'single' method which considers the shortest distance between clusters
linkage_matrix = linkage(X, method='single')

# Step 3: Performance Check and Dendrogram Visualization
# Dendrograms are crucial for visual representation of hierarchy
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram (Agglomerative)")
plt.xlabel("Data Point Index")
plt.ylabel("Distance (Closeness)")
plt.grid(True, axis='y', linestyle=':', alpha=0.5)
plt.show()

# Step 4: Model Initialization with Scikit-Learn
# setting distance_threshold to 0 ensures we explore the entire hierarchy
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
cluster_labels = model.fit_predict(X)

# Results Output: Iterating through points to display their assigned cluster label
print("Clustering Results:")
for point, label in zip(X, cluster_labels):
    print(f"Point {point[0]} is assigned to Cluster: {label}")