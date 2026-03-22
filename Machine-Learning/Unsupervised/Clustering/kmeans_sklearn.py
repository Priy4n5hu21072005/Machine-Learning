# Problem: K-Means Clustering using Scikit-Learn
# Description: This script demonstrates the use of the powerful KMeans implementation 
# from the Scikit-Learn library for clustering datasets effortlessly with built-in optimizations.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Pre-defined 2D data points for clustering demonstration
X = np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11], [8,2], [10,2], [9,3]])

# Initialize and setup the KMeans algorithm from sklearn
# Setting n_clusters to 2 and defining a fixed random_state for consistency in results
kmeans = KMeans(n_clusters=2, random_state=23, n_init=10)

# Execute the fitting process on the dataset
kmeans.fit(X)

# Retrieve the labels assigned by the model and the final computed centroids
predicted_labels = kmeans.labels_
final_centroids = kmeans.cluster_centers_

# Presentation: Scatter plot of clusters and their centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', s=60, label='Data Clusters')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=250, label='Final Centroids')

# Chart annotations for a professional look
plt.title("K-Means Clustering Analysis (using Scikit-Learn)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Display summary results
print("Final Cluster Centroids:\n", final_centroids)
print("Point-wise Cluster Labels:\n", predicted_labels)