# Problem: Divisive Hierarchical Clustering (Top-Down) Implementation
# Description: This script demonstrates how a larger cluster is iteratively split 
# into smaller sub-clusters based on cluster variance and KMeans splitting.

import numpy as np
from sklearn.cluster import KMeans

# Pre-defined sample data for clustering demonstration
X = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])

def clustering_variance(cluster):
    """Calculates the sum of squared distances to the cluster mean (variance)."""
    if len(cluster) == 0:
        return 0
    mean = np.mean(cluster, axis=0)
    return np.sum((cluster - mean) ** 2)

# Initialize clustering process with the entire dataset as one cluster
initial_clusters = [X]

# Calculate initial variance for the root cluster
initial_variance = [clustering_variance(c) for c in initial_clusters]

# Identify the cluster with the highest variance to split (initially only one)
highest_variance_index = np.argmax(initial_variance)

# Use KMeans to split the identified cluster into two sub-clusters
kmeans_splitter = KMeans(n_clusters=2, random_state=23, n_init=10)
split_labels = kmeans_splitter.fit_predict(initial_clusters[highest_variance_index])

# Extract the sub-clusters based on predicted KMeans labels
sub_cluster1 = initial_clusters[highest_variance_index][split_labels == 1]
sub_cluster2 = initial_clusters[highest_variance_index][split_labels == 0]

# Update the overall cluster list by replacing the root with his children sub-clusters
# Note: For multiple iterations, this would be within a loop.
final_clusters = [sub_cluster1, sub_cluster2]

# Results Display for validation
print("Clustering Process Result:")
print("--------------------------")
print("Sub-Cluster 1 Results:\n", sub_cluster1)
print("Sub-Cluster 2 Results:\n", sub_cluster2)