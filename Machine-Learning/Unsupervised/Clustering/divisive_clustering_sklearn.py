# Problem: Ward's Minimum Variance Hierarchical Clustering (Agglomerative)
# Description: This script demonstrates the Ward linkage clustering method 
# which minimizes the total within-cluster variance when combining two clusters.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Defined dataset for clustering
X = np.array([[1, 1], [2, 1], [4, 3], [5, 4]])

# Step 1: Model Initialization using Scikit-Learn
# ward linkage is the default and minimizes the variance within merging clusters
hierarchical_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
cluster_labels = hierarchical_model.fit_predict(X)

# Step 2: Dendrogram Visualization with SciPy
# Using ward linkage for the linkage matrix to match the model logic
linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)

# Presentation Improvements
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)")
plt.xlabel("Data Point Indices")
plt.ylabel("Ward Distance (Variance minimization)")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()

# Final Clustering Results Display
print("Data Points and their Assigned Cluster Labels:")
for point, label in zip(X, cluster_labels):
    print(f"Point: {point} -> Cluster: {label}")