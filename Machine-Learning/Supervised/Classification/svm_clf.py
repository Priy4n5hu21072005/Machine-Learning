# Problem: Linear Separability using Support Vector Classifier (SVC)
# Description: This script demonstrates SVM classification on synthetic data, 
# visualizing the decision boundary and support vectors.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generate synthetic binary classification data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_clusters_per_class=1,
    n_redundant=0,
    class_sep=0.5,
    random_state=23,
    flip_y=0.1
)

# Initialize and train the Support Vector Classifier with a linear kernel
# C is the regularization parameter
model = SVC(kernel="linear", C=0.1)
model.fit(X, y)

# Extract weights and intercept to calculate the decision boundary
w = model.coef_[0]
b = model.intercept_[0]

# Calculate points for the decision boundary line
x_points = np.linspace(X[:, 0].min(), X[:, 0].max())
y_boundary = -(w[0] * x_points + b) / w[1]

# Plot mapping: Scatter plot for data points, colored by class
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Data Points')

# Highlight the Support Vectors (the critical points that define the margin)
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="Red",
    label='Support Vectors'
)

# Plot the decision boundary
plt.plot(x_points, y_boundary, color='black', linestyle='--', label='Decision Boundary')

# Add titles and labels for clarity
plt.title("SVM Linear Classification with Support Vectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()