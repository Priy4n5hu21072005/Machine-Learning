# Problem: K-Nearest Neighbors (KNN) Implementation from Scratch
# Description: This script implements a basic KNN algorithm from scratch using NumPy and 
# the collections module to classify a test point based on Euclidean distance to its neighbors.

import numpy as np
from collections import Counter

def euclidean_distance(X, Y):
    """
    Computes the Euclidean distance between two points (arrays).
    """
    return np.sqrt(np.sum((np.array(X) - np.array(Y)) ** 2))

def knn_prediction(training_data, training_label, test_points, k):
    """
    Predicts the class of a given test point based on training data.
    
    Args:
        training_data: List/Array of feature points.
        training_label: List/Array of corresponding labels.
        test_points: The point to classify.
        k: The number of neighbors to consider.
    """
    distances = []
    
    # Calculate Euclidean distance from the test point to all training points
    for i in range(len(training_data)):
        dist = euclidean_distance(test_points, training_data[i])
        distances.append((dist, training_label[i]))

    # Sort distances in ascending order (closest neighbors first)
    distances.sort(key=lambda x: x[0])

    # Identify the labels of the top 'k' nearest neighbors
    k_nearest = [label for _, label in distances[:k]]
    
    # Return the most frequent label among the neighbors (majority voting)
    return Counter(k_nearest).most_common(1)[0][0]

# Sample Training Dataset
training_data = [[1,2],[2,3],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
training_label = ['A','A','B','B','C','C','C','C']

# Target point to classify
test_point = [3,4]
k_neighbors = 3

# Execute prediction
prediction = knn_prediction(training_data, training_label, test_point, k_neighbors)
print(f"The predicted label for point {test_point} is: {prediction}")
