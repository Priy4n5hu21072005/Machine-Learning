# Problem: K-Nearest Neighbors (KNN) from Scratch
# Description: This script implements the K-Nearest Neighbors algorithm from scratch using NumPy and the collections module to perform classification based on Euclidean distance.

import numpy as np

from collections import Counter

def euclidean_distance(X, Y):
    return np.sqrt(np.sum((np.array(X) - np.array(Y)) ** 2))

def knn_prediction(traning_data, traning_label, test_points, k):
    distance = []
    for i in range(len(traning_data)):
        dist = euclidean_distance(test_points, traning_data[i])
        distance.append((dist, traning_label[i]))

    distance.sort(key=lambda x: x[0])

    k_nearest = [label for _, label in distance[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

traning_data = [[1,2],[2,3],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
traning_label = ['A','A','B','B','C','C','C','C']
test_points = [3,4]
k = 3

print(knn_prediction(traning_data, traning_label, test_points, k))
