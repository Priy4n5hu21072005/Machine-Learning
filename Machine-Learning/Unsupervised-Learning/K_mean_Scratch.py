# Problem: K-Means Clustering from Scratch
# Description: This script implements the K-Means clustering algorithm from scratch using NumPy and visualizes the clusters and centroids using Matplotlib.

import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]])
k=2
np.random.seed(42)
centroid=X[np.random.choice(len(X),k,replace=False)]
def euclindean(x,y):
    return np.sqrt(np.sum((x-y)**2))
def assign_cluster(X,centroid):
    cluster=[]
    for p in X:
        distance=[euclindean(p,center) for center in centroid]
        cluster.append(np.argmin(distance))
    return np.array(cluster)
def update_centroid(X,cluster,k):
    new_centroid=[]
    for i in range(k):
        new_centroid.append(X[cluster==i].mean(axis=0))
    return np.array(new_centroid)
for _ in range(10):
    cluster=assign_cluster(X,centroid)
    new_centroid=update_centroid(X,cluster,k)
    if np.all(centroid==new_centroid):
        break
    centroid=new_centroid
print(centroid)
plt.scatter(X[:,0],X[:,1],c=cluster)
plt.scatter(centroid[:,0],centroid[:,1],c='red',marker='X',s=200)
plt.title("K-Mean clustering through scratch")
plt.show()

    