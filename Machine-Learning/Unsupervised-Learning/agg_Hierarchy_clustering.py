# This is the agglomerative hierarchy clustering 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster import AgglomerativeClustering
X=np.array([[1],[3],[8],[9]])
link=linkage(X,method='single')
plt.figure(figsize=(5,4))
dendrogram(link)
plt.title("Aggeromative Clustering through dendrogram")
plt.show()
model=AgglomerativeClustering(distance_threshold=0,n_clusters=None)
labels=model.fit_predict(X)
for points,label in zip(X,labels):
    print(points,"-clustering",label)