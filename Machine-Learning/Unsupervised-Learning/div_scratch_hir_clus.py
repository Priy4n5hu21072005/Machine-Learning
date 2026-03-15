import numpy as np
from sklearn.cluster import KMeans
X=np.array([[1,1],[2,1],[4,3],[5,4]])
def clustering_variance(cluster):
    mean=np.mean(cluster,axis=0)
    return np.sum((cluster-mean)**2)
cluster=[X]
variance=[clustering_variance(c) for c in cluster]
idx=np.argmax(variance)
kmean=KMeans(n_clusters=2,random_state=23)
labels=kmean.fit_predict(cluster[idx])
cluster1=cluster[idx][labels==1]
cluster2=cluster[idx][labels==0]
cluster.pop(idx)
cluster.append(cluster1)
cluster.append(cluster2)
print("This is the cluster 1 :",cluster1)
print("This is the cluster 2 :",cluster2)