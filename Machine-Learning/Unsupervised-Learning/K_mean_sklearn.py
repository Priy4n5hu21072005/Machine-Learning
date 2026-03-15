from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],[8,2],[10,2],[9,3]])
kmean=KMeans(n_clusters=2,random_state=23)
kmean.fit(X)
labels=kmean.labels_
centroids=kmean.cluster_centers_
plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],c='red',marker='X',s=200)
plt.title("K-Mean through sklearn")
plt.show()