import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
X=np.array([[1,1],[2,1],[4,3],[5,4]])
model=AgglomerativeClustering(n_clusters=2,linkage='ward')
labels=model.fit_predict(X)
z=linkage(X,method='ward')
dendrogram(z)
plt.show()