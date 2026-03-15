from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
X,y=make_classification(
    n_samples=100,
    n_features=2,
    n_clusters_per_class=1,
    n_redundant=0,
    class_sep=0.5,
    random_state=23,
    flip_y=0.1
)
model=SVC(kernel="linear",C=0.1)
model.fit(X,y)
w=model.coef_[0]
b=model.intercept_[0]
x_points=np.linspace(X[:,0].min(),X[:,0].max())
y_boundary=-(w[0]*x_points+b)/w[1]
plt.scatter(X[:,0],X[:,1],c=y)
plt.scatter(
    model.support_vectors_[:,0],
    model.support_vectors_[:,1],
    s=150,
    facecolors ="none",
    edgecolors="Red"
)
plt.plot(x_points,y_boundary)
plt.title("Linear spearble data")
plt.show()