


# Binomial Logistic Regression on Breast Cancer Dataset

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y = load_breast_cancer(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=7)
clf=LogisticRegression(max_iter=10000,random_state=0)
clf.fit(X_train,y_train)
acc = accuracy_score(y_test,clf.predict(X_test))*100
print(f'Accuracy: {acc:.2f}%')

