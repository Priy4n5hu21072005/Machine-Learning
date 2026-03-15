# Problem: Breast Cancer Classification using Random Forest
# Description: This script implements a Random Forest classifier using Scikit-Learn to predict breast cancer based on the UCI Breast Cancer Wisconsin dataset.

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    random_state=42
)
rf.fit(X_train,y_train)
y_train_pred=rf.predict(X_train)
y_test_pred=rf.predict(X_test)
print(accuracy_score(y_test,y_test_pred))
print(accuracy_score(y_train,y_train_pred))


