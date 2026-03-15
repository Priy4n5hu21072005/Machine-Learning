from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = load_breast_cancer()
X=data.data
y=data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)
# Train the model
dt = DecisionTreeClassifier(
    criterion='gini', # gini formula 
    max_depth=5,
    random_state=23)
dt.fit(X_train,y_train)
y_train_pred=dt.predict(X_train)
y_test_pred=dt.predict(X_test)
print("Accuracy of train data",accuracy_score(y_train,y_train_pred))
print("Accuracy of test data",accuracy_score(y_test,y_test_pred))
