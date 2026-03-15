from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
data=load_breast_cancer()
X = data.data
y =data.target
print(X.shape)
print(y.shape)
data.feature_names
data.target_names
set(y)
df = pd.DataFrame(X,columns=data.feature_names)
df['target']=y
df.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
scal=StandardScaler()
X_train_scaled=scal.fit_transform(X_train)
X_test_scaled=scal.transform(X_test)
print(X_train[0][:5])
print(X_test_scaled[0][:5])
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
print(y_pred[:10])
print(accuracy_score(y_test,y_pred))
