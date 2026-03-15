# Problem: Income Prediction using Random Forest Regressor
# Description: This script implements a Random Forest regression model to predict personal income based on demographic data, utilizing Scikit-Learn pipelines and preprocessing.

import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score 
df = pd.read_csv("datainfo.csv")
df = df.drop(columns=["phone_number","name","email"])
X = df.drop(columns=["income"])
y = df["income"]
num_cols=["age"]
cat_cols=[col for col in X.columns if col not in num_cols]
prepro=ColumnTransformer(
    transformers=[
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols),
        ("num","passthrough",num_cols)
    ]
)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=23,
    n_jobs=-1
)
model = Pipeline(steps=[
    ("preprocessing",prepro),
    ("model",rf)
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mean = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)