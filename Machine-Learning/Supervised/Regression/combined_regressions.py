# Problem: Exploratory Data Analysis and Logistic Regression on Breast Cancer Dataset
# Description: This script performs Data Preparation, Feature Scaling, and Logistic 
# Regression classification on the UCI Breast Cancer dataset, showcasing the end-to-end workflow.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Data Acquisition
data = load_breast_cancer()
X = data.data
y = data.target

# Optional: Convert to Pandas DataFrame for easier visualization during development
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Step 2: Data Splitting
# Using a 70-30 split for training and validation purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Step 3: Feature Scaling
# Essential for Logistic Regression to ensure faster convergence and balanced feature importance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Initialization and Training
# Specifying max_iter to ensure convergence
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test_scaled)

# Display results
print(f"Data Shape: {X.shape}")
print(f"Sample Predictions: {y_pred[:10]}")
print(f"Model Accuracy (Scaled Data): {accuracy_score(y_test, y_pred) * 100:.2f}%")
