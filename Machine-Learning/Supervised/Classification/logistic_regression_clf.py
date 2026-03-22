# Problem: Binomial Logistic Regression for Cancer Classification
# Description: This script demonstrates how to implement a Logistic Regression classifier 
# using the Scikit-Learn library on the UCI Breast Cancer dataset, evaluating the model's accuracy.

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the features and target labels directly from the dataset loader
X, y = load_breast_cancer(return_X_y=True)

# Partition data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=7)

# Initialize the Logistic Regression model
# Increased max_iter for convergence on the specific dataset and defined random_state for reproducibility
model = LogisticRegression(max_iter=10000, random_state=0)

# Training processing
model.fit(X_train, y_train)

# Calculate testing accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions) * 100

# Results output
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}%")
