# Problem: Breast Cancer Detection using Decision Tree
# Description: This script implements a Decision Tree Classifier to predict breast cancer 
# based on feature data, evaluating both training and testing accuracy.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the dataset (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Initialize the Decision Tree Classifier
# Using 'gini' criterion for node splitting and a max depth of 5 to prevent overfitting
dt = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=23
)

# Train the model on the training set
dt.fit(X_train, y_train)

# Perform predictions for performance evaluation
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Output evaluation metrics
print(f"Decision Tree - Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%")
print(f"Decision Tree - Testing Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%")
