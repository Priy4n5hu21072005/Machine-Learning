# Problem: Linear Regression and Visualization
# Description: This script displays the fitting of a simple linear regression model 
# on synthetic noise-injected data, including a visualization of the regression line.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic linear data with random noise for modeling
np.random.seed(42)
X = np.random.rand(50, 1) * 100
Y = 3.5 * X + np.random.rand(50, 1) * 20

# Initialize and fit the Linear Regression model to the generated points
model = LinearRegression()
model.fit(X, Y)

# Use the trained model to predict Y values based on X
Y_pred = model.predict(X)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', label='Actual Data Points')
plt.plot(X, Y_pred, color='red', linewidth=2, label='Fitted Regression Line')

# Chart annotations for a professional presentation
plt.title('Simple Linear Regression Analysis (Synthetic Dataset)')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Coefficients output
print(f"Slope (Coefficient): {model.coef_[0][0]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")