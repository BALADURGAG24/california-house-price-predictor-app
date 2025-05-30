# -*- coding: utf-8 -*-
"""California House Price Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nv65KwbMVyS7gmhmlQefZXdiAbTSx3_6
"""

pip install numpy pandas matplotlib seaborn scikit-learn xgboost

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load the California Housing dataset
housing = fetch_california_housing()
house_price_dataframe = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target (price) to the dataframe
house_price_dataframe['price'] = housing.target

# Display first 5 rows
print(house_price_dataframe.head())

# Shape of the dataset
print("Dataset shape:", house_price_dataframe.shape)

# Check for missing values
print("\nMissing values:\n", house_price_dataframe.isnull().sum())

# Statistical summary
print("\nDataset description:\n", house_price_dataframe.describe())

# Correlation matrix
correlation = house_price_dataframe.corr()

# Heatmap for correlation
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting data into features and target
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

# Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Initialize and train XGBoost model
model = XGBRegressor(random_state=42)
model.fit(X_train, Y_train)

# Predict on training data
train_predictions = model.predict(X_train)

# Training performance
r2_train = r2_score(Y_train, train_predictions)
mae_train = mean_absolute_error(Y_train, train_predictions)

print("\nTraining R2 Score:", r2_train)
print("Training MAE:", mae_train)

# Scatter plot for training data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_train, y=train_predictions, alpha=0.5)
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Training Data)")
plt.show()

# Predict on test data
test_predictions = model.predict(X_test)

# Test performance
r2_test = r2_score(Y_test, test_predictions)
mae_test = mean_absolute_error(Y_test, test_predictions)
rmse_test = np.sqrt(mean_squared_error(Y_test, test_predictions))

print("\nTest R2 Score:", r2_test)
print("Test MAE:", mae_test)
print("Test RMSE:", rmse_test)

# Scatter plot for test data
plt.figure(figsize=(8, 6))
sns.scatterplot(x=Y_test, y=test_predictions, alpha=0.5, color='green')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices (Test Data)")
plt.show()