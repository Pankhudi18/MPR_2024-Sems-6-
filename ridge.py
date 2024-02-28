import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class RidgeRegression:
    def __init__(self, alpha=2.0):
        self.alpha = alpha  # Regularization parameter (lambda)

    def fit(self, X, y):
        # Add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        n_features = X.shape[1]
        
        # Compute coefficients using closed-form solution
        self.coef_ = np.linalg.inv(X.T.dot(X) + self.alpha * np.identity(n_features)).dot(X.T).dot(y)
        
    def predict(self, X):
        # Add a column of ones to X for the intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X.dot(self.coef_)

# Load data from CSV file
data = pd.read_csv('auto-mpg.csv')
    
X = data[['displacement']].values
y = data['mpg'].values.reshape(-1, 1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Ridge Regression model
ridge_reg = RidgeRegression(alpha=0.1)
ridge_reg.fit(X_train, y_train)

# Make predictions
y_pred_train = ridge_reg.predict(X_train)
y_pred_test = ridge_reg.predict(X_test)

# Print coefficients
print("Intercept:", ridge_reg.coef_[0])
print("Coefficient:", ridge_reg.coef_[1])

# Plot the training data and the best-fit line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, y_pred_train, color='red', label='Best Fit Line')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.title('Ridge Regression - Training Data')
plt.legend()
plt.show()

# Plot the test data and the best-fit line
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred_test, color='red', label='Best Fit Line')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Ridge Regression - Test Data')
plt.legend()
plt.show()

# Evaluate model on training data
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

print("Training Data:")
print("Mean Squared Error:", train_mse)
print("R-squared:", train_r2)

# Evaluate model on test data
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\nTest Data:")
print("Mean Squared Error:", test_mse)
print("R-squared:", test_r2)
