import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset into a pandas DataFrame
data = pd.read_csv('C:\python\Lib\CarPrice_Assignment.csv')

# Data preprocessing and feature selection
X = data[['citympg', 'highwaympg','compressionratio','enginesize','enginesize','boreratio','stroke','symboling','horsepower','peakrpm','carlength','carwidth','carheight']]  # Features (independent variables)
y = data['price']
for i in X.T:
    fmean = X.mean()
    frange = X.max() - X.min()
    X -= fmean
    X /= frange
print(X)

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize coefficients (beta) with zeros
beta = np.zeros(X_train.shape[1])

# Hyperparameters for gradient descent
learning_rate = 0.5
num_iterations = 100000

def costF(y_train,predictions):
    N = len(y_train)
    sq_error = (predictions - y_train) ** 2
    print(1.0 / (2 * N) * sq_error.sum())

# Perform gradient descent on the training set
X['constant'] = 1
for _ in range(num_iterations):
    # Calculate predictions on the training set
    predictions = np.dot(X_train, beta)

    # Calculate the gradient (derivative) of the mean squared error on the training set
    error = y_train-predictions
    gradient = np.dot(-X_train.T, error) / len(y_train)

    costF(y_train,predictions)
    # Update the coefficients using the gradient and learning rate
    beta -= gradient*learning_rate
    #print(beta)
# Make predictions on the testing set using the updated coefficients
predictions_test = np.dot(X_test, beta)

# Calculate accuracy
correct_predictions = 0
total_predictions = len(y_test)

for i in range(total_predictions):
    if abs(predictions_test[i] - y_test.iloc[i]) <= 1000:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions

print(f'Accuracy: {accuracy * 100:.2f}%')

# Visualize the data and the multiple linear regression line
plt.scatter(y_test, predictions_test, marker='o', color='blue', label='Predictions')
plt.scatter(y_test, y_test, marker='x', color='red', label='Actual Values')  # Use 'x' marker for actual values

plt.xlabel('Actual Prices (Test Set)')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs. Predicted Prices')
plt.legend()
plt.show()
