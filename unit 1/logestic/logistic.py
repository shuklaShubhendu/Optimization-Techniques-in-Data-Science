# Values Used:
# - X: Feature matrix (including intercept term). Example: X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
# - y: Target labels (0 or 1). Example: y = [0, 0, 1, 1, 1]
# - theta: Parameters initialized to zero. Example: theta = [0, 0]
# - learning_rate: Step size for gradient descent. Example: learning_rate = 0.1
# - iterations: Number of times gradient descent will run. Example: iterations = 1000

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost function for Logistic Regression
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent for Logistic Regression
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for i in range(iterations):
        gradient = (1 / m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]}, Theta = {theta}")
    return theta, cost_history

# Logistic Regression
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    theta = np.zeros(X.shape[1])
    theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)
    return theta, cost_history

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Add intercept term
y = np.array([0, 0, 1, 1, 1])

# Run logistic regression
theta, cost_history = logistic_regression(X, y, learning_rate=0.1, iterations=1000)
print(f"Optimized theta: {theta}")

# Plot cost function history
plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function during gradient descent')
plt.show()

