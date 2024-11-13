# Values Used:
# - X:  Example: X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]
# - y: . Example: y = [5, 7, 9, 11, 13]
# - theta: Parameters initialized to zero. Example: theta = [0, 0]
# - learning_rate:  Example: learning_rate = 0.1
# - iterations:. Example: iterations = 1000

import numpy as np
import matplotlib.pyplot as plt

# cost function
def compute_cost(X, y, theta):
    return (1 / (2 * len(y))) * np.sum((X.dot(theta) - y) ** 2)

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for i in range(iterations):
        theta -= (learning_rate / m) * X.T.dot(X.dot(theta) - y)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {compute_cost(X, y, theta)}, Theta = {theta}")
    return theta

# Linear Regression
def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    theta = np.zeros(X.shape[1])
    return gradient_descent(X, y, theta, learning_rate, iterations)

# Sample data
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Add intercept term
y = np.array([5, 7, 9, 11, 13])

# Run gradient descent
theta = linear_regression(X, y, learning_rate=0.1, iterations=1000)
print(f"Optimized theta: {theta}")

# Plot cost function history and fitted line
plt.plot(range(1000), [compute_cost(X, y, theta)] * 1000)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
plt.scatter(X[:, 1], y, color='blue')
plt.plot(X[:, 1], X.dot(theta), color='red')
plt.show()
