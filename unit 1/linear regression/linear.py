import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define the Linear Regression Model Class
class LinearRegressionModel:
    def __init__(self):
        self.W = None  # weights
        self.b = None  # bias

    def hypothesis(self, X):
        """Hypothesis function: y_hat = X * W + b"""
        return np.dot(X, self.W) + self.b

    def loss(self, X, y):
        """Compute the mean squared error (MSE) loss function"""
        predictions = self.hypothesis(X)
        errors = predictions - y
        return (1 / (2 * len(y))) * np.sum(errors ** 2)

    def gradient(self, X, y):
        """Compute the gradients for W and b"""
        predictions = self.hypothesis(X)
        errors = predictions - y
        gradient_W = (1 / len(y)) * np.dot(X.T, errors)
        gradient_b = (1 / len(y)) * np.sum(errors)
        return gradient_W, gradient_b

    def fit(self, X, y, learning_rate=0.01, iterations=1000):
        """Train the model using gradient descent"""
        # Initialize parameters (weights and bias)
        self.W = np.zeros(X.shape[1])  # d-dimensional vector (weights)
        self.b = 0
        
        # To track loss and weights for plotting
        loss_history = []
        weight_history = []
        iteration_numbers = []

        # Perform gradient descent
        for i in range(iterations):
            gradient_W, gradient_b = self.gradient(X, y)
            # Update parameters
            self.W -= learning_rate * gradient_W
            self.b -= learning_rate * gradient_b
            
            # Calculate current loss and append to loss history
            current_loss = self.loss(X, y)
            loss_history.append(current_loss)
            weight_history.append(self.W.copy())  # Store a copy of current weights
            iteration_numbers.append(i)
            
            # Print weights, bias, and loss for each iteration
            if i % 100 == 0:  # Print every 100 iterations for clarity
                print(f"Iteration {i}: Loss = {current_loss:.4f}, Weights = {self.W}, Bias = {self.b}")

        return loss_history, weight_history, iteration_numbers

# 2. Create a 3x3 input feature matrix (X) and a target vector (y)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])  # 3x3 matrix (3 samples, 3 features)

y = np.array([6, 15, 24])  # Target vector corresponding to each sample

# 3. Train the linear regression model
model = LinearRegressionModel()
loss_history, weight_history, iteration_numbers = model.fit(X, y, learning_rate=0.01, iterations=1000)

# 4. Plotting the 2D Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(iteration_numbers, loss_history, color='blue', label='Loss over Iterations')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve: Linear Regression Training Progress")
plt.grid(True)
plt.legend()
plt.show()

# 5. Plotting the 3D graph with iterations, weights, and loss
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare data for the 3D plot
iteration_numbers = np.array(iteration_numbers)
weight_history = np.array(weight_history)
loss_history = np.array(loss_history)

# Plot loss vs. iteration
ax.plot(iteration_numbers, loss_history, zs=0, zdir='z', label='Loss Curve', color='blue', linewidth=2)

# Plot weights in 3D: Iterate over each weight dimension (since weights have multiple dimensions)
for dim in range(weight_history.shape[1]):
    ax.plot(iteration_numbers, weight_history[:, dim], zs=dim + 1, zdir='y', label=f'Weight {dim + 1}')

# Labels for axes
ax.set_xlabel('Iteration')
ax.set_ylabel('Weight Dimension')
ax.set_zlabel('Loss / Weight Value')
ax.set_title('3D Visualization of Linear Regression: Loss and Weights')

# Adding legend
ax.legend()

plt.show()

# 6. Final predictions
predictions = model.hypothesis(X)

# 7. Print final results
print("\nFinal Trained Weights:", model.W)
print("Final Trained Bias:", model.b)
print("Final Predictions:", predictions)
