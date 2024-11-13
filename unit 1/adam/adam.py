# Parameters used:
# W_init: Initial weight parameter (W = 10.0)
# alpha: Learning rate (0.1)
# beta1: Smoothing parameter for first moment (0.9)
# beta2: Smoothing parameter for second moment (0.99)
# epsilon: Small constant for numerical stability (1e-8)
# max_iters: Maximum number of iterations (1000)

import numpy as np
import matplotlib.pyplot as plt

# Define a sample smooth loss function L(W) = W^2
def loss_function(W):
    return W**2

# Define the gradient of the loss function
def gradient(W):
    return 2 * W

# Adam optimization function
def adam_optimizer(loss_function, gradient, W_init, alpha=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8, max_iters=1000):
    W = W_init
    M, G = 0, 0
    path = [W]

    for t in range(1, max_iters + 1):
        grad = gradient(W)

        # Update biased moments
        M = beta1 * M + (1 - beta1) * grad
        G = beta2 * G + (1 - beta2) * grad**2

        # Bias-corrected moments
        M_hat = M / (1 - beta1**t)
        G_hat = G / (1 - beta2**t)

        # Update weight
        W = W - alpha * M_hat / (np.sqrt(G_hat) + epsilon)
        path.append(W)

        if np.abs(grad) < epsilon:
            break

    return W, np.array(path)

# Parameters for Adam
W_init = 10.0
alpha = 0.1
beta1 = 0.9
beta2 = 0.99
epsilon = 1e-8

# Run Adam optimizer
optimal_W, path = adam_optimizer(loss_function, gradient, W_init, alpha, beta1, beta2, epsilon)

# Print the results
print("Optimal W:", optimal_W)
print("Path of W:", path)

# Plotting in 2D
W_values = np.linspace(-10, 10, 400)
loss_values = loss_function(W_values)

plt.figure(figsize=(10, 5))
plt.plot(W_values, loss_values, label="Loss Function L(W) = W^2")
plt.plot(path, loss_function(path), 'ro-', markersize=4, label="Optimization Path")
plt.xlabel("Weight (W)")
plt.ylabel("Loss L(W)")
plt.title("2D Plot of Loss Function with Adam Optimization Path")
plt.legend()
plt.grid(True)
plt.show()

# Plotting in 3D
from mpl_toolkits.mplot3d import Axes3D

W_range = np.linspace(-10, 10, 100)
W_mesh, Loss_mesh = np.meshgrid(W_range, loss_function(W_range))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W_mesh, Loss_mesh, Loss_mesh, cmap="viridis", alpha=0.6)
ax.plot(path, loss_function(path), loss_function(path), 'ro-', markersize=5, label="Optimization Path")

ax.set_xlabel("Weight (W)")
ax.set_ylabel("Loss L(W)")
ax.set_zlabel("Loss Value")
ax.set_title("3D Plot of Loss Function Surface and Adam Optimization Path")
plt.legend()
plt.show()
