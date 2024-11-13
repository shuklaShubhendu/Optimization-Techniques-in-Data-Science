import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Define the Rosenbrock function (loss function)
def rosenbrock_loss(W):
    w1, w2 = W
    return (1 - w1)**2 + 100 * (w2 - w1**2)**2

# Define the gradient of the Rosenbrock function
def gradient_rosenbrock(W):
    w1, w2 = W
    grad_w1 = -2 * (1 - w1) - 400 * w1 * (w2 - w1**2)
    grad_w2 = 200 * (w2 - w1**2)
    return np.array([grad_w1, grad_w2])

# Adagrad optimization function
def adagrad(learning_rate, epsilon, max_iter=1000):
    # Initialize weight vector and gradient accumulation vector
    W = np.array([0.0, 0.0])  # Initial weights (w1_0, w2_0)
    G = np.array([0.0, 0.0])  # Initial gradient accumulation (g1_0, g2_0)
    path = []  # To track the path taken by the optimization

    # Iterate through the optimization steps
    for k in range(max_iter):
        # Compute the gradient of the Rosenbrock loss function
        grad = gradient_rosenbrock(W)
        
        # Update the gradient accumulation vector
        G += grad**2
        
        # Update weights using the Adagrad rule
        W -= learning_rate * grad / (np.sqrt(G + epsilon))
        
        # Store the path (weights at each step)
        path.append(W.copy())
        
        # Check for stopping criteria (optional: based on gradient norm)
        if np.linalg.norm(grad) < epsilon:
            print(f"Converged at iteration {k+1}")
            break

    return W, np.array(path)

# Set parameters
learning_rate = 0.1
epsilon = 1e-8
max_iter = 50  # Set number of iterations for visualization

# Run Adagrad to optimize the Rosenbrock function
optimal_weights, path = adagrad(learning_rate, epsilon, max_iter)

# Prepare data for plotting
w1_range = np.linspace(-2, 2, 100)
w2_range = np.linspace(-2, 2, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = np.array([rosenbrock_loss(np.array([w1, w2])) for w1, w2 in zip(np.ravel(W1), np.ravel(W2))])
Z = Z.reshape(W1.shape)

# Create figure for the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 2D Contour Plot setup
contour = ax1.contour(W1, W2, Z, 50, cmap='viridis')
ax1.set_xlabel('$w_1$')
ax1.set_ylabel('$w_2$')
ax1.set_title('Adagrad Optimization Path (2D Contour)')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
fig.colorbar(contour, ax=ax1, label="Loss")  # Corrected line

# 3D Surface Plot setup
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)
ax2.set_xlabel('$w_1$')
ax2.set_ylabel('$w_2$')
ax2.set_zlabel('Loss')
ax2.set_title('Adagrad Optimization Path (3D)')

# Function to update the plot for each iteration
def update_frame(i):
    # Clear previous paths
    for c in ax1.collections:
        c.remove()
    ax1.contour(W1, W2, Z, 50, cmap='viridis')  # Redraw the contour
    ax1.plot(path[:i+1, 0], path[:i+1, 1], marker='o', color='r', markersize=5)

    # Update 3D plot
    ax2.clear()
    ax2.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)
    ax2.plot(path[:i+1, 0], path[:i+1, 1], [rosenbrock_loss(w) for w in path[:i+1]], marker='o', color='r', markersize=5)
    ax2.set_xlabel('$w_1$')
    ax2.set_ylabel('$w_2$')
    ax2.set_zlabel('Loss')
    ax2.set_title('Adagrad Optimization Path (3D)')

# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=len(path), repeat=False)

# Save the animation as a .mp4 file (or .gif if preferred)
ani.save('adagrad_rosenbrock_optimization.mp4', writer='ffmpeg', fps=10)

plt.show()
