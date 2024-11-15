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

# Adagrad optimization function starting at (-2.8, -2.8)
def adagrad_start_from_custom(learning_rate, epsilon, max_iter=1000):
    W = np.array([-2.8, -2.8])  # Initial weights (starting point)
    G = np.array([0.0, 0.0])  # Initial gradient accumulation
    path = []  # To track the path taken by the optimization

    for k in range(max_iter):
        grad = gradient_rosenbrock(W)
        G += grad**2
        W -= learning_rate * grad / (np.sqrt(G + epsilon))
        path.append(W.copy())
        
        # Convergence check: stop if close to (1, 1)
        if np.linalg.norm(W - np.array([1, 1])) < 1e-3:
            print(f"Converged to (1, 1) at iteration {k+1}")
            break

    return W, np.array(path)

# Set parameters
learning_rate = 0.1
epsilon = 1e-8
max_iter = 3000

# Run Adagrad to optimize the Rosenbrock function
optimal_weights_custom, path_custom = adagrad_start_from_custom(learning_rate, epsilon, max_iter)

# Prepare data for plotting
w1_range = np.linspace(-3, 3, 100)
w2_range = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = np.array([rosenbrock_loss(np.array([w1, w2])) for w1, w2 in zip(np.ravel(W1), np.ravel(W2))])
Z = Z.reshape(W1.shape)

# Create figure for the plots
fig_custom, (ax1_custom, ax2_custom) = plt.subplots(1, 2, figsize=(14, 6))

# 2D Contour Plot setup
contour_custom = ax1_custom.contour(W1, W2, Z, levels=np.logspace(0, 3, 35), cmap='viridis')
ax1_custom.set_xlabel('$x$')
ax1_custom.set_ylabel('$y$')
ax1_custom.set_title('2D Contour Plot with Optimization Path')
ax1_custom.set_xlim(-3, 3)
ax1_custom.set_ylim(-3, 3)
fig_custom.colorbar(contour_custom, ax=ax1_custom, label="Loss")
iteration_text_2d = ax1_custom.text(-2.8, 2.8, '', color='red', fontsize=12, fontweight='bold')

# 3D Surface Plot setup
ax2_custom = fig_custom.add_subplot(122, projection='3d')
ax2_custom.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)
ax2_custom.set_xlabel('$x$')
ax2_custom.set_ylabel('$y$')
ax2_custom.set_zlabel('$f(x, y)$')
ax2_custom.set_title('3D Surface Plot with Optimization Path')
iteration_text_3d = ax2_custom.text2D(0.05, 0.95, '', transform=ax2_custom.transAxes, color='red', fontsize=12, fontweight='bold')

# Function to update the plot for each iteration
def update_frame_custom(i):
    # Update 2D contour plot with the optimization path up to iteration i
    ax1_custom.plot(path_custom[:i+1, 0], path_custom[:i+1, 1], 'ro-', markersize=3)
    iteration_text_2d.set_text(f"Iteration: {i+1}")
    
    # Update 3D path with the optimization path up to iteration i
    ax2_custom.plot(path_custom[:i+1, 0], path_custom[:i+1, 1], [rosenbrock_loss(w) for w in path_custom[:i+1]], 'ro-', markersize=3)
    iteration_text_3d.set_text(f"Iteration: {i+1}")

# Create the animation
ani_custom = animation.FuncAnimation(fig_custom, update_frame_custom, frames=len(path_custom), repeat=False)

# Show plot
plt.show()
