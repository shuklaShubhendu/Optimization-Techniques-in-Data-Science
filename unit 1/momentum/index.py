import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the Rosenbrock function and its gradient
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def gradient_rosenbrock(x):
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

# Optimized Momentum Gradient Descent with Iteration Tracking
def momentum_gradient_descent(f, grad_f, x_init, gamma=0.9, alpha=0.001, epsilon=1e-6, max_iter=10000):
    x_k = x_init
    v_k = np.zeros_like(x_k)
    path = [x_k.copy()]
    
    for _ in range(max_iter):
        grad = grad_f(x_k)
        if np.linalg.norm(grad) < epsilon:
            break
        v_k = gamma * v_k + alpha * grad
        x_k = x_k - v_k
        path.append(x_k.copy())
        
    return x_k, np.array(path)

# Initial parameters
x_init = np.array([-1.5, 1.5])
gamma, alpha, epsilon = 0.9, 0.001, 1e-6

# Run optimizer and get path
_, path = momentum_gradient_descent(rosenbrock, gradient_rosenbrock, x_init, gamma, alpha, epsilon)

# Generate a grid for the Rosenbrock function
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# Set up the figure and axes for the 2D and 3D plots
fig = plt.figure(figsize=(14, 6))

# 2D Contour Plot
ax1 = fig.add_subplot(1, 2, 1)
contour = ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('2D Contour Plot with Optimization Path')

# 3D Surface Plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='k')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x, y)')
ax2.set_title('3D Surface Plot with Optimization Path')

# Initialize paths on both plots
path_x, path_y = path[:, 0], path[:, 1]
line2d, = ax1.plot([], [], 'r-', marker='o', markersize=4)
line3d, = ax2.plot([], [], [], 'r-', marker='o', markersize=4)

# Initialize variables for iteration text
iteration_text_2d = None
iteration_text_3d = None
final_annotation_2d = None
final_annotation_3d = None

# Update function for animation
def update(i):
    global final_annotation_2d, final_annotation_3d, iteration_text_2d, iteration_text_3d, ani
    
    # Update 2D plot
    line2d.set_data(path_x[:i+1], path_y[:i+1])
    
    # Update 3D plot
    line3d.set_data(path_x[:i+1], path_y[:i+1])
    line3d.set_3d_properties(rosenbrock([path_x[:i+1], path_y[:i+1]]))

    # Remove previous iteration number annotations
    if iteration_text_2d:
        iteration_text_2d.remove()
    if iteration_text_3d:
        iteration_text_3d.remove()
    
    # Add the current iteration number as text on the graph
    iteration_text_2d = ax1.text(0.05, 0.95, f"Iteration: {i+1}", transform=ax1.transAxes, 
                                 fontsize=12, color='red', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))
    iteration_text_3d = ax2.text2D(0.05, 0.95, f"Iteration: {i+1}", transform=ax2.transAxes, 
                                   fontsize=12, color='red', ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7))

    # Stop the animation at the final iteration
    if i == len(path) - 1:
        # Final point in green
        ax1.plot(path_x[i], path_y[i], 'go', markersize=6)  # Green dot on 2D plot
        ax2.plot([path_x[i]], [path_y[i]], [rosenbrock([path_x[i], path_y[i]])], 'go', markersize=6)  # Green dot on 3D plot
        
        # Add iteration number as a text annotation for the final iteration
        final_annotation_2d = ax1.annotate(f"Iter: {i+1}", xy=(path_x[i], path_y[i]), 
                                           xytext=(5, 5), textcoords='offset points', color='green')
        final_annotation_3d = ax2.text(path_x[i], path_y[i], rosenbrock([path_x[i], path_y[i]]), 
                                       f"Iter: {i+1}", color='green')
        
        # Stop the animation at the final frame
        ani.event_source.stop()

    return line2d, line3d

# Create the animation
ani = FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)

plt.show()
