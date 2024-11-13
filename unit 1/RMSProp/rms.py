import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the RMSProp optimizer class with enhanced stopping criteria
class RMSProp:
    def __init__(self, learning_rate=0.01, decay=0.99, epsilon=1e-8, max_iterations=300, tol=1e-6):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tol = tol
        self.path = []

    def optimize(self, initial_point, gradient_func):
        current_point = initial_point
        num_iter = 0
        grad_sq = np.zeros_like(current_point)
        consecutive_small_steps = 0

        while num_iter < self.max_iterations:
            self.path.append(current_point.copy())
            grad = gradient_func(current_point)
            
            if np.linalg.norm(grad) < self.tol:
                print(f"Convergence achieved at iteration {num_iter} due to small gradient norm.")
                break
            
            grad_sq = self.decay * grad_sq + (1 - self.decay) * grad**2
            adjusted_grad = grad / (np.sqrt(grad_sq) + self.epsilon)
            next_point = current_point - self.learning_rate * adjusted_grad

            if np.linalg.norm(next_point - current_point) < self.tol:
                consecutive_small_steps += 1
            else:
                consecutive_small_steps = 0

            if consecutive_small_steps >= 5:
                print(f"Convergence achieved at iteration {num_iter} due to consecutive small parameter changes.")
                break

            current_point = next_point
            num_iter += 1

        return current_point

# Define the gradient of the Rosenbrock function
def rosenbrock_gradient(point):
    x, y = point
    grad_x = -2 * (1 - x) - 400 * x * (y - x**2)
    grad_y = 200 * (y - x**2)
    return np.array([grad_x, grad_y])

# Set initial point and run optimizer
initial_point = np.array([-1.2, 1.0])
rmsprop_optimizer = RMSProp(learning_rate=0.1, decay=0.99)
result = rmsprop_optimizer.optimize(initial_point, rosenbrock_gradient)
path = np.array(rmsprop_optimizer.path)
x_vals, y_vals = path[:, 0], path[:, 1]

# Generate mesh grid for the Rosenbrock function
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# Initialize figure and subplots for 2D and 3D plots
fig = plt.figure(figsize=(14, 6))
ax2d = fig.add_subplot(1, 2, 1)
ax3d = fig.add_subplot(1, 2, 2, projection='3d')

# Set up 2D contour plot
ax2d.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
line_2d, = ax2d.plot([], [], 'o-', color='red', label='RMSProp Path')
goal_point_2d, = ax2d.plot([], [], 'go', label='Final Point')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('2D Path of RMSProp Optimization')
ax2d.legend()
text_2d = ax2d.text(-1.5, 2.8, '', fontsize=12)

# Set up 3D surface plot
ax3d.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, edgecolor='none')
path_z = (1 - x_vals)**2 + 100 * (y_vals - x_vals**2)**2
line_3d, = ax3d.plot([], [], [], 'o-', color='red', label='RMSProp Path')
goal_point_3d = ax3d.scatter([], [], [], color='green', s=50, label='Final Point')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z (Rosenbrock)')
ax3d.set_title('3D Path of RMSProp Optimization')
ax3d.view_init(elev=30, azim=120)
ax3d.legend()
text_3d = ax3d.text2D(0.05, 0.9, '', transform=ax3d.transAxes, fontsize=12)

# Update function for animation
def update(i):
    line_2d.set_data(x_vals[:i+1], y_vals[:i+1])
    goal_point_2d.set_data([x_vals[-1]], [y_vals[-1]])
    text_2d.set_text(f'Iteration: {i+1}')
    
    line_3d.set_data(x_vals[:i+1], y_vals[:i+1])
    line_3d.set_3d_properties(path_z[:i+1])
    goal_point_3d._offsets3d = ([x_vals[-1]], [y_vals[-1]], [path_z[-1]])
    text_3d.set_text(f'Iteration: {i+1}')

    plt.pause(0.05)  # Pause for 50 milliseconds between frames for smooth pacing
    return line_2d, goal_point_2d, text_2d, line_3d, goal_point_3d, text_3d

# Create the animation with adjusted interval and stopping behavior
ani = FuncAnimation(fig, update, frames=len(path), interval=300, repeat=False)

# Show plot and wait for the window to close
plt.show()
