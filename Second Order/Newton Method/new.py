import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
#hi this is small test change
EPSILON = 0.0001

class NewtonMethod:
    def __init__(self, alpha=0.01, damp_factor=0.1, alpha_optim=None):
        self.alpha = alpha
        self.damp_factor = damp_factor
        self.alpha_optim = alpha_optim
        self.path = []  # To store the optimization path
        self.num_iter = 0  # Initialize the iteration counter

    def _reset(self):
        self.alpha = 0.01
        self.damp_factor = 0.1
        self.num_iter = 0

    def _next(self, x, func_callback, grad_func_callback, hessian_func_callback):
        if np.all(np.linalg.eigvals(hessian_func_callback(x)) > 0):
            return x - np.linalg.inv(hessian_func_callback(x)) @ grad_func_callback(x)
        else:
            # Levenberg-Marquardt modification
            delta = np.linalg.inv(
                hessian_func_callback(x) + self.damp_factor * np.identity(x.shape[0])
            ) @ grad_func_callback(x)
            return x - self.alpha * delta

    def optimize(self, x, func_callback, grad_func_callback, hessian_func_callback, is_plot=False):
        self.path = [x]
        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)
            self.path.append(x)
        
        self._reset()

        if is_plot:
            path = np.array(self.path)
            x_vals, y_vals = path[:, 0], path[:, 1]

            # Generate mesh grid for the Rosenbrock function
            x = np.linspace(-2, 2, 100)
            y = np.linspace(-1, 3, 100)
            X, Y = np.meshgrid(x, y)
            Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

            # Initialize figure and subplots for 2D and 3D plots
            fig = plt.figure(figsize=(14, 6))
            ax2d = fig.add_subplot(1, 2, 1)
            ax3d = fig.add_subplot(1, 2, 2, projection='3d')

            # Set up 2D contour plot
            ax2d.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
            line_2d, = ax2d.plot([], [], 'o-', color='red', label='Newton Method Path')
            goal_point_2d, = ax2d.plot([], [], 'go', label='Final Point')
            ax2d.set_xlabel('X')
            ax2d.set_ylabel('Y')
            ax2d.set_title('2D Path of Newton Method')
            ax2d.legend()
            text_2d = ax2d.text(-1.5, 2.8, '', fontsize=12)

            # Set up 3D surface plot
            ax3d.plot_surface(X, Y, Z, cmap='jet', alpha=0.6, edgecolor='none')
            path_z = (1 - x_vals)**2 + 100 * (y_vals - x_vals**2)**2
            line_3d, = ax3d.plot([], [], [], 'o-', color='red', label='Newton Method Path')
            goal_point_3d = ax3d.scatter([], [], [], color='green', s=50, label='Final Point')
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Z (Rosenbrock)')
            ax3d.set_title('3D Path of Newton Method')
            ax3d.view_init(elev=30, azim=120)
            ax3d.legend()
            text_3d = ax3d.text2D(0.05, 0.9, '', transform=ax3d.transAxes, fontsize=12)

            # Update function for animation
            def update(i):
                # Update 2D plot
                line_2d.set_data(x_vals[:i + 1], y_vals[:i + 1])
                goal_point_2d.set_data([x_vals[-1]], [y_vals[-1]])  # Pass lists (sequences) here
                text_2d.set_text(f'Iteration: {i + 1}')

                # Update 3D plot
                line_3d.set_data(x_vals[:i + 1], y_vals[:i + 1])
                line_3d.set_3d_properties(path_z[:i + 1])
                goal_point_3d._offsets3d = ([x_vals[-1]], [y_vals[-1]], [path_z[-1]])  # Pass lists here as well
                text_3d.set_text(f'Iteration: {i + 1}')

                plt.pause(0.05)  # Pause for 50 milliseconds between frames
                return line_2d, goal_point_2d, text_2d, line_3d, goal_point_3d, text_3d

            # Create the animation with increased interval and stop until user closes the window
            ani = FuncAnimation(fig, update, frames=len(self.path), interval=300, repeat=False)

            plt.show()
            return x, self.path

        return x

# Define the gradient and hessian functions for the Rosenbrock function
def rosenbrock_grad(x):
    grad_x = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad_y = 200 * (x[1] - x[0]**2)
    return np.array([grad_x, grad_y])

def rosenbrock_hessian(x):
    hessian = np.zeros((2, 2))
    hessian[0, 0] = 1200 * x[0]**2 - 400 * x[1] + 2
    hessian[0, 1] = -400 * x[0]
    hessian[1, 0] = -400 * x[0]
    hessian[1, 1] = 200
    return hessian

# Set initial point and create the Newton Method optimizer
initial_point = np.array([-1.2, 1.0])
newton_method = NewtonMethod(alpha=0.01, damp_factor=0.1)

# Run optimization with path plotting enabled
result, path = newton_method.optimize(initial_point, lambda x: rosenbrock_grad(x), rosenbrock_grad, rosenbrock_hessian, is_plot=True)


