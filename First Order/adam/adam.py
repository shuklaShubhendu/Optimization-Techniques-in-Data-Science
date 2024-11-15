import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from math import sqrt
import warnings

class AdamOptimizer:
    def __init__(self, alpha=0.01, beta1=0.85, beta2=0.995, epsilon=1e-8):
        """
        Initialize Adam optimizer with faster convergence parameters
        - Higher initial learning rate with decay for faster convergence
        - Smaller beta1 for quicker adaptation
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.target = np.array([1.0, 1.0])
        self.tolerance = 1e-4  # Slightly looser tolerance for faster convergence

    @staticmethod
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    @staticmethod
    def rosenbrock_gradient(x, y):
        dfdx = -2 * (1 - x) - 400 * x * (y - x**2)
        dfdy = 200 * (y - x**2)
        return np.array([dfdx, dfdy])

    def optimize(self, max_iterations=1000, x_start=None):
        """
        Perform Adam optimization with early stopping and learning rate decay
        """
        solutions = []
        distances = []
        
        # Start closer to the minimum
        x = x_start if x_start is not None else np.array([0.5, 0.5])  # Closer start to (1,1)
        
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        for t in range(max_iterations):
            g = self.rosenbrock_gradient(x[0], x[1])
            
            # Update moment estimates
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * g**2
            
            # Bias correction
            mhat = m / (1.0 - self.beta1**(t+1))
            vhat = v / (1.0 - self.beta2**(t+1))
            
            # Decaying learning rate
            effective_alpha = self.alpha * (0.9 ** (t / max_iterations))
            
            # Update with decayed learning rate
            update = effective_alpha * mhat / (np.sqrt(vhat) + self.epsilon)
            x = x - update
            
            solutions.append(x.copy())
            current_distance = np.linalg.norm(x - self.target)
            distances.append(current_distance)
            
            # Early stopping if we're close enough to the minimum
            if current_distance < self.tolerance:
                print(f"Converged after {t+1} iterations!")
                break
                
            # Adaptive momentum adjustment if needed
            if t > 0 and distances[-1] > distances[-2]:
                self.beta1 *= 0.98  # Reduce momentum if moving away from the minimum
        
        return np.array(solutions), np.array(distances)

class OptimizationVisualizer:
    def __init__(self, optimizer, n_points=100):
        self.optimizer = optimizer
        self.n_points = n_points
        self.setup_mesh_grid()
        
    def setup_mesh_grid(self):
        x = np.linspace(-2, 2, self.n_points)
        y = np.linspace(-1, 3, self.n_points)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = self.optimizer.rosenbrock(self.X, self.Y)

    def create_animation(self, solutions, distances, interval=100):  # Faster animation
        x_vals, y_vals = solutions[:, 0], solutions[:, 1]
        path_z = [self.optimizer.rosenbrock(x, y) for x, y in solutions]

        fig = plt.figure(figsize=(14, 6))
        ax2d = fig.add_subplot(1, 2, 1)
        ax3d = fig.add_subplot(1, 2, 2, projection='3d')

        # 2D plot setup
        levels = np.logspace(-1, 3, 20)
        ax2d.contour(self.X, self.Y, self.Z, levels=levels, cmap='viridis')
        line_2d, = ax2d.plot([], [], 'r.-', label='Optimization Path', alpha=0.7)
        target_point_2d, = ax2d.plot([1], [1], 'g*', markersize=15, label='Global Minimum (1,1)')
        current_point_2d, = ax2d.plot([], [], 'bo', markersize=8, label='Current Point')
        
        ax2d.set_xlabel('X')
        ax2d.set_ylabel('Y')
        ax2d.set_title('2D View of Adam Optimization')
        ax2d.legend()
        ax2d.grid(True)
        text_2d = ax2d.text(0.02, 0.98, '', transform=ax2d.transAxes, 
                           verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        # 3D plot setup
        surf = ax3d.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.6)
        line_3d, = ax3d.plot([], [], [], 'r.-', label='Optimization Path', alpha=0.7)
        target_point_3d = ax3d.scatter([1], [1], [self.optimizer.rosenbrock(1, 1)], 
                                     color='green', s=100, marker='*', label='Global Minimum')
        current_point_3d = ax3d.scatter([], [], [], color='blue', s=100)
        
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Function Value')
        ax3d.set_title('3D View of Adam Optimization')
        ax3d.view_init(elev=30, azim=45)
        text_3d = ax3d.text2D(0.02, 0.98, '', transform=ax3d.transAxes, 
                             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        def update(frame):
            # Update 2D plot
            line_2d.set_data(x_vals[:frame+1], y_vals[:frame+1])
            current_point_2d.set_data([x_vals[frame]], [y_vals[frame]])
            text_2d.set_text(f'Iteration: {frame+1}\nDistance to (1,1): {distances[frame]:.6f}')
            
            # Update 3D plot
            line_3d.set_data(x_vals[:frame+1], y_vals[:frame+1])
            line_3d.set_3d_properties(path_z[:frame+1])
            current_point_3d._offsets3d = ([x_vals[frame]], [y_vals[frame]], [path_z[frame]])
            text_3d.set_text(f'Iteration: {frame+1}\nDistance to minimum: {distances[frame]:.6f}')
            
            return line_2d, current_point_2d, text_2d, line_3d, current_point_3d, text_3d

        ani = FuncAnimation(fig, update, frames=len(solutions), 
                          interval=interval, blit=True, repeat=False)
        plt.tight_layout()
        return ani

def main():
    np.random.seed(42)
    
    # Initialize optimizer with faster convergence parameters
    optimizer = AdamOptimizer(alpha=0.01, beta1=0.85, beta2=0.995)
    
    # Start closer to the minimum
    x_start = np.array([0.5, 0.5])
    
    # Run optimization with early stopping
    solutions, distances = optimizer.optimize(max_iterations=10000, x_start=x_start)
    
    # Create and display visualization
    visualizer = OptimizationVisualizer(optimizer)
    ani = visualizer.create_animation(solutions, distances, interval=100)
    plt.show()

if __name__ == "__main__":
    main()
