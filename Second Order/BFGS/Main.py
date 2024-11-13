import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    '''Objective function (Rosenbrock function)'''
    d = len(x)
    return sum(100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(d-1))

def grad(f, x):
    '''Gradient calculation using central finite difference'''
    h = np.cbrt(np.finfo(float).eps)
    d = len(x)
    nabla = np.zeros(d)
    for i in range(d):
        x_for = np.copy(x)
        x_back = np.copy(x)
        x_for[i] += h
        x_back[i] -= h
        nabla[i] = (f(x_for) - f(x_back)) / (2 * h)
    return nabla

def line_search(f, x, p, nabla):
    '''Backtracking line search with Wolfe conditions'''
    a = 1
    c1 = 1e-4
    c2 = 0.9
    fx = f(x)
    x_new = x + a * p
    nabla_new = grad(f, x_new)
    while f(x_new) >= fx + c1 * a * nabla.T @ p or nabla_new.T @ p <= c2 * nabla.T @ p:
        a *= 0.5
        x_new = x + a * p
        nabla_new = grad(f, x_new)
    return a

def BFGS(f, x0, max_it):
    '''BFGS Quasi-Newton method implementation'''
    d = len(x0)
    nabla = grad(f, x0)
    H = np.eye(d)
    x = x0[:]
    it = 2

    # Store all iteration data for plotting
    x_vals_for_plotting = []  # For 2D plotting
    y_vals_for_plotting = []  # For 2D plotting
    z_vals_for_plotting = []  # For 3D plotting

    # Set up the figure for dynamic plotting
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection='3d')

    while np.linalg.norm(nabla) > 1e-5:  # While gradient is non-zero
        if it > max_it:
            print('Maximum iterations reached!')
            break
        it += 1
        p = -H @ nabla  # Search direction (Newton direction)
        a = line_search(f, x, p, nabla)  # Line search to find optimal step size
        s = a * p
        x_new = x + a * p
        nabla_new = grad(f, x_new)
        y = nabla_new - nabla
        s = np.reshape(s, (d, 1))
        y = np.reshape(y, (d, 1))
        r = 1 / (y.T @ s)
        li = np.eye(d) - r * (s @ y.T)
        ri = np.eye(d) - r * (y @ s.T)
        H = li @ H @ ri + r * (s @ s.T)

        nabla = nabla_new[:]
        x = x_new[:]

        # Store data for plotting
        x_vals_for_plotting.append(x[0])
        y_vals_for_plotting.append(x[1])
        z_vals_for_plotting.append(f(x))

        # Update 2D plot
        ax.clear()
        ax.plot(range(len(x_vals_for_plotting)), z_vals_for_plotting, marker='o', color='b', label='f(x) over iterations')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('f(x)')
        ax.set_title(f'2D Plot: Objective function value vs Iterations (Iteration {it})')
        ax.grid(True)
        ax.legend()

        # Update 3D plot
        ax3d.clear()
        ax3d.plot(x_vals_for_plotting, y_vals_for_plotting, z_vals_for_plotting, marker='o', color='r', label='Path of Optimization')
        ax3d.set_xlabel('x[0]')
        ax3d.set_ylabel('x[1]')
        ax3d.set_zlabel('f(x)')
        ax3d.set_title(f'3D Plot: Path of Optimization (Iteration {it})')
        ax3d.legend()

        # Draw and pause to update the plot
        plt.draw()
        plt.pause(0.1)  # Pause for 100ms to update the plot

    plt.show()

    # Return the optimal x found by BFGS
    return x

# Run the optimization and plot
x_opt = BFGS(f, [-1.2, 1], 100)
print("Optimal x:", x_opt)
