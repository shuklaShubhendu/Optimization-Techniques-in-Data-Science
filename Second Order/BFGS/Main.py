import numpy as np
import matplotlib.pyplot as plt

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
    '''BFGS Quasi-Newton method with dynamic 2D visualization'''
    d = len(x0)
    nabla = grad(f, x0)
    H = np.eye(d)
    x = np.array(x0)
    it = 2

    # Store all iteration data for plotting
    x_vals = [x[0]]  # For x-axis plotting
    y_vals = [x[1]]  # For y-axis plotting

    # Generate a contour plot grid
    X, Y = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-1, 3, 400))
    Z = np.array([f([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Set up the plot
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    path_plot, = plt.plot([], [], 'ro-', label='Optimization Path')
    plt.xlabel('x[0]')
    plt.ylabel('x[1]')
    plt.title('Contour Plot of Rosenbrock Function with Optimization Path')
    plt.legend()
    plt.grid(True)
    plt.pause(1)

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
        x_vals.append(x[0])
        y_vals.append(x[1])

        # Update path plot dynamically
        path_plot.set_data(x_vals, y_vals)
        plt.scatter(x[0], x[1], color='blue')  # Highlight the current point
        plt.pause(0.5)  # Pause for visualization

    # plt.show()

    # Return the optimal x found by BFGS
    return x

# Run the optimization and visualize
x_opt = BFGS(f, [-1.2, 1], 100)
print("Optimal x:", x_opt)
