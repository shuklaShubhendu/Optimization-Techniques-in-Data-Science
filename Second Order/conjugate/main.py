import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def conjugate_gradient(A, b, x0, max_iter=100, tol=1e-8):
    """
    Conjugate Gradient Method for solving Ax = b.

    Parameters:
    A : ndarray
        Symmetric positive definite matrix.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray
        Initial guess for the solution.
    max_iter : int, optional
        Maximum number of iterations (default is 100).
    tol : float, optional
        Convergence tolerance (default is 1e-8).
    
    Returns:
    x_history : list
        List of solution vectors at each iteration.
    residual_history : list
        List of residual norms at each iteration.
    """
    x = x0  # Initial guess
    r = b - np.dot(A, x)  # Initial residual
    p = r  # Initial search direction
    r_old = np.dot(r, r)  # Initial dot product for residual
    x_history = [x.copy()]  # Store the solution vector at each iteration
    residual_history = [np.linalg.norm(r)]  # Store residual norm at each iteration

    k = 0  # Iteration counter

    # Iterative process
    while k < max_iter:
        # Compute step size alpha
        Ap = np.dot(A, p)
        alpha = r_old / np.dot(p, Ap)
        
        # Update the solution
        x = x + alpha * p
        x_history.append(x.copy())  # Store the solution vector
        
        # Update the residual
        r = r - alpha * Ap
        residual_history.append(np.linalg.norm(r))  # Store the residual norm
        
        # Compute new dot product for residual
        r_new = np.dot(r, r)
        
        # Check convergence
        if np.sqrt(r_new) < tol:
            break
        
        # Compute beta for updating the search direction
        beta = r_new / r_old
        
        # Update the search direction
        p = r + beta * p
        
        # Update residual dot product for next iteration
        r_old = r_new
        
        k += 1
    
    return np.array(x_history), residual_history

# Example usage:
A = np.array([[4, 1], [1, 3]])  # Symmetric positive definite matrix
b = np.array([1, 2])  # Right-hand side vector
x0 = np.zeros_like(b)  # Initial guess (zero vector)

# Run the Conjugate Gradient method for a maximum of 100 iterations
x_history, residual_history = conjugate_gradient(A, b, x0, max_iter=100)

# 2D Plot - Tracking the solution components over iterations
plt.figure(figsize=(10, 6))
plt.plot(x_history[:, 0], label="x1 component", marker='o')
plt.plot(x_history[:, 1], label="x2 component", marker='x')
plt.xlabel("Iteration")
plt.ylabel("Solution value")
plt.title("Conjugate Gradient: Solution components over iterations")
plt.legend()
plt.grid(True)
plt.show()

# 3D Plot - Tracking the residual norm over iterations
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
iterations = np.arange(len(residual_history))

ax.plot(iterations, residual_history, zs=0, zdir='y', label='Residual norm', color='b', marker='o')
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual")
ax.set_zlabel("Norm of Residual")
ax.set_title("Conjugate Gradient: Residual norm over iterations")
plt.show()
