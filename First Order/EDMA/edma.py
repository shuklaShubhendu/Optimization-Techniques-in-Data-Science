import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Calculate Exponential Weighted Moving Average (EWMA)
def ewma(data, alpha):
    ewma_values = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ewma_value = alpha * data[i] + (1 - alpha) * ewma_values[i - 1]
        ewma_values.append(ewma_value)
    return np.array(ewma_values)

# Example data (time series data points)
data = np.array([10, 20, 15, 30, 25, 35, 30, 40, 45, 50])

# Define the smoothing factor (alpha)
alpha = 0.8

# Calculate the EWMA for the data
ewma_result = ewma(data, alpha)

# Print the results
print("Original Data:", data)
print("EWMA Result:", ewma_result)

# 2D Plotting the original data and the EWMA values
plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data", marker='o', linestyle='--', color='blue')
plt.plot(ewma_result, label="EWMA", marker='x', color='red', linewidth=2)
plt.title('Exponentially Weighted Moving Average (EWMA) - 2D')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 3D Plotting the original data and the EWMA values
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create an array for the time index (as the Z-axis)
time_index = np.arange(len(data))

# Plot original data in 3D (X: Time, Y: Data, Z: Constant Value for better visualization)
ax.scatter(time_index, data, zs=0, zdir='z', label='Original Data', color='blue', marker='o')

# Plot EWMA in 3D (X: Time, Y: EWMA, Z: Constant Value for better visualization)
ax.scatter(time_index, ewma_result, zs=1, zdir='z', label='EWMA', color='red', marker='x')

# Set labels and title
ax.set_title('Exponentially Weighted Moving Average (EWMA) - 3D')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_zlabel('EWMA')

# Add legend
ax.legend()

plt.show()
