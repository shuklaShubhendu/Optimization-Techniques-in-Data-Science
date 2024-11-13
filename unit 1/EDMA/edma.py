# Values Used:
# - data: The input data for which EWMA will be calculated. Example: data = [10, 20, 30, 40, 50]
# - alpha: The smoothing factor, controlling the weight of the previous value. Example: alpha = 0.1
# - window: The number of periods for the EWMA calculation. Example: window = 5

import numpy as np
import matplotlib.pyplot as plt

# Calculate Exponential Weighted Moving Average (EWMA)
def ewma(data, alpha):
    ewma_values = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ewma_value = alpha * data[i] + (1 - alpha) * ewma_values[i - 1]
        ewma_values.append(ewma_value)
    return np.array(ewma_values)

# Sample data
data = np.array([10, 20, 30, 40, 50])  # Input data
alpha = 0.1  # Smoothing factor

# Calculate EWMA
ewma_values = ewma(data, alpha)

# Plot the data and its EWMA
plt.figure(figsize=(8, 6))
plt.plot(data, label='Original Data', marker='o', linestyle='-', color='blue')
plt.plot(ewma_values, label='EWMA', marker='x', linestyle='-', color='red')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Exponential Weighted Moving Average (EWMA)')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the function to calculate EWMA
def ewma(data, beta):
    # Initialize the EWMA list
    ewma_values = []
    ewma_previous = data[0]  # Start with the first data point as EWMA_1
    ewma_values.append(ewma_previous)
    
    # Calculate EWMA for each data point
    for t in range(1, len(data)):
        ewma_current = beta * data[t] + (1 - beta) * ewma_previous
        ewma_values.append(ewma_current)
        ewma_previous = ewma_current
    
    return np.array(ewma_values)

# Example data (time series data points)
data = np.array([10, 20, 15, 30, 25, 35, 30, 40, 45, 50])

# Define the smoothing factor (beta)
beta = 0.8

# Calculate the EWMA for the data
ewma_result = ewma(data, beta)

# Print the results
print("Original Data:", data)
print("EWMA Result:", ewma_result)

# Plotting the original data and the EWMA values
plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data", marker='o', linestyle='--', color='blue')
plt.plot(ewma_result, label="EWMA", marker='x', color='red', linewidth=2)
plt.title('Exponentially Weighted Moving Average (EWMA)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
