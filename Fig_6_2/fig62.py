import numpy as np
import matplotlib.pyplot as plt

# Define the range for x1 and x2 with a step size of 0.1
x1_min = 1  # x1=R/r
x1_max = 1.5  # setting a practical maximum value for x1
x2_min = 0  # x2=angle
x2_max = 360
Smax = 51.5
Smin = 90
Po = 31.5

step_size_x1 = 0.01  # Reduced step size for practicality
step_size_x2 = 0.1  # Reduced step size for practicality

# Generate values for x1 and x2 within the specified ranges
x1 = np.arange(x1_min, x1_max + step_size_x1, step_size_x1)
x2 = np.arange(x2_min, x2_max + step_size_x2, step_size_x2)

# Initialize the result matrix y
y = np.zeros((len(x1), len(x2)))

# Calculate the values of y for each combination of x1 and x2
for i in range(len(x1)):
    for j in range(len(x2)):
        term1 = 0.5 * (Smax + Smin - 2 * Po) * (1 + 1 / (x1[i] ** 2))
        term2 = 0.5 * (Smax - Smin) * (1 + 3 * 1 / (x1[i] ** 4)) * np.cos(np.radians(2 * x2[j]))
        y[i, j] = term1 - term2

# Convert polar coordinates to Cartesian coordinates for plotting
X2, X1 = np.meshgrid(np.deg2rad(x2), x1)  # Convert degrees to radians for x2
XX, YY = X1 * np.cos(X2), X1 * np.sin(X2)


def plot_a():
    plt.figure()
    plt.pcolor(XX, YY, y, shading='auto', cmap='jet', vmin=0, vmax=160)
    plt.colorbar(label='tangential stress (MPa)')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.title(f'Borehole Stability of Smax = {Smax}, Smin = {Smin}, Po = {Po}')
    plt.savefig('a.png')
    plt.show()


def plot_b():
    # Find indices where YY is approximately 0 and XX is between 1 and 1.5
    indices = np.where((np.isclose(YY, 0, atol=1e-8)) & (XX >= 1) & (XX <= 1.5))

    # Extract corresponding values from XX and y
    XX_values = XX[indices]
    y_values = y[indices]

    # Plot y versus XX when YY=0 and XX is between 1 and 1.5
    plt.figure()
    plt.grid(True)
    plt.plot(XX_values, y_values)
    plt.xlabel('Normalized radius (R/r)')
    plt.ylabel('tangential stress (MPa)')
    plt.ylim([0, 150])  # Set the y-axis limit
    plt.title('theta=0')
    plt.savefig('b.png')
    plt.show()


def plot_c():
    indices = np.where((np.isclose(XX, 0, atol=1e-8)) & (YY >= 1) & (YY <= 1.5))

    # Extract corresponding values from YY and y
    YY_values = YY[indices]
    y_values = y[indices]

    # Plot y versus YY when XX=0 and 1 <= YY <= 1.5
    plt.figure()
    plt.grid(True)
    plt.plot(YY_values, y_values)
    plt.xlabel('Normalized radius (R/r)')
    plt.ylabel('tangential stress (MPa)')
    plt.ylim([0, 150])  # Set the y-axis limit
    plt.title('theta=90')
    plt.savefig('c.png')
    plt.show()


plot_a()
