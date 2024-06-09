import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator


# Define the range for x1 and x2 with a step size of 0.1
x1_min = 1  # x1=R/r
x1_max = 1.5  # setting a practical maximum value for x1
x2_min = 0  # x2=angle
x2_max = 360
Smax = 90
Smin = 51.5
Sv = 88.2
Po = 31.5
friction_angle = 45
poissons_ratio = 0.25

delta_p = 0

step_size_x1 = 0.01  # Reduced step size for practicality
step_size_x2 = 0.1  # Reduced step size for practicality

# Generate values for x1 and x2 within the specified ranges
x1 = np.arange(x1_min, x1_max + step_size_x1, step_size_x1)
x2 = np.arange(x2_min, x2_max + step_size_x2, step_size_x2)

# Initialize the result matrix y
sigma_tan = np.zeros((len(x1), len(x2)))
sigma_rad = np.zeros((len(x1), len(x2)))
sigma_z = np.zeros((len(x1), len(x2)))
required_UCS = np.zeros((len(x1), len(x2)))

# Calculate the values of y for each combination of x1 and x2
for i in range(len(x1)):
    for j in range(len(x2)):
        term1 = 0.5 * (Smax + Smin - 2 * Po) * (1 + 1 / (x1[i] ** 2))
        term2 = 0.5 * (Smax - Smin) * (1 + 3 * 1 / (x1[i] ** 4)) * np.cos(np.radians(2 * x2[j]))
        term3 = delta_p * (1 / (x1[i] ** 2))
        sigma_tan[i, j] = term1 - term2 - term3

        term1 = 0.5 * (Smax + Smin - 2 * Po) * (1 - 1 / (x1[i] ** 2))
        term2 = 0.5 * (Smax - Smin) * (1 - (4 * 1 / (x1[i] ** 2)) + (3 * 1 / (x1[i] ** 4))) * np.cos(np.radians(2 * x2[j]))
        sigma_rad[i, j] = term1 + term2 + term3

        sigma_z[i, j] = Sv - 2*poissons_ratio*(Smax-Smin)*(1 / (x1[i] ** 2)) * np.cos(np.radians(2 * x2[j])) - Po

        sigma_1 = max(float(sigma_tan[i, j]), float(sigma_rad[i, j]), float(sigma_z[i, j]))
        sigma_3 = min(float(sigma_tan[i, j]), float(sigma_rad[i, j]), float(sigma_z[i, j]))
        if sigma_tan[i, j] < 0:
            required_UCS[i, j] = np.NaN
        else:
            required_UCS[i, j] = (sigma_1 - sigma_3 * (1 + np.sin(np.radians(friction_angle))) /
                                  (1 - np.sin(np.radians(friction_angle))))

        # print(sigma_1, sigma_3, required_UCS[i, j])



# Convert polar coordinates to Cartesian coordinates for plotting
X2, X1 = np.meshgrid(np.deg2rad(x2), x1)  # Convert degrees to radians for x2
XX, YY = X1 * np.cos(X2), X1 * np.sin(X2)


def plot_a():
    plt.figure()
    plt.pcolor(XX, YY, required_UCS, shading='auto', cmap='jet', vmin=-100, vmax=150)
    plt.colorbar(label='Required UCS (MPa)')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Required UCS')
    plt.savefig('a.png')
    plt.show()


def plot_b():
    # Find indices where X1 is approximately 1
    indices = np.where(np.isclose(X1, 1, atol=1e-8))

    # Extract corresponding values from sigma_tan, sigma_rad, sigma_z, and X2
    sigma_tan_values = sigma_tan[indices]
    sigma_rad_values = sigma_rad[indices]
    sigma_z_values = sigma_z[indices]
    X2_values = np.degrees(X2[indices])  # Convert X2 from radians to degrees

    # Plot sigma_tan, sigma_rad, and sigma_z when X1 is approximately 1
    plt.figure()
    plt.grid(True)
    plt.plot(X2_values, sigma_tan_values, label='sigma_tan')
    plt.plot(X2_values, sigma_rad_values, label='sigma_rad')
    plt.plot(X2_values, sigma_z_values, label='sigma_z')
    plt.xlabel('theta (degrees)')
    plt.ylabel('Stress (MPa)')
    plt.legend()
    plt.savefig('b.png')
    plt.show()





plot_a()