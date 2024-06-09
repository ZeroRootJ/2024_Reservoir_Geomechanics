import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ucs = [0, 155.5]
data_points = [ucs]

# Create a new plot
fig, ax = plt.subplots(figsize=(5, 5))

# For each data point, create a Circle (which represents the upper half of a circle) and add it to the plot
for point in data_points:
    circle = patches.Circle(((point[0]+point[1])/2, 0), radius=(point[1]-point[0])/2, fill=False)
    ax.add_patch(circle)

    sigma_2 = 75.95
    circle = patches.Circle(((point[0]+sigma_2)/2, 0), radius=(sigma_2-point[0])/2, fill=False)
    ax.add_patch(circle)
    circle = patches.Circle(((point[1]+sigma_2)/2, 0), radius=(point[1]-sigma_2)/2, fill=False)
    ax.add_patch(circle)


# Set the x and y limits of the plot to ensure all circles are visible
ax.set_xlim(-20, max([point[1] for point in data_points]))
ax.set_ylim(0, max([point[1] for point in data_points]))

# Define the slope and y-intercept
m = 1
c = 9.32

# Generate x values
x = np.linspace(-10, 200, 400)

# Calculate corresponding y values
y = m * x + c
plt.plot(x, y)
plt.xlabel('Normal Stress (MPa)')
plt.ylabel('Shear Stress (MPa)')
plt.title('Mohr diagram @ theta=90')
# Display the plot
plt.savefig('c.png')
plt.show()
