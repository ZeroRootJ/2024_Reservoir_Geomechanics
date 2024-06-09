import numpy as np
import matplotlib.pyplot as plt

S1 = 67
S2 = 45
S3 = 70
pp = 32

S = np.array([[S1, 0, 0],
             [0, S2, 0],
             [0, 0, S3]], dtype=float)

step_deviation = 2
step_direction = 2

x1 = np.arange(0, 90, step_deviation)
x2 = np.arange(0, 360, step_direction)

y = np.zeros((len(x1), len(x2)))

for i in range(len(x1)):
    for j in range(len(x2)):

        well_deviation = x1[i]
        well_direction = x2[j]

        well_direction = np.radians(well_direction)
        well_deviation = np.radians(well_deviation)

        Rb = np.array([[-np.cos(well_direction)*np.cos(well_deviation), -np.sin(well_direction)*np.cos(well_deviation), np.sin(well_deviation)],
                       [np.sin(well_direction), -np.cos(well_direction), 0],
                       [np.cos(well_direction)*np.sin(well_deviation), np.sin(well_direction)*np.sin(well_deviation), np.cos(well_deviation)]])

        Sb = Rb @ S @ Rb.T

        sigma_theta_theta = lambda theta: Sb[0, 0] + Sb[1, 1] - 2*(Sb[0, 0] - Sb[1, 1])*np.cos(2*theta) - 4*Sb[0, 1]*np.sin(2*theta)
        tau_theta_z = lambda theta: 2*(-Sb[0, 2]*np.sin(theta) + Sb[1, 2]*np.cos(theta))
        sigma_zz = lambda theta: Sb[2, 2] - 2*0.25*(Sb[0, 0] - Sb[1, 1])*np.cos(2*theta) - Sb[0, 1]*np.sin(2*theta)

        tmax = lambda theta: 0.5 * (sigma_theta_theta(theta) + sigma_zz(theta) + np.sqrt((sigma_theta_theta(theta) - sigma_zz(theta))**2 + 4*tau_theta_z(theta)**2))
        tmin = lambda theta: 0.5 * (sigma_theta_theta(theta) + sigma_zz(theta) - np.sqrt((sigma_theta_theta(theta) - sigma_zz(theta))**2 + 4*tau_theta_z(theta)**2))

        y[i, j] = max([tmax(np.radians(theta)) for theta in range(0, 360, 1)])-2*pp


X2, X1 = np.meshgrid(np.deg2rad(x2), x1)  # Convert degrees to radians for x2
XX, YY = X1 * np.cos(X2), X1 * np.sin(X2)

plt.figure()
plt.pcolor(XX, YY, y, shading='auto', cmap='jet')
plt.colorbar(label='tangential stress (MPa)')
plt.xlabel('X')
plt.ylabel('Y')
# plt.title(f'Borehole Stability of Smax = {Smax}, Smin = {Smin}, Po = {Po}')
plt.savefig('Normal.png')
plt.show()
