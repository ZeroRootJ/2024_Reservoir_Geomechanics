import numpy as np

S1 = 67
S2 = 45
S3 = 70

S = np.array([[S1, 0, 0],
             [0, S2, 0],
             [0, 0, S3]], dtype=float)

well_deviation = 90
well_direction = 0

well_direction = np.radians(well_direction)
well_deviation = np.radians(well_deviation)

Rb = np.array([[-np.cos(well_direction)*np.cos(well_deviation), -np.sin(well_direction)*np.cos(well_deviation), np.sin(well_deviation)],
               [np.sin(well_direction), -np.cos(well_direction), 0],
               [np.cos(well_direction)*np.sin(well_deviation), np.sin(well_direction)*np.sin(well_deviation), np.cos(well_deviation)]])

Sb = Rb @ S @ Rb.T

pp = 32
friction_angle = 45
Smax = max(float(Sb[0, 0]), float(Sb[1, 1]))
Smin = min(float(Sb[0, 0]), float(Sb[1, 1]))

sigma_tan = Smax + Smin
sigma_rad = 0
sigma_z = Sb[2, 2]

sigma_1 = max(float(sigma_tan), float(sigma_rad), float(sigma_z))
sigma_3 = min(float(sigma_tan), float(sigma_rad), float(sigma_z))

# print(sigma_1, sigma_3)
#
# print(sigma_1 - sigma_3 * (1 + np.sin(np.radians(friction_angle))) /
#                               (1 - np.sin(np.radians(friction_angle))))

sigma_theta_theta = lambda theta: Sb[0, 0] + Sb[1, 1] - 2*(Sb[0, 0] - Sb[1, 1])*np.cos(2*theta) - 4*Sb[0, 1]*np.sin(2*theta)
tau_theta_z = lambda theta: 2*(-Sb[0, 2]*np.sin(theta) + Sb[1, 2]*np.cos(theta))
sigma_zz = lambda theta: Sb[2, 2] - 2*0.25*(Sb[0, 0] - Sb[1, 1])*np.cos(2*theta) - Sb[0, 1]*np.sin(2*theta)

theta=0
print(sigma_theta_theta(np.radians(theta)), tau_theta_z(np.radians(theta)), sigma_zz(np.radians(theta)))

tmax = lambda theta: 0.5 * (sigma_theta_theta(theta) + sigma_zz(theta) + np.sqrt((sigma_theta_theta(theta) - sigma_zz(theta))**2 + 4*tau_theta_z(theta)**2))
tmin = lambda theta: 0.5 * (sigma_theta_theta(theta) + sigma_zz(theta) - np.sqrt((sigma_theta_theta(theta) - sigma_zz(theta))**2 + 4*tau_theta_z(theta)**2))

print(tmax(np.radians(theta)), tmin(np.radians(theta)))
print(max([tmax(theta) for theta in range(0, 360, 1)]), min([tmin(theta) for theta in range(0, 360, 1)]))
#
# ucs = lambda theta: tmax(np.radians(theta)) - 5.82843 * tmin(np.radians(theta))
# # find largest value of ucs when theta is between 0 and 360
#
# tmaxmax = max([tmax(theta) for theta in range(0, 360, 1)])
# tminmin = min([tmin(theta) for theta in range(0, 360, 1)])


