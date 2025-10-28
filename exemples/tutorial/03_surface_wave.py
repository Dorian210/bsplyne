# %% Imports
from bsplyne import BSpline
import numpy as np
import matplotlib.pyplot as plt

# %% Create a 2D tensor-product B-spline basis (a surface)
degrees = [2, 2]
knots = [np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1])]
spline = BSpline(degrees, knots)

# Define a simple control grid (3×3) with a single elevated corner
x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 3)
X, Y = np.meshgrid(x, y, indexing='ij')
Z = np.zeros((3, 3))
Z[2, 2] = 1
ctrl_pts = np.array([X, Y, Z], dtype='float')

# Visualize the control grid
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, c='r')
plt.show()

# %% Evaluate and plot the B-spline surface defined by the control grid
XI = spline.linspace(100)
xyz = spline(ctrl_pts, XI)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, c='r')
ax.plot_surface(*xyz, linewidth=1, alpha=0.8)
plt.show()

# %% Elevate the polynomial degree along both axes
ctrl_pts = spline.orderElevation(ctrl_pts, [1, 1])
spline.plot(ctrl_pts)

# %% Insert 2 additional knots along both axes
knots_to_add = [np.array([0.33, 0.67]), 2] # can either provide the knots or the number of knots to add per element
ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)
spline.plot(ctrl_pts)

# %%