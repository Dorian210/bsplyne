# %% Imports
from bsplyne import BSpline
import numpy as np

# %% Create a 1D B-spline curve from one basis and two sets of control points (x and y)
degrees = [2]
knots = [np.array([0, 0, 0, 1, 1, 1])]
spline = BSpline(degrees, knots)

# Random control points defining x and y coordinates
x = np.random.randn(spline.getNbFunc())
y = np.random.randn(spline.getNbFunc())
ctrl_pts = np.array([x, y], dtype='float')

# Plot the initial quadratic B-spline curve
spline.plot(ctrl_pts)

# %% Elevate the polynomial degree while keeping the same curve shape
# (B-spline order elevation preserves geometry but adds more control points)
ctrl_pts = spline.orderElevation(ctrl_pts, [1])
spline.plot(ctrl_pts)

# %% Insert additional knots in the basis (refinement without changing geometry)
knots_to_add = [np.array([0.33, 0.67])]
ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)
spline.plot(ctrl_pts)

# %%
