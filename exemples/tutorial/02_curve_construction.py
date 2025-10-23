# %% Imports
from bsplyne import BSpline
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

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

# %% Fit a semicircle with the created B-spline curve
def semicircle(xi):
    return np.cos(np.pi*xi), np.sin(np.pi*xi)

XI, dXI = spline.gauss_legendre_for_integration()
xi, = XI
dxi, = dXI

N = spline.DN(XI).tocsc()
A = N.T.multiply(dxi) @ N

x, y = semicircle(xi)
bx = N.T.multiply(dxi) @ x
by = N.T.multiply(dxi) @ y

ctrl_x = sps.linalg.spsolve(A, bx)
ctrl_y = sps.linalg.spsolve(A, by)

ctrl_pts = np.array([ctrl_x, ctrl_y], dtype='float')

ax = spline.plot(ctrl_pts, show=False)
ax.plot(*semicircle(np.linspace(0, 1, 100)), 'r-', lw=0.5, label='semicircle', zorder=100)
ax.legend()
plt.show()

# %% Save the B-spline curve to a file
spline.save('semicircle.json', ctrl_pts)

# %%
