# %% Imports
from bsplyne import BSpline
import numpy as np
import matplotlib.pyplot as plt

# %% 1. Definition of the B-spline basis
# Quadratic (degree 2) B-spline defined on [0, 1]
degrees = [2]
knots = [np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])]

spline = BSpline(degrees, knots)

# %% 2. Control points: quarter circle approximation
# We approximate a quarter of a circle of radius R using 3 control points
R = 1.0

# Control points array shape: (physical_dimension, number_of_ctrl_points)
ctrl_pts = np.zeros((2, spline.getCtrlShape()[0]))

ctrl_pts[:, 0] = [R, 0.0]  # first control point
ctrl_pts[:, 1] = [R, R]  # second control point
ctrl_pts[:, 2] = [0.0, R]  # third control point

# %% 3. Quick visualization using the built-in plot method
# This is convenient for fast inspection
spline.plot(ctrl_pts)

# %% 4. Explicit curve evaluation using basis functions
# This approach exposes the mathematical structure of the B-spline evaluation
xi_eval = np.linspace(0.0, 1.0, 100)

# Basis function matrix (no derivatives)
Nmatrix = spline.DN([xi_eval], 0)

# Curve evaluation: C(xi) = N(xi) * P
Ceval = Nmatrix @ ctrl_pts.T

plt.figure()
plt.plot(Ceval[:, 0], Ceval[:, 1], "-b", label="B-spline curve")
plt.plot(ctrl_pts[0, :], ctrl_pts[1, :], "--or", label="Control points")
plt.axis("equal")
plt.legend()
plt.title("Explicit B-spline evaluation (N matrix)")
plt.show()

# %% 5. Direct curve evaluation using the BSpline call operator
# This is the recommended user-level API
Ceval2 = spline(ctrl_pts, [xi_eval]).T

plt.figure()
plt.plot(Ceval2[:, 0], Ceval2[:, 1], "-b", label="B-spline curve")
plt.plot(ctrl_pts[0, :], ctrl_pts[1, :], "--or", label="Control points")
plt.axis("equal")
plt.legend()
plt.title("Direct B-spline evaluation")
plt.show()

# %% 6. Refinement step 1: polynomial degree elevation
# Degree elevation preserves the geometry but increases the number of control points
ctrl_pts_elev = spline.orderElevation(ctrl_pts, [1])

spline.plot(ctrl_pts_elev)

print("After degree elevation:")
print("  Degree =", spline.getDegrees())
print("  Knot vector =", spline.getKnots())
print("  Control points:\n", ctrl_pts_elev)

# %% 7. Refinement step 2: knot insertion
# Insert a single internal knot.
# For a cubic spline (p = 3), this creates a C^2-continuous knot.
ctrl_pts_ref = spline.knotInsertion(ctrl_pts_elev, [np.array([0.5])])

spline.plot(ctrl_pts_ref)

print("After knot insertion:")
print("  Degree =", spline.getDegrees())
print("  Knot vector =", spline.getKnots())
print("  Control points:\n", ctrl_pts_ref)

# %% 8. Knot multiplicity increase
# At this stage, the spline is cubic (p = 3) and the knot xi = 0.5
# already has multiplicity 1 due to the previous knot insertion.
#
# Adding three more identical knots increases the total multiplicity
# to (p + 1) = 4, which enforces interpolation at xi = 0.5 of the
# coresponding control point.
ctrl_pts_interp = spline.knotInsertion(ctrl_pts_ref, [np.array([0.5, 0.5, 0.5])])

spline.plot(ctrl_pts_interp)

print("After knot multiplicity increase:")
print("  Degree =", spline.getDegrees())
print("  Knot vector =", spline.getKnots())
print("  Control points:\n", ctrl_pts_interp)

# %%
