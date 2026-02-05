# %% imports
from bsplyne import BSplineBasis
import numpy as np
import matplotlib.pyplot as plt

# %% Create a B-spline basis of functions of degree 2 with one element
degree = 2
knots = [0, 0, 0, 1, 1, 1]

basis = BSplineBasis(degree, knots)
basis.plotN()

# %% Use the orderElevation method to elevate the degree to 3
basis.orderElevation(1)
basis.plotN()

# %%Use the knotInsertion method to insert knots, turning it into a three elements basis
basis.knotInsertion(np.array([0.33, 0.67]))
basis.plotN()

# %% Take random values for the control points to create a B-spline curved line
xi = basis.greville_abscissa()
y = np.random.randn(basis.n + 1)

xi_hat = basis.linspace(100)
N = basis.N(xi_hat)
y_hat = N @ y
plt.plot(xi_hat, y_hat)
plt.scatter(xi, y)

# %% Compute the integral under the B-spline curve on the interval [0.2, 0.6]
xi, weights = basis.gauss_legendre_for_integration(bounding_box=(0.2, 0.6))
N = basis.N(xi)
integral = np.sum((N @ y) * weights)
print(integral)

# %% Plot the first and second derivatives of the basis functions
basis.plotN(k=1)
basis.plotN(k=2)

# %% Exercise: create an approximation of a circle and compute its radius by integration

# Target circle radius
R = 1

# Univariate basis
basis = BSplineBasis(1, [0, 0, 1, 1])
basis.knotInsertion(np.array([0.25, 0.5, 0.75]))
basis.orderElevation(1)
basis.plotN()

# control points
P = np.array(
    [[R, 0], [R, R], [0, R], [-R, R], [-R, 0], [-R, -R], [0, -R], [R, -R], [R, 0]]
)

# Evaluation of the basis functions
xi_eval = basis.linspace(100)
N = basis.N(xi_eval)

# B-spline curve formulation
bspline = N @ P

# Plot and compare with the exact circle
plt.plot(bspline[:, 0], bspline[:, 1], label="B-spline")
plt.plot(P[:, 0], P[:, 1], "--o", label="control mesh")
plt.plot(
    R * np.cos(2 * np.pi * xi_eval),
    R * np.sin(2 * np.pi * xi_eval),
    "k",
    label="exact circle",
)
plt.gca().set_aspect("equal")
plt.legend()
plt.show()

# Numerical integration of the curve length
xi_int, dxi_int = basis.gauss_legendre_for_integration(basis.p + 1)
dN_dxi = basis.N(xi_int, k=1)
dX_dxi = dN_dxi @ P
dx_dxi, dy_dxi = dX_dxi.T
integral = np.sum(np.sqrt(dx_dxi**2 + dy_dxi**2) * dxi_int)
print(f"Circle circumference : {2*np.pi*R:.3f}\nB-spline length : {integral:.3f}")
