from bsplyne import BSplineBasis
import numpy as np
import matplotlib.pyplot as plt

# Objective circle radius
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

# Evaluation of basis functions
xi_eval = basis.linspace(100)
N = basis.N(xi_eval)

# B-spline formulation
bspline = N @ P

# NURBS weight
sq = np.sqrt(2) / 2
W = np.array([[1], [sq], [1], [sq], [1], [sq], [1], [sq], [1]])

# NURBS formulation
nurbs = N @ (P * W) / (N @ W)

# Plot against the real circle
plt.plot(bspline[:, 0], bspline[:, 1], label="B-spline")
plt.plot(nurbs[:, 0], nurbs[:, 1], lw=5, label="NURBS")
plt.plot(P[:, 0], P[:, 1], "--o", label="control mesh")
plt.plot(
    R * np.cos(2 * np.pi * xi_eval),
    R * np.sin(2 * np.pi * xi_eval),
    "k",
    label="real circle",
)
plt.gca().set_aspect("equal")
plt.legend()
plt.show()
