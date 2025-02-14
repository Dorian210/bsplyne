# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne.b_spline import BSpline


# %% Example of evaluating surface B-spline basis functions
degrees = [2, 2]
knots = [
    np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
    np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
]
spline = BSpline(degrees, knots)

xi = np.linspace(0, 1, 50)
eta = np.linspace(0, 1, 50)
Xi, Eta = np.meshgrid(xi, eta, indexing='ij')
XI = np.array([Xi.ravel(), Eta.ravel()])
basis_values = spline.DN(XI, k=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for basis_value in basis_values.toarray().T:
    ax.plot_surface(Xi, Eta, basis_value.reshape((50, 50)), alpha=0.5)
ax.set_title("Evaluation of Surface B-Spline Basis Functions")
ax.set_xlabel("Isoparameter ξ")
ax.set_ylabel("Isoparameter η")
ax.set_zlabel("Basis Functions")
plt.show()


# %% 2D Surface Example: 2D Isoparametric Space in a 2D Physical Space
ctrl_pts = np.array([
    [[1.00, 2.00, 3.00, 4.00], [1.67, 2.25, 2.75, 3.33], [1.00, 2.00, 3.00, 4.00], [0.00, 1.67, 3.33, 9.00]],
    [[0.00, 0.33, 0.33, 0.00], [1.00, 1.67, 1.67, 1.00], [2.00, 3.00, 3.00, 2.00], [3.00, 4.50, 4.50, 3.00]]
])

spline.plotMPL(ctrl_pts)
plt.title("2D Surface B-Spline")
plt.show()


# %% Example of order elevation in surface B-Spline
print("Control points before order elevation:")
print(ctrl_pts)

t = [1, 1]  # Increase the degree by 1 in each dimension
ctrl_pts = spline.orderElevation(ctrl_pts, t)

print("Control points after order elevation:")
print(ctrl_pts)

spline.plotMPL(ctrl_pts)
plt.title("Surface B-Spline after Order Elevation")
plt.show()


# %% Example of knot insertion in surface B-Spline
print("Control points before knot insertion:")
print(ctrl_pts)

knots_to_add = [np.array([0.25, 0.75]), np.array([])]  # Insert knots only in the first dimension
ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)

print("Control points after knot insertion:")
print(ctrl_pts)

spline.plotMPL(ctrl_pts)
plt.title("Surface B-Spline after Knot Insertion")
plt.show()


# %%
