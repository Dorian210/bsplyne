# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne.b_spline import BSpline


# %% Example of evaluating linear B-spline basis functions
degrees = [2]
knots = [np.array([0, 0, 0, 0.5, 1, 1, 1])]
spline = BSpline(degrees, knots)

xi_vals = np.linspace(0, 1, 100)
basis_values = spline.DN(xi_vals.reshape((1, -1))) # the first argument is a numpy array with shape (1, 100)
print(f"The output of spline.DN is a sparse matrix with shape {basis_values.shape}: \n(number of evaluation points, number of basis functions)")

plt.plot(xi_vals, basis_values.toarray())
plt.title("Evaluation of Linear B-Spline Basis Functions")
plt.xlabel("Isoparameter ξ")
plt.ylabel("Basis Functions")
plt.grid()
plt.show()


# %% Linear example in 2D: B-Spline visualization
ctrl_pts = np.array([[0, 1, 2, 1], 
                     [1, 0, 1, 3]])
print(f"Control points array of size {ctrl_pts.shape}. First index for x or y, and second for the 4 control points.")
spline.plotMPL(ctrl_pts) # notably displays plt.plot(basis_values@ctrl_pts[0], basis_values@ctrl_pts[1])
plt.title("Linear B-Spline")
plt.show()


# %% Example of order elevation in linear B-Spline
print("Control points before order elevation:")
print(ctrl_pts)

t = [1]  # Increase the degree by 1
ctrl_pts = spline.orderElevation(ctrl_pts, t)

print("Control points after order elevation:")
print(ctrl_pts)

spline.plotMPL(ctrl_pts)
plt.title("Linear B-Spline after Order Elevation")
plt.show()


# %% Example of knot insertion in linear B-Spline
print("Control points before knot insertion:")
print(ctrl_pts)

knots_to_add = [np.array([0.25, 0.75])]
ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)

print("Control points after knot insertion:")
print(ctrl_pts)

spline.plotMPL(ctrl_pts)
plt.title("Linear B-Spline after Knot Insertion")
plt.show()


# %%
