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

# %%
