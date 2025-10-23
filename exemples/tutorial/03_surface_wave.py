# %% Imports
from bsplyne import BSpline
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

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

# %% Evaluate and plot the B-spline surface defined by the control grid
XI = spline.linspace(100)
xyz = spline(ctrl_pts, XI)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, c='r')
ax.plot_surface(*xyz, linewidth=1, alpha=0.8)

# %% Elevate the polynomial degree along both axes
ctrl_pts = spline.orderElevation(ctrl_pts, [1, 1])
spline.plot(ctrl_pts)

# %% Insert 2 additional knots along both axes
knots_to_add = [np.array([0.33, 0.67]), 2] # can either provide the knots or the number of knots to add per element
ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)
spline.plot(ctrl_pts)

# %% Define a square-to-disk mapping function
def disk(xi, eta):
    """
    Map a square [0,1]^2 to the [-1,1]^2 disk.
    """
    xi = 2 * np.array(xi) - 1
    eta = 2 * np.array(eta) - 1
    r = np.maximum(np.abs(xi), np.abs(eta))
    r = r**0.75
    theta = np.arctan2(eta, xi)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

xi, eta = np.linspace(0, 1, 30), np.linspace(0, 1, 30)
XI, ETA = np.meshgrid(xi, eta, indexing='ij')
X, Y = disk(XI, ETA)
plt.plot(X, Y, 'k-', lw=0.5)
plt.plot(X.T, Y.T, 'k-', lw=0.5)
plt.gca().set_aspect('equal')

# %% Create a 2D B-spline surface and fit it to a disk using least squares
degrees = [1, 1]
knots = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
spline = BSpline(degrees, knots)
spline.orderElevation(None, [2, 2])
# spline.knotInsertion(None, [4, 4])

# Generate quadrature points and weights
XI, dXI = spline.gauss_legendre_for_integration([10, 10])
dXI = np.outer(dXI[0], dXI[1]).ravel()

# Compute the target disk coordinates
xi, eta = np.meshgrid(*XI, indexing='ij')
x_hat, y_hat = disk(xi, eta)

# Compute the LHS operators
N = spline.DN(XI).tocsc()
A = N.T.multiply(dXI) @ N

# Compute the RHS for the x and y coordinates problems
bx = N.T.multiply(dXI) @ x_hat.ravel()
by = N.T.multiply(dXI) @ y_hat.ravel()

# impose the position of unstable corners
inds = np.arange(spline.getNbFunc()).reshape(spline.getCtrlShape())
corners = np.array([inds[0, 0], inds[-1, 0], inds[-1, -1], inds[0, -1]])
angles = np.array([-3*np.pi/4, -np.pi/4, np.pi/4, 3*np.pi/4])
A[corners, :] = 0
A[:, corners] = 0
A[corners, corners] = 1
bx[corners] = np.cos(angles)
by[corners] = np.sin(angles)

# Solve for control points (weighted least squares fitting)
X = sps.linalg.spsolve(A, bx)
Y = sps.linalg.spsolve(A, by)

# Reshape and plot the fitted control net
new_ctrl_pts = np.array([X, Y], dtype='float').reshape((2, *spline.getCtrlShape()))
# ax = spline.plot(new_ctrl_pts, show=False)
fig, ax = plt.subplots()
x, y = spline(new_ctrl_pts, XI)
ax.plot(x, y, 'r-', lw=0.5)
ax.plot(x.T, y.T, 'r-', lw=0.5)
ax.plot(x_hat, y_hat, 'k-', lw=0.5)
ax.plot(x_hat.T, y_hat.T, 'k-', lw=0.5)
ax.set_aspect('equal')

# %% Let's do the same in 3D for a ball
def ball(xi, eta, zeta):
    """
    Map a cube [0,1]^3 to the [-1,1]^3 ball.
    """
    p = 50
    xi = 2*np.array(xi) - 1
    eta = 2*np.array(eta) - 1
    zeta = 2*np.array(zeta) - 1
    n_inf = np.maximum(np.maximum(np.abs(xi), np.abs(eta)), np.abs(zeta))
    npx = np.sqrt(xi**p + eta**2 + zeta**2)
    npy = np.sqrt(xi**2 + eta**p + zeta**2)
    npz = np.sqrt(xi**2 + eta**2 + zeta**p)
    xc = npx*np.tan(xi*np.arctan(1/np.where(npx==0, 1, npx)))
    yc = npy*np.tan(eta*np.arctan(1/np.where(npy==0, 1, npy)))
    zc = npz*np.tan(zeta*np.arctan(1/np.where(npz==0, 1, npz)))
    n = np.sqrt(xc**2 + yc**2 + zc**2)
    n = np.where(n==0, 1, n)
    x = n_inf*xc/n
    y = n_inf*yc/n
    z = n_inf*zc/n
    return x, y, z

n = 5
xi, eta, zeta = np.linspace(0, 1, n), np.linspace(0, 1, n), np.linspace(0, 1, n)
XI, ETA, ZETA = np.meshgrid(xi, eta, zeta, indexing='ij')
X, Y, Z = ball(XI, ETA, ZETA)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for k in range(0, n):
    ax.plot_wireframe(X[:, :, k], Y[:, :, k], Z[:, :, k], color='k', linewidth=0.5)
    ax.plot_wireframe(X[:, k, :], Y[:, k, :], Z[:, k, :], color='k', linewidth=0.5)
    ax.plot_wireframe(X[k, :, :], Y[k, :, :], Z[k, :, :], color='k', linewidth=0.5)
ax.set_box_aspect((1, 1, 1))
plt.show()

# %% Create a 3D B-spline volume and fit it to a ball using least squares
degrees = [2, 2, 2]
knots = [np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1])]
spline = BSpline(degrees, knots)
# spline.knotInsertion(None, [4, 4, 4])

# Generate quadrature points and weights
XI, dXI = spline.linspace_for_integration(10)
N = spline.DN(XI).tocsc()
dXI = np.outer(dXI[0], np.outer(dXI[1], dXI[2]).ravel()).ravel()

# Compute the target ball coordinates
xi, eta, zeta = np.meshgrid(*XI, indexing='ij')
x_hat, y_hat, z_hat = ball(xi.ravel(), eta.ravel(), zeta.ravel())

# Solve for control points (weighted least squares fitting)
X = sps.linalg.spsolve(N.T.multiply(dXI) @ N, N.T.multiply(dXI) @ x_hat)
Y = sps.linalg.spsolve(N.T.multiply(dXI) @ N, N.T.multiply(dXI) @ y_hat)
Z = sps.linalg.spsolve(N.T.multiply(dXI) @ N, N.T.multiply(dXI) @ z_hat)

# Reshape and plot the fitted control net
new_ctrl_pts = np.array([X, Y, Z], dtype='float').reshape((3, *spline.getCtrlShape()))
spline.plot(new_ctrl_pts)


# %%
