import numpy as np
import pytest
from bsplyne.b_spline import BSpline

@pytest.fixture
def cubic_basis_3d():
    degrees = [3, 3, 3]
    knots = [
        np.array([0., 0., 0., 0., 0.3, 0.7, 1., 1., 1., 1.], dtype=float),
        np.array([0., 0., 0., 0., 0.5, 1., 1., 1., 1.], dtype=float),
        np.array([0., 0., 0., 0., 0.4, 0.6, 1., 1., 1., 1.], dtype=float)
    ]
    return BSpline(degrees, knots)

@pytest.fixture
def control_points_3d():
    # 3D control points for a volumetric B-spline
    shape = (3, 6, 5, 6)  # (NPh, n1, n2, n3)
    return np.random.rand(*shape)

@pytest.fixture
def quadratic_surface():
    degrees = [2, 2]
    knots = [
        np.array([0., 0., 0., 0.5, 1., 1., 1.], dtype=float),
        np.array([0., 0., 0., 0.4, 0.7, 1., 1., 1.], dtype=float)
    ]
    return BSpline(degrees, knots)

@pytest.fixture
def control_points_surface():
    # 3D control points for a surface B-spline
    shape = (3, 4, 5)  # (NPh, n1, n2)
    return np.random.rand(*shape)

def test_integration_gauss_legendre(cubic_basis_3d, control_points_3d):
    # Define a test function f(ξ,η,ζ) = ξ²η³ζ in isoparametric space
    def test_function(xi, eta, zeta):
        return xi**2 * eta**3 * zeta
    
    # Get Gauss-Legendre points and weights
    (xi, eta, zeta), (dxi, deta, dzeta) = cubic_basis_3d.gauss_legendre_for_integration()
    
    # Evaluate basis functions at quadrature points
    N = cubic_basis_3d.DN((xi, eta, zeta))
    
    # Compute function values at quadrature points
    f_vals = test_function(xi[:, None, None], eta[None, :, None], zeta[None, None, :])
    
    # Compute numerical integral
    numerical_integral = np.sum(f_vals * dxi[:, None, None] * deta[None, :, None] * dzeta[None, None, :])
    
    # Exact integral over unit cube = 1/24
    exact_integral = 1/24
    
    np.testing.assert_almost_equal(numerical_integral, exact_integral, decimal=4)

def test_partition_of_unity(quadratic_surface):
    # Test partition of unity property in isoparametric space
    xi = np.linspace(0, 1, 10)
    eta = np.linspace(0, 1, 12)
    
    # Evaluate basis functions
    N = quadratic_surface.DN((xi, eta))
    
    # Sum of basis functions should be 1 everywhere
    basis_sum = np.array(N.sum(axis=1)).ravel()
    np.testing.assert_almost_equal(basis_sum, np.ones_like(basis_sum))

def test_derivative_properties(quadratic_surface, control_points_surface):
    # Test derivative properties in isoparametric space
    xi = np.array([0.3, 0.7])
    eta = np.array([0.2, 0.5, 0.8])
    
    # Get derivatives
    dN_dxi, dN_deta = quadratic_surface.DN((xi, eta), k=1)
    
    # Sum of first derivatives should be zero
    np.testing.assert_almost_equal(dN_dxi.sum(axis=1).A.ravel(), np.zeros(xi.size*eta.size))
    np.testing.assert_almost_equal(dN_deta.sum(axis=1).A.ravel(), np.zeros(xi.size*eta.size))

def test_mixed_derivatives(quadratic_surface, control_points_surface):
    # Test mixed derivatives computation in isoparametric space
    xi = np.array([0.25, 0.75])
    eta = np.array([0.25, 0.75])
    
    # Compute mixed derivative d²/dξdη
    d2N = quadratic_surface.DN((xi, eta), k=[1, 1])
    
    # Shape should match number of evaluation points
    expected_shape = (len(xi) * len(eta), quadratic_surface.getNbFunc())
    assert d2N.shape == expected_shape
    
    # Mixed derivatives should be continuous
    values = quadratic_surface(control_points_surface, (xi, eta), k=[1, 1])
    assert values.shape == (3, len(xi), len(eta))

def test_knot_insertion_invariance(quadratic_surface, control_points_surface):
    # Test geometric invariance under knot insertion in isoparametric space
    xi = np.linspace(0, 1, 20)
    eta = np.linspace(0, 1, 20)
    
    # Original geometry
    original = quadratic_surface(control_points_surface, (xi, eta))
    
    # Insert knots
    knots_to_add = [np.array([0.25, 0.75]), np.array([0.3, 0.6])]
    new_control_points = quadratic_surface.knotInsertion(control_points_surface, knots_to_add)
    
    # Refined geometry
    refined = quadratic_surface(new_control_points, (xi, eta))
    
    # Geometry should remain unchanged
    np.testing.assert_almost_equal(original, refined)

def test_degree_elevation_invariance(quadratic_surface, control_points_surface):
    # Test geometric invariance under degree elevation in isoparametric space
    xi = np.linspace(0, 1, 15)
    eta = np.linspace(0, 1, 15)
    
    # Original geometry
    original = quadratic_surface(control_points_surface, (xi, eta))
    
    # Elevate degrees
    new_control_points = quadratic_surface.orderElevation(control_points_surface, [1, 1])
    
    # Elevated geometry
    elevated = quadratic_surface(new_control_points, (xi, eta))
    
    # Geometry should remain unchanged
    np.testing.assert_almost_equal(original, elevated)