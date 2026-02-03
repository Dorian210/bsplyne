import numpy as np
import pytest
from bsplyne.b_spline_basis import BSplineBasis

@pytest.fixture
def quadratic_basis():
    """Create a standard quadratic B-spline basis for testing."""
    return BSplineBasis(2, [0., 0., 0., 1., 1., 1.])

def test_linspace(quadratic_basis):
    """Test generation of evenly spaced points in isoparametric space."""
    xi = quadratic_basis.linspace(5)
    np.testing.assert_array_almost_equal(
        xi, 
        np.array([0., 0.2, 0.4, 0.6, 0.8, 1.])
    )

def test_linspace_for_integration(quadratic_basis):
    """Test generation of integration points and weights in isoparametric space."""
    xi, dxi = quadratic_basis.linspace_for_integration(4)
    
    # Check points are within the isoparametric space bounds
    assert np.all(xi >= 0) and np.all(xi <= 1)
    
    # Check weights sum approximately to span length
    np.testing.assert_almost_equal(np.sum(dxi), 1.0)

def test_gauss_legendre_for_integration(quadratic_basis):
    """Test Gauss-Legendre quadrature points and weights in isoparametric space."""
    xi, dxi = quadratic_basis.gauss_legendre_for_integration(3)
    
    # Check points are within the isoparametric space bounds
    assert np.all(xi >= 0) and np.all(xi <= 1)
    
    # Check weights sum approximately to span length
    np.testing.assert_almost_equal(np.sum(dxi), 1.0)

def test_normalize_knots():
    """Test normalization of knot vector to unit interval."""
    basis = BSplineBasis(2, [0., 0., 0., 2., 2., 2.])
    basis.normalize_knots()
    
    np.testing.assert_array_equal(
        basis.knot,
        np.array([0., 0., 0., 1., 1., 1.])
    )
    assert basis.span == (0, 1)

def test_N(quadratic_basis):
    """Test evaluation of basis functions in isoparametric space."""
    # Test basis function values at specific points
    N = quadratic_basis.N(np.array([0., 0.5, 1.])).toarray()
    
    expected = np.array([
        [1.  , 0.  , 0.  ],
        [0.25, 0.5 , 0.25],
        [0.  , 0.  , 1.  ]
    ])
    np.testing.assert_array_almost_equal(N, expected)
    
    # Test first derivatives
    dN = quadratic_basis.N(np.array([0., 0.5, 1.]), k=1).toarray()
    expected_deriv = np.array([
        [-2. ,  2. ,  0. ],
        [-1. ,  0. ,  1. ],
        [ 0. , -2. ,  2. ]
    ])
    np.testing.assert_array_almost_equal(dN, expected_deriv)

def test_plotN(quadratic_basis):
    """Test plotting functionality without displaying."""
    # Test basic plotting (no display)
    quadratic_basis.plotN(show=False)
    
    # Test derivative plotting (no display)
    quadratic_basis.plotN(k=1, show=False)

def test_knotInsertion(quadratic_basis):
    """Test knot insertion in isoparametric space."""
    # Insert a knot at ξ = 0.5
    D = quadratic_basis.knotInsertion(np.array([0.5]))
    
    # Check new knot vector
    expected_knots = np.array([0., 0., 0., 0.5, 1., 1., 1.])
    np.testing.assert_array_almost_equal(quadratic_basis.knot, expected_knots)
    
    # Check transformation matrix
    D_array = D.toarray()
    assert D_array.shape == (4, 3)  # New control points vs old control points
    
    # Check partition of unity is preserved
    np.testing.assert_array_almost_equal(
        np.sum(D_array, axis=1),
        np.ones(D_array.shape[0])
    )

def test_orderElevation(quadratic_basis):
    """Test polynomial degree elevation in isoparametric space."""
    # Elevate from quadratic to cubic
    STD = quadratic_basis.orderElevation(1)
    
    # Check new degree
    assert quadratic_basis.p == 3
    
    # Check new knot vector
    expected_knots = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    np.testing.assert_array_almost_equal(quadratic_basis.knot, expected_knots)
    
    # Check transformation matrix
    STD_array = STD.toarray()
    assert STD_array.shape[0] > STD_array.shape[1]  # More control points after elevation
    
    # Check partition of unity is preserved
    np.testing.assert_array_almost_equal(
        np.sum(STD_array, axis=1),
        np.ones(STD_array.shape[0])
    )