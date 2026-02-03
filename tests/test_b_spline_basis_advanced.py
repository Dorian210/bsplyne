import numpy as np
import pytest
from bsplyne.b_spline_basis import BSplineBasis

@pytest.fixture
def cubic_basis():
    """Create a cubic B-spline basis with internal knots for testing."""
    return BSplineBasis(3, [0., 0., 0., 0., 0.3, 0.7, 1., 1., 1., 1.])

def test_integration_with_bounds(cubic_basis):
    """Test integration point generation with different boundary conditions."""
    # Test integration over subinterval
    xi, dxi = cubic_basis.gauss_legendre_for_integration(
        n_eval_per_elem=4, 
        bounding_box=(0.2, 0.8)
    )
    
    # Check points are within specified bounds
    assert np.all(xi >= 0.2) and np.all(xi <= 0.8)
    
    # Check weights sum to interval length
    np.testing.assert_almost_equal(np.sum(dxi), 0.6)
    
    # Test with different number of points per element
    xi_dense, dxi_dense = cubic_basis.gauss_legendre_for_integration(
        n_eval_per_elem=6,
        bounding_box=(0.2, 0.8)
    )
    
    # Should have more points but same total weight
    assert len(xi_dense) > len(xi)
    np.testing.assert_almost_equal(np.sum(dxi_dense), 0.6)

def test_multiple_knot_insertion(cubic_basis):
    """Test multiple knot insertions and their effects."""
    # Insert multiple knots at once
    knots_to_add = np.array([0.4, 0.5, 0.5, 0.6])  # Note: repeated knot
    D = cubic_basis.knotInsertion(knots_to_add)
    
    # Check multiplicity of knots
    unique_knots, counts = np.unique(cubic_basis.knot, return_counts=True)
    knot_dict = dict(zip(unique_knots, counts))
    
    assert knot_dict[0.5] == 2  # Multiplicity 2 at 0.5
    assert knot_dict[0.4] == 1  # Single knot at 0.4
    assert knot_dict[0.6] == 1  # Single knot at 0.6
    
    # Check transformation matrix properties
    D_array = D.toarray()
    
    # Matrix should have correct dimensions
    assert D_array.shape == (len(cubic_basis.knot) - cubic_basis.p - 1, 
                           len(cubic_basis.knot) - knots_to_add.size - cubic_basis.p - 1)
    
    # Check partition of unity
    np.testing.assert_array_almost_equal(
        np.sum(D_array, axis=1),
        np.ones(D_array.shape[0])
    )

def test_combined_operations(cubic_basis):
    """Test composition of multiple basis operations."""
    # First insert knots
    knots_to_add = np.array([0.45, 0.55])
    D1 = cubic_basis.knotInsertion(knots_to_add)
    
    # Then elevate order
    D2 = cubic_basis.orderElevation(1)
    
    # Finally normalize
    cubic_basis.normalize_knots()
    
    # Check final state
    assert cubic_basis.p == 4  # Degree increased by 1
    assert cubic_basis.span == (0, 1)  # Normalized span
    
    # Combined transformation matrix
    D_combined = D2 @ D1
    
    # Check partition of unity preservation through all operations
    np.testing.assert_array_almost_equal(
        np.sum(D_combined.toarray(), axis=1),
        np.ones(D_combined.shape[0])
    )

def test_basis_function_properties(cubic_basis):
    """Test mathematical properties of basis functions in isoparametric space."""
    # Test points including boundaries and internal points
    xi = np.array([0., 0.25, 0.5, 0.75, 1.])
    
    # Evaluate basis and derivatives
    N = cubic_basis.N(xi).toarray()
    dN = cubic_basis.N(xi, k=1).toarray()
    d2N = cubic_basis.N(xi, k=2).toarray()
    
    # Test partition of unity
    np.testing.assert_array_almost_equal(
        np.sum(N, axis=1),
        np.ones(len(xi))
    )
    
    # Test sum of derivatives equals zero
    np.testing.assert_array_almost_equal(
        np.sum(dN, axis=1),
        np.zeros(len(xi))
    )
    
    # Test second derivatives sum to zero
    np.testing.assert_array_almost_equal(
        np.sum(d2N, axis=1),
        np.zeros(len(xi))
    )

def test_gauss_legendre_integration(cubic_basis):
    """Test numerical integration using Gauss-Legendre quadrature in isoparametric space."""
    # Define a function to integrate: f(ξ) = ξ²(1-ξ)³
    def f(xi):
        return xi**2 * (1-xi)**3
    
    # Get Gauss-Legendre points and weights
    xi, dxi = cubic_basis.gauss_legendre_for_integration(n_eval_per_elem=4)
    
    # Evaluate basis functions at quadrature points
    N = cubic_basis.N(xi)
    
    # Define control points that represent f(ξ) in the B-spline space
    # These coefficients were pre-computed to approximate f(ξ) 
    # with a brute force constant weight least squares method :
    # from scipy.sparse.linalg import spsolve
    # xi = cubic_basis.linspace(n_eval_per_elem=1_000_000)
    # N, b = cubic_basis.N(xi), f(xi)
    # control_points = spsolve(N.T@N, N.T@b)
    control_points = np.array([-0.00026, 0.00188841, 0.05314124, 0.01401539, -0.00288727, 0.00087471])
    
    # Compute the integral using Gauss-Legendre quadrature
    f_spline = N @ control_points
    numerical_integral = np.sum(f_spline * dxi)
    
    # The exact integral of ξ²(1-ξ)³ from 0 to 1 is 1/60
    exact_integral = 1/60
    
    # Check the numerical integration accuracy
    np.testing.assert_almost_equal(
        numerical_integral,
        exact_integral,
        decimal=4,
        err_msg="Gauss-Legendre integration failed to match exact value"
    )
    
    # Test integration over a subinterval [0.2, 0.8]
    xi_sub, dxi_sub = cubic_basis.gauss_legendre_for_integration(
        n_eval_per_elem=4,
        bounding_box=(0.2, 0.8)
    )
    
    # Evaluate over subinterval
    N_sub = cubic_basis.N(xi_sub).toarray()
    f_spline_sub = N_sub @ control_points
    numerical_integral_sub = np.sum(f_spline_sub * dxi_sub)
    
    # Exact integral over [0.2, 0.8]
    exact_integral_sub = 0.01474
    
    # Check subinterval integration accuracy
    np.testing.assert_almost_equal(
        numerical_integral_sub,
        exact_integral_sub,
        decimal=4,
        err_msg="Subinterval Gauss-Legendre integration failed to match exact value"
    )