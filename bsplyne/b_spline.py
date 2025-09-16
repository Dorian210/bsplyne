import os
from typing import Iterable, Literal, Union

import numpy as np
import scipy.sparse as sps
import meshio as io
import matplotlib as mpl
from scipy.interpolate import griddata

from bsplyne.b_spline_basis import BSplineBasis
from bsplyne.my_wide_product import my_wide_product
from bsplyne.save_utils import writePVD

class BSpline:
    """
    BSpline class for representing and manipulating B-spline curves, surfaces and volumes.

    A class providing functionality for evaluating, manipulating and visualizing B-splines of arbitrary dimension.
    Supports knot insertion, order elevation, and visualization through Paraview and Matplotlib.

    Attributes
    ----------
    NPa : int
        Dimension of the isoparametric space.
    bases : np.ndarray[BSplineBasis]
        Array containing `BSplineBasis` instances for each isoparametric dimension.

    Notes
    -----
    - Supports B-splines of arbitrary dimension (curves, surfaces, volumes, etc.)
    - Provides methods for evaluation, derivatives, refinement and visualization
    - Uses Cox-de Boor recursion formulas for efficient basis function evaluation
    - Visualization available through Paraview (VTK) and Matplotlib

    See Also
    --------
    `BSplineBasis` : Class representing one-dimensional B-spline basis functions
    `numpy.ndarray` : Array type used for control points and evaluations
    `scipy.sparse` : Sparse matrix formats used for basis function evaluations
    """
    NPa: int
    bases: np.ndarray[BSplineBasis]
    
    def __init__(self, degrees: Iterable[int], knots: Iterable[np.ndarray[np.floating]]):
        """
        Initialize a `BSpline` instance with specified degrees and knot vectors.

        Creates a `BSpline` object by generating basis functions for each isoparametric dimension
        using the provided polynomial degrees and knot vectors.

        Parameters
        ----------
        degrees : Iterable[int]
            Collection of polynomial degrees for each isoparametric dimension.
            The length determines the dimensionality of the parametric space (`NPa`).
            For example:
            - [p] for a curve
            - [p, q] for a surface
            - [p, q, r] for a volume
            - ...

        knots : Iterable[np.ndarray[np.floating]]
            Collection of knot vectors for each isoparametric dimension.
            Each knot vector must be a numpy array of `floats`.
            The number of knot vectors must match the number of degrees.
            For a degree `p`, the knot vector must have size `m + 1` where `m>=p`.

        Notes
        -----
        - The number of control points in each dimension will be `m - p` where `m` is
        the size of the knot vector minus 1 and `p` is the degree
        - Each knot vector must be non-decreasing
        - The multiplicity of each knot must not exceed `p + 1`

        Examples
        --------
        Create a 2D B-spline surface:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)

        Create a 1D B-spline curve:
        >>> degree = [3]
        >>> knot = [np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype='float')]
        >>> curve = BSpline(degree, knot)
        """
        self.NPa = len(degrees)
        self.bases = np.empty(self.NPa, dtype='object')
        for idx in range(self.NPa):
            p = degrees[idx]
            knot = knots[idx]
            self.bases[idx] = BSplineBasis(p, knot)
    
    @classmethod
    def from_bases(cls, bases: Iterable[BSplineBasis]) -> "BSpline":
        """
        Create a BSpline instance from an array of `BSplineBasis` objects.
        This is an alternative constructor that allows direct initialization from 
        existing basis functions rather than creating new ones from degrees and knot 
        vectors.

        Parameters
        ----------
        bases : Iterable[BSplineBasis]
            An iterable (e.g. list, tuple, array) containing `BSplineBasis` instances.
            Each basis represents one parametric dimension of the resulting B-spline.
            The number of bases determines the dimensionality of the parametric space.

        Returns
        -------
        BSpline
            A new `BSpline` instance with the provided basis functions.

        Notes
        -----
        - The method initializes a new `BSpline` instance with empty degrees and knots
        - The bases array is populated with the provided `BSplineBasis` objects
        - The dimensionality (`NPa`) is determined by the number of basis functions

        Examples
        --------
        >>> basis1 = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1]))
        >>> basis2 = BSplineBasis(2, np.array([0, 0, 0, 0.5, 1, 1, 1]))
        >>> spline = BSpline.from_bases([basis1, basis2])
        """
        self = cls([], [])
        self.NPa = len(bases)
        self.bases = np.empty(self.NPa, dtype='object')
        self.bases[:] = bases
        return self
    
    def getDegrees(self) -> np.ndarray[np.integer]:
        """
        Returns the polynomial degree of each basis function in the isoparametric space.

        Returns
        -------
        degrees : np.ndarray[np.integer]
            Array containing the polynomial degrees of the B-spline basis functions.
            The array has length `NPa` (dimension of isoparametric space), where each element
            represents the degree of the corresponding isoparametric dimension.

        Notes
        -----
        - For a curve (1D), returns [degree_xi]
        - For a surface (2D), returns [degree_xi, degree_eta]
        - For a volume (3D), returns [degree_xi, degree_eta, degree_zeta]
        - ...

        Examples
        --------
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getDegrees()
        array([2, 2])
        """
        degrees = np.array([basis.p for basis in self.bases], dtype='int')
        return degrees
    
    def getKnots(self) -> list[np.ndarray[np.floating]]:
        """
        Returns the knot vector of each basis function in the isoparametric space.

        This method collects all knot vectors from each `BSplineBasis` instance stored
        in the `bases` array. The knot vectors define the isoparametric space partitioning
        and the regularity properties of the B-spline.

        Returns
        -------
        knots : list[np.ndarray[np.floating]]
            List containing the knot vectors of the B-spline basis functions.
            The list has length `NPa` (dimension of isoparametric space), where each element
            is a `numpy.ndarray` containing the knots for the corresponding isoparametric dimension.

        Notes
        -----
        - For a curve (1D), returns [`knots_xi`]
        - For a surface (2D), returns [`knots_xi`, `knots_eta`]
        - For a volume (3D), returns [`knots_xi`, `knots_eta`, `knots_zeta`]
        - Each knot vector must be non-decreasing
        - The multiplicity of interior knots determines the continuity at that point

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getKnots()
        [array([0., 0., 0., 0.5, 1., 1., 1.]),
         array([0., 0., 0., 0.5, 1., 1., 1.])]
        """
        knots = [basis.knot for basis in self.bases]
        return knots

    def getNbFunc(self) -> int:
        """
        Compute the total number of basis functions in the B-spline.

        This method calculates the total number of basis functions by multiplying
        the number of basis functions in each isoparametric dimension (`n + 1` for each dimension).

        Returns
        -------
        int
            Total number of basis functions in the B-spline. This is equal to
            the product of (`n + 1`) for each basis, where `n` is the last index
            of each basis function.

        Notes
        -----
        - For a curve (1D), returns (`n + 1`)
        - For a surface (2D), returns (`n1 + 1`) × (`n2 + 1`)
        - For a volume (3D), returns (`n1 + 1`) × (`n2 + 1`) × (`n3 + 1`)
        - The number of basis functions equals the number of control points needed

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getNbFunc()
        16
        """
        return np.prod([basis.n + 1 for basis in self.bases])

    def getSpans(self) -> list[tuple[float, float]]:
        """
        Returns the span of each basis function in the isoparametric space.

        This method collects the spans (intervals of definition) from each `BSplineBasis`
        instance stored in the `bases` array.

        Returns
        -------
        spans : list[tuple[float, float]]
            List containing the spans of the B-spline basis functions.
            The list has length `NPa` (dimension of isoparametric space), where each element
            is a tuple (`a`, `b`) containing the lower and upper bounds of the span
            for the corresponding isoparametric dimension.

        Notes
        -----
        - For a curve (1D), returns [(`xi_min`, `xi_max`)]
        - For a surface (2D), returns [(`xi_min`, `xi_max`), (`eta_min`, `eta_max`)]
        - For a volume (3D), returns [(`xi_min`, `xi_max`), (`eta_min`, `eta_max`), (`zeta_min`, `zeta_max`)]
        - The span represents the interval where the B-spline is defined
        - Each span is determined by the `p`-th and `(m - p)`-th knots, where `p` is the degree
        and `m` is the last index of the knot vector

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getSpans()
        [(0.0, 1.0), (0.0, 1.0)]
        """
        spans = [basis.span for basis in self.bases]
        return spans
    
#     def get_indices(self, begining=0):
#         """
#         Create an array containing the indices of the control points of 
#         the B-spline.
# 
#         Parameters
#         ----------
#         begining : int, optional
#             First index of the arrayof indices, by default 0
# 
#         Returns
#         -------
#         indices : np.array of int
#             Indices of the control points in the same shape as the 
#             control points.
#         """
#         indices = np.arange(begining, begining + self.ctrl_pts.size).reshape(self.ctrl_pts.shape)
#         return indices

    def linspace(self, n_eval_per_elem: Union[int, Iterable[int]]=10) -> tuple[np.ndarray[np.floating], ...]:
        """
        Generate sets of evaluation points over the span of each basis in the isoparametric space.

        This method creates evenly spaced points for each isoparametric dimension by calling
        `linspace` on each `BSplineBasis` instance stored in the `bases` array.

        Parameters
        ----------
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            If an `int` is provided, the same number is used for all dimensions.
            If an `Iterable` is provided, each value corresponds to a different dimension.
            By default, 10.

        Returns
        -------
        XI : tuple[np.ndarray[np.floating], ...]
            Tuple containing arrays of evaluation points for each isoparametric dimension.
            The tuple has length `NPa` (dimension of isoparametric space).

        Notes
        -----
        - For a curve (1D), returns (`xi` points, )
        - For a surface (2D), returns (`xi` points, `eta` points)
        - For a volume (3D), returns (`xi` points, `eta` points, `zeta` points)
        - The number of points returned for each dimension depends on the number of
        elements in that dimension times the value of `n_eval_per_elem`

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> xi, eta = spline.linspace(n_eval_per_elem=2)
        >>> xi
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> eta
        array([0. , 0.5, 1. ])
        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa # type: ignore
        XI = tuple([basis.linspace(n) for basis, n in zip(self.bases, n_eval_per_elem)]) # type: ignore
        return XI

    def linspace_for_integration(
        self, 
        n_eval_per_elem: Union[int, Iterable[int]]=10, 
        bounding_box: Union[Iterable, None]=None
        ) -> tuple[tuple[np.ndarray[np.floating], ...], tuple[np.ndarray[np.floating], ...]]:
        """
        Generate sets of evaluation points and their integration weights over each basis span.

        This method creates evenly spaced points and their corresponding integration weights
        for each isoparametric dimension by calling `linspace_for_integration` on each
        `BSplineBasis` instance stored in the `bases` array.

        Parameters
        ----------
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            If an `int` is provided, the same number is used for all dimensions.
            If an `Iterable` is provided, each value corresponds to a different dimension.
            By default, 10.

        bounding_box : Union[Iterable[tuple[float, float]], None], optional
            Lower and upper bounds for each isoparametric dimension.
            If `None`, uses the span of each basis.
            Format: [(`xi_min`, `xi_max`), (`eta_min`, `eta_max`), ...].
            By default, None.

        Returns
        -------
        XI : tuple[np.ndarray[np.floating], ...]
            Tuple containing arrays of evaluation points for each isoparametric dimension.
            The tuple has length `NPa` (dimension of isoparametric space).

        dXI : tuple[np.ndarray[np.floating], ...]
            Tuple containing arrays of integration weights for each isoparametric dimension.
            The tuple has length `NPa` (dimension of isoparametric space).

        Notes
        -----
        - For a curve (1D), returns ((`xi` points), (`xi` weights))
        - For a surface (2D), returns ((`xi` points, `eta` points), (`xi` weights, `eta` weights))
        - For a volume (3D), returns ((`xi` points, `eta` points, `zeta` points), 
                                    (`xi` weights, `eta` weights, `zeta` weights))
        - The points are centered in their integration intervals
        - The weights represent the size of the integration intervals

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> (xi, eta), (dxi, deta) = spline.linspace_for_integration(n_eval_per_elem=2)
        >>> xi  # xi points
        array([0.125, 0.375, 0.625, 0.875])
        >>> dxi  # xi weights
        array([0.25, 0.25, 0.25, 0.25])
        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa # type: ignore
        if bounding_box is None:
            bounding_box = [b.span for b in self.bases] # type: ignore
        XI = []
        dXI = []
        for basis, (n, bb) in zip(self.bases, zip(n_eval_per_elem, bounding_box)): # type: ignore
            xi, dxi = basis.linspace_for_integration(n, bb)
            XI.append(xi)
            dXI.append(dxi)
        XI = tuple(XI)
        dXI = tuple(dXI)
        return XI, dXI
    
    def gauss_legendre_for_integration(
        self, 
        n_eval_per_elem: Union[int, Iterable[int], None]=None, 
        bounding_box: Union[Iterable, None]=None
        ) -> tuple[tuple[np.ndarray[np.floating], ...], tuple[np.ndarray[np.floating], ...]]:
        """
        Generate sets of evaluation points and their Gauss-Legendre integration weights over each basis span.

        This method creates Gauss-Legendre quadrature points and their corresponding integration weights
        for each isoparametric dimension by calling `gauss_legendre_for_integration` on each
        `BSplineBasis` instance stored in the `bases` array.

        Parameters
        ----------
        n_eval_per_elem : Union[int, Iterable[int], None], optional
            Number of evaluation points per element for each isoparametric dimension.
            If an `int` is provided, the same number is used for all dimensions.
            If an `Iterable` is provided, each value corresponds to a different dimension.
            If `None`, uses `(p + 2)//2` points per element where `p` is the degree of each basis.
            This number of points ensures an exact integration of a `p`-th degree polynomial.
            By default, None.

        bounding_box : Union[Iterable[tuple[float, float]], None], optional
            Lower and upper bounds for each isoparametric dimension.
            If `None`, uses the span of each basis.
            Format: [(`xi_min`, `xi_max`), (`eta_min`, `eta_max`), ...].
            By default, None.

        Returns
        -------
        XI : tuple[np.ndarray[np.floating], ...]
            Tuple containing arrays of Gauss-Legendre points for each isoparametric dimension.
            The tuple has length `NPa` (dimension of isoparametric space).

        dXI : tuple[np.ndarray[np.floating], ...]
            Tuple containing arrays of Gauss-Legendre weights for each isoparametric dimension.
            The tuple has length `NPa` (dimension of isoparametric space).

        Notes
        -----
        - For a curve (1D), returns ((`xi` points), (`xi` weights))
        - For a surface (2D), returns ((`xi` points, `eta` points), (`xi` weights, `eta` weights))
        - For a volume (3D), returns ((`xi` points, `eta` points, `zeta` points), 
                                    (`xi` weights, `eta` weights, `zeta` weights))
        - The points and weights follow the Gauss-Legendre quadrature rule
        - When `n_eval_per_elem` is `None`, uses `(p + 2)//2` points per element for exact
        integration of polynomials up to degree `p`

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> (xi, eta), (dxi, deta) = spline.gauss_legendre_for_integration()
        >>> xi  # xi points
        array([0.10566243, 0.39433757, 0.60566243, 0.89433757])
        >>> dxi  # xi weights
        array([0.25, 0.25, 0.25, 0.25])
        """
        if n_eval_per_elem is None:
            n_eval_per_elem = (self.getDegrees() + 2)//2
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa # type: ignore
        if bounding_box is None:
            bounding_box = [None]*self.NPa # type: ignore
        XI = []
        dXI = []
        for basis, (n_eval_per_elem_axis, bb) in zip(self.bases, zip(n_eval_per_elem, bounding_box)): # type: ignore
            xi, dxi = basis.gauss_legendre_for_integration(n_eval_per_elem_axis, bb)
            XI.append(xi)
            dXI.append(dxi)
        XI = tuple(XI)
        dXI = tuple(dXI)
        return XI, dXI
    
    def normalize_knots(self):
        """
        Maps all knot vectors to the interval [0, 1] in each isoparametric dimension.

        This method normalizes the knot vectors of each `BSplineBasis` instance stored
        in the `bases` array by applying an affine transformation that maps the span
        interval to [0, 1].

        Notes
        -----
        - The transformation preserves the relative spacing between knots
        - The transformation preserves the multiplicity of knots
        - The transformation is applied independently to each isoparametric dimension
        - This operation modifies the knot vectors in place

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([-1, -1, -1, 0, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 2, 4, 4, 4], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getKnots()
        [array([-1., -1., -1.,  0.,  1.,  1.,  1.]),
         array([0., 0., 0., 2., 4., 4., 4.])]
        >>> spline.normalize_knots()
        >>> spline.getKnots()
        [array([0., 0., 0., 0.5, 1., 1., 1.]),
         array([0., 0., 0., 0.5, 1., 1., 1.])]
        """
        for basis in self.bases:
            basis.normalize_knots()
    
    def DN(
        self, 
        XI: Union[np.ndarray[np.floating], tuple[np.ndarray[np.floating], ...]], 
        k: Union[int, Iterable[int]]=0
        ) -> Union[sps.spmatrix, np.ndarray[sps.spmatrix]]:
        """
        Compute the `k`-th derivative of the B-spline basis at given points in the isoparametric space.

        This method evaluates the basis functions or their derivatives at specified points, returning
        a matrix that can be used to evaluate the B-spline through a dot product with the control points.

        Parameters
        ----------
        XI : Union[np.ndarray[np.floating], tuple[np.ndarray[np.floating], ...]]
            Points in the isoparametric space where to evaluate the basis functions.
            Two input formats are accepted:
            1. `numpy.ndarray`: Array of coordinates with shape (`NPa`, n_points).
            Each column represents one evaluation point [`xi`, `eta`, ...].
            The resulting matrices will have shape (n_points, number of functions).
            2. `tuple`: Contains `NPa` arrays of coordinates (`xi`, `eta`, ...).
            The resulting matrices will have (n_xi × n_eta × ...) rows.

        k : Union[int, Iterable[int]], optional
            Derivative orders to compute. Two formats are accepted:
            1. `int`: Same derivative order along all axes. Common values:
            - `k=0`: Evaluate basis functions (default)
            - `k=1`: Compute first derivatives (gradient)
            - `k=2`: Compute second derivatives (hessian)
            2. `list[int]`: Different derivative orders for each axis.
            Example: `[1, 0]` computes first derivative w.r.t `xi`, no derivative w.r.t `eta`.
            By default, 0.

        Returns
        -------
        DN : Union[sps.spmatrix, np.ndarray[sps.spmatrix]]
            Sparse matrix or array of sparse matrices containing the basis evaluations:
            - If `k` is a `list` or is 0: Returns a single sparse matrix containing the mixed 
            derivative specified by the list.
            - If `k` is an `int` > 0: Returns an array of sparse matrices with shape [`NPa`]*`k`.
            For example, if `k=1`, returns `NPa` matrices containing derivatives along each axis.

        Notes
        -----
        - For evaluating the B-spline with control points in `NPh`-D space:
        `values = DN @ ctrl_pts.reshape((NPh, -1)).T`
        - When using tuple input format for `XI`, points are evaluated at all combinations of coordinates
        - When using array input format for `XI`, each column represents one evaluation point
        - The gradient (`k=1`) returns `NPa` matrices for derivatives along each axis
        - Mixed derivatives can be computed using a list of derivative orders

        Examples
        --------
        Create a 2D quadratic B-spline:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)

        Evaluate basis functions at specific points using array input:
        >>> XI = np.array([[0, 0.5, 1],     # xi coordinates
        ...                [0, 0.5, 1]])    # eta coordinates
        >>> N = spline.DN(XI, k=0)
        >>> N.A  # Convert sparse matrix to dense for display
        array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.25, 0.25, 0., 0., 0.25, 0.25, 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

        Compute first derivatives using tuple input:
        >>> xi = np.array([0, 0.5])
        >>> eta = np.array([0, 1])
        >>> dN = spline.DN((xi, eta), k=1)  # Returns [NPa] matrices
        >>> len(dN)  # Number of derivative matrices
        2
        >>> dN[0].A  # Derivative w.r.t xi
        array([[-4., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0., 0., 0.,-4., 0., 0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0., 0., 0., 0.,-2., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
               [ 0., 0., 0., 0., 0., 0., 0.,-2., 0., 0., 0., 2., 0., 0., 0., 0.]])
        >>> dN[1].A  # Derivative w.r.t eta
        array([[-4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0., 0.,-4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0., 0., 0., 0.,-2., 2., 0., 0.,-2., 2., 0., 0., 0., 0., 0., 0.],
               [ 0., 0., 0., 0., 0., 0.,-2., 2., 0., 0.,-2., 2., 0., 0., 0., 0.]])

        Compute mixed derivatives:
        >>> d2N = spline.DN((xi, eta), k=[1, 1])  # Second derivative: d²/dxi·deta
        >>> d2N.A
        array([[16.,-16.,  0.,  0.,-16.,16.,  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0.,  0., 16.,-16.,  0., 0.,-16.,16., 0., 0., 0., 0., 0., 0., 0., 0.],
               [ 0.,  0.,  0.,  0.,  8.,-8.,  0., 0.,-8., 8., 0., 0., 0., 0., 0., 0.],
               [ 0.,  0.,  0.,  0.,  0., 0.,  8.,-8., 0.,-0.,-8., 8., 0., 0., 0., 0.]])
        """
        
        if isinstance(XI, np.ndarray):
            fct = my_wide_product
            XI = XI.reshape((self.NPa, -1))
        else:
            fct = sps.kron
        
        if isinstance(k, int):
            if k==0:
                k = [0]*self.NPa # type: ignore
        
        if isinstance(k, int):
            dkbasis_dxik = np.empty((self.NPa, k + 1), dtype='object')
            for idx in range(self.NPa):
                basis = self.bases[idx]
                xi = XI[idx] - np.finfo('float').eps * (XI[idx]==basis.knot[-1])
                for k_querry in range(k + 1):
                    dkbasis_dxik[idx, k_querry] = basis.N(xi, k=k_querry)
            DN = np.empty([self.NPa]*k, dtype='object')
            dic = {}
            for axes in np.ndindex(*DN.shape):
                u, c = np.unique(axes, return_counts=True)
                k_arr = np.zeros(self.NPa, dtype='int')
                k_arr[u] = c
                key = tuple(k_arr)
                if key not in dic:
                    for idx in range(self.NPa):
                        k_querry = k_arr[idx]
                        if idx==0:
                            dic[key] = dkbasis_dxik[idx, k_querry]
                        else:
                            dic[key] = fct(dic[key], dkbasis_dxik[idx, k_querry])
                DN[axes] = dic[key]
            return DN
        else:
            for idx in range(self.NPa):
                basis = self.bases[idx]
                k_idx = k[idx]
                xi = XI[idx] - np.finfo('float').eps * (XI[idx]==basis.knot[-1])
                DN_elem = basis.N(xi, k=k_idx)
                if idx==0:
                    DN = DN_elem
                else:
                    DN = fct(DN, DN_elem)
            return DN
    
    def __call__(
        self, 
        ctrl_pts: np.ndarray[np.floating], 
        XI: Union[np.ndarray[np.floating], tuple[np.ndarray[np.floating], ...]], 
        k: Union[int, Iterable[int]]=0
        ) -> np.ndarray[np.floating]:
        """
        Evaluate the `k`-th derivative of the B-spline at given points in the isoparametric space.

        This method evaluates the B-spline or its derivatives at specified points by computing
        the basis functions and performing a dot product with the control points.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the B-spline geometry.
            Shape: (`NPh`, n1, n2, ...) where:
            - `NPh` is the dimension of the physical space
            - ni is the number of control points in the i-th isoparametric dimension, 
            i.e. the number of basis functions on this isoparametric axis

        XI : Union[np.ndarray[np.floating], tuple[np.ndarray[np.floating], ...]]
            Points in the isoparametric space where to evaluate the B-spline.
            Two input formats are accepted:
            1. `numpy.ndarray`: Array of coordinates with shape (`NPa`, n_points).
            Each column represents one evaluation point [`xi`, `eta`, ...].
            2. `tuple`: Contains `NPa` arrays of coordinates (`xi`, `eta`, ...).

        k : Union[int, Iterable[int]], optional
            Derivative orders to compute. Two formats are accepted:
            1. `int`: Same derivative order along all axes. Common values:
            - `k=0`: Evaluate the B-spline mapping (default)
            - `k=1`: Compute first derivatives (gradient)
            - `k=2`: Compute second derivatives (hessian)
            2. `list[int]`: Different derivative orders for each axis.
            Example: `[1, 0]` computes first derivative w.r.t `xi`, no derivative w.r.t `eta`.
            By default, 0.

        Returns
        -------
        values : np.ndarray[np.floating]
            B-spline evaluation at the specified points.
            Shape depends on input format and derivative order:
            - For array input: (`NPh`, shape of derivative, n_points)
            - For tuple input: (`NPh`, shape of derivative, n_xi, n_eta, ...)
            Where "shape of derivative" depends on `k`:
            - For `k=0` or `k` as list: Empty shape
            - For `k=1`: Shape is (`NPa`,) for gradient
            - For `k>1` as int: Shape is (`NPa`,) repeated `k` times

        Notes
        -----
        - The method first computes basis functions using `DN` then performs a dot product
        with control points
        - When using tuple input format, points are evaluated at all combinations of coordinates
        - When using array input format, each column represents one evaluation point
        - The gradient (`k=1`) returns derivatives along each isoparametric axis
        - Mixed derivatives can be computed using a list of derivative orders

        Examples
        --------
        Create a 2D quadratic B-spline:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 4)  # 3D control points

        Evaluate B-spline at specific points using array input:
        >>> XI = np.array([[0, 0.5, 1],    # xi coordinates
        ...                [0, 0.5, 1]])    # eta coordinates
        >>> values = spline(ctrl_pts, XI, k=0)
        >>> values.shape
        (3, 3)  # (NPh, n_points)

        Evaluate gradient using tuple input:
        >>> xi = np.array([0, 0.5])
        >>> eta = np.array([0, 1])
        >>> derivatives = spline(ctrl_pts, (xi, eta), k=1)
        >>> derivatives.shape
        (2, 3, 2, 2)  # (NPa, NPh, n_xi, n_eta)

        Compute mixed derivatives:
        >>> mixed = spline(ctrl_pts, (xi, eta), k=[1, 1])
        >>> mixed.shape
        (3, 2, 2)  # (NPh, n_xi, n_eta)
        """
        if isinstance(XI, np.ndarray):
            XI_shape = XI.shape[1:]
        else:
            XI_shape = [xi.size for xi in XI]
        DN = self.DN(XI, k)
        NPh = ctrl_pts.shape[0]
        if isinstance(DN, np.ndarray):
            values = np.empty((*DN.shape, NPh, *XI_shape), dtype='float')
            for axes in np.ndindex(*DN.shape):
                values[axes] = (DN[axes] @ ctrl_pts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        else:
            values = (DN @ ctrl_pts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        return values
    
    def knotInsertion(
        self, 
        ctrl_pts: np.ndarray[np.floating], 
        knots_to_add: Iterable[Union[np.ndarray[np.float64], int]]
        ) -> np.ndarray[np.floating]:
        """
        Add knots to the B-spline while preserving its geometry.

        This method performs knot insertion by adding new knots to each isoparametric dimension
        and computing the new control points to maintain the exact same geometry. The method
        modifies the `BSpline` object by updating its basis functions with the new knots.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the B-spline geometry.
            Shape: (`NPh`, n1, n2, ...) where:
            - `NPh` is the dimension of the physical space
            - ni is the number of control points in the i-th isoparametric dimension

        knots_to_add : Iterable[Union[np.ndarray[np.floating], int]]
            Refinement specification for each isoparametric dimension.
            For each dimension, two formats are accepted:
            1. `numpy.ndarray`: Array of knots to insert. These knots must lie within
            the span of the existing knot vector.
            2. `int`: Number of equally spaced knots to insert in each element.

        Returns
        -------
        new_ctrl_pts : np.ndarray[np.floating]
            New control points after knot insertion.
            Shape: (`NPh`, m1, m2, ...) where mi ≥ ni is the new number of
            control points in the i-th isoparametric dimension.

        Notes
        -----
        - Knot insertion preserves the geometry and parameterization of the B-spline
        - The number of new control points depends on the number and multiplicity of inserted knots
        - When using integer input, knots are inserted with uniform spacing in each element
        - The method modifies the basis functions but maintains `C^{p-m}` continuity,
        where `p` is the degree and `m` is the multiplicity of the inserted knot

        Examples
        --------
        Create a 2D quadratic B-spline and insert knots:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 4)  # 3D control points

        Insert specific knots in first dimension only:
        >>> knots_to_add = [np.array([0.25, 0.75], dtype='float'),
        ...                 np.array([], dtype='float')]
        >>> new_ctrl_pts = spline.knotInsertion(ctrl_pts, knots_to_add)
        >>> new_ctrl_pts.shape
        (3, 6, 4)  # Two new control points added in first dimension
        >>> spline.getKnots()[0]  # The knot vector is modified
        array([0.  , 0.  , 0.  , 0.25, 0.5 , 0.75, 1.  , 1.  , 1.  ])

        Insert two knots per element in both dimensions:
        >>> new_ctrl_pts = spline.knotInsertion(new_ctrl_pts, [1, 1])
        >>> new_ctrl_pts.shape
        (3, 10, 6)  # Uniform refinement in both dimensions
        >>> spline.getKnots()[0]  # The knot vectors are further modified
        array([0.   , 0.   , 0.   , 0.125, 0.25 , 0.375, 0.5  , 0.625, 0.75 ,
               0.875, 1.   , 1.   , 1.   ])
        """
        true_knots_to_add = []
        for axis, knots_to_add_elem in enumerate(knots_to_add):
            if isinstance(knots_to_add_elem, int): # It is a number of knots to add in each element
                u_knot = np.unique(self.bases[axis].knot)
                a, b = u_knot[:-1, None], u_knot[1:, None]
                mu = np.linspace(0, 1, knots_to_add_elem + 1, endpoint=False)[None, 1:]
                true_knots_to_add.append((a + (b - a)*mu).ravel())
            else:
                true_knots_to_add.append(knots_to_add_elem)
        knots_to_add = true_knots_to_add
        
        pts_shape = np.empty(self.NPa, dtype='int')
        D = None
        for idx in range(self.NPa):
            basis = self.bases[idx]
            knots_to_add_elem = knots_to_add[idx]
            D_elem = basis.knotInsertion(knots_to_add_elem)
            pts_shape[idx] = D_elem.shape[0]
            if D is None:
                D = D_elem
            else:
                D = sps.kron(D, D_elem)
        NPh = ctrl_pts.shape[0]
        pts = (D @ ctrl_pts.reshape((NPh, -1)).T).T
        ctrl_pts = pts.reshape((NPh, *pts_shape))
        return ctrl_pts
    
    def orderElevation(
        self, 
        ctrl_pts: np.ndarray[np.floating], 
        t: Iterable[int]
        ) -> np.ndarray[np.floating]:
        """
        Elevate the polynomial degree of the B-spline while preserving its geometry.

        This method performs order elevation by increasing the polynomial degree of each
        isoparametric dimension and computing the new control points to maintain the exact
        same geometry. The method modifies the `BSpline` object by updating its basis
        functions with the new degrees.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the B-spline geometry.
            Shape: (`NPh`, n1, n2, ...) where:
            - `NPh` is the dimension of the physical space
            - ni is the number of control points in the i-th isoparametric dimension

        t : Iterable[int]
            Degree elevation for each isoparametric dimension.
            For each dimension i, the new degree will be `p_i + t_i` where `p_i`
            is the current degree.

        Returns
        -------
        new_ctrl_pts : np.ndarray[np.floating]
            New control points after order elevation.
            Shape: (`NPh`, m1, m2, ...) where mi ≥ ni is the new number of
            control points in the i-th isoparametric dimension.

        Notes
        -----
        - Order elevation preserves the geometry and parameterization of the B-spline
        - The number of new control points depends on the current degree and number of 
        elements
        - The method modifies the `BSpline` object by updating its basis functions
        - This operation is more computationally expensive than knot insertion

        Examples
        --------
        Create a 2D quadratic B-spline and elevate its order:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 4)  # 3D control points

        Elevate order by 1 in first dimension only:
        >>> t = [1, 0]  # Increase degree by 1 in first dimension
        >>> new_ctrl_pts = spline.orderElevation(ctrl_pts, t)
        >>> new_ctrl_pts.shape
        (3, 6, 4)  # Two new control points added in first dimension (one per element)
        >>> spline.getDegrees()  # The degrees are modified
        array([3, 2])
        """
        pts_shape = np.empty(self.NPa, dtype='int')
        STD = None
        for idx in range(self.NPa):
            basis = self.bases[idx]
            t_basis = t[idx]
            STD_elem = basis.orderElevation(t_basis)
            pts_shape[idx] = STD_elem.shape[0]
            if STD is None:
                STD = STD_elem
            else:
                STD = sps.kron(STD, STD_elem)
        NPh = ctrl_pts.shape[0]
        pts = (STD @ ctrl_pts.reshape((NPh, -1)).T).T
        ctrl_pts = pts.reshape((NPh, *pts_shape))
        return ctrl_pts
    
    def greville_abscissa(
        self, 
        return_weights: bool=False
        ) -> Union[list[np.ndarray[np.floating]], tuple[list[np.ndarray[np.floating]], list[np.ndarray[np.floating]]]]:
        """
        Compute the Greville abscissa and optionally their weights for each isoparametric dimension.

        The Greville abscissa can be interpreted as the "position" of the control points in the 
        isoparametric space. They are often used as interpolation points for B-splines.

        Parameters
        ----------
        return_weights : bool, optional
            If `True`, also returns the weights (span lengths) of each basis function.
            By default, False.

        Returns
        -------
        greville : list[np.ndarray[np.floating]]
            List containing the Greville abscissa for each isoparametric dimension.
            The list has length `NPa`, where each element is an array of size `n + 1`,
            `n` being the last index of the basis functions in that dimension.

        weights : list[np.ndarray[np.floating]], optional
            Only returned if `return_weights` is `True`.
            List containing the weights for each isoparametric dimension.
            The list has length `NPa`, where each element is an array containing
            the span length of each basis function.

        Notes
        -----
        - For a curve (1D), returns [`xi` abscissa]
        - For a surface (2D), returns [`xi` abscissa, `eta` abscissa]
        - For a volume (3D), returns [`xi` abscissa, `eta` abscissa, `zeta` abscissa]
        - The Greville abscissa are computed as averages of `p` consecutive knots
        - The weights represent the size of the support of each basis function
        - The number of abscissa in each dimension equals the number of control points

        Examples
        --------
        Compute Greville abscissa for a 2D B-spline:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> greville = spline.greville_abscissa()
        >>> greville[0]  # xi coordinates
        array([0.  , 0.25, 0.75, 1.  ])
        >>> greville[1]  # eta coordinates
        array([0. , 0.5, 1. ])

        Compute both abscissa and weights:
        >>> greville, weights = spline.greville_abscissa(return_weights=True)
        >>> weights[0]  # weights for xi direction
        array([0.5, 1. , 1. , 0.5])
        """
        greville = []
        weights = []
        for idx in range(self.NPa):
            basis = self.bases[idx]
            p = basis.p
            knot = basis.knot
            greville.append(np.convolve(knot[1:-1], np.ones(p, dtype=int), 'valid')/p)
            weights.append(knot[(p+1):] - knot[:-(p+1)])
        if return_weights:
            return greville, weights
        return greville
    
    def make_control_poly_meshes(self, 
                                 ctrl_pts: np.ndarray[np.floating], 
                                 n_eval_per_elem: Union[int, Iterable[int]]=10, 
                                 n_step: int=1, 
                                 fields: dict={}, 
                                 XI: Union[None, tuple[np.ndarray[np.floating], ...]]=None, 
                                 paraview_sizes: dict={}) -> list[io.Mesh]:
        """
        Create meshes containing all the data needed to plot the control polygon of the B-spline.

        This method generates a list of `io.Mesh` objects representing the control mesh 
        (polygonal connectivity) of the B-spline, suitable for visualization (e.g. in Paraview). 
        It supports time-dependent fields and arbitrary dimension.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Array of control points of the B-spline, with shape 
            (`NPh`, number of elements for dim 1, ..., number of elements for dim `NPa`), 
            where `NPh` is the physical space dimension and `NPa` is the dimension of the 
            isoparametric space.
        n_step : int, optional
            Number of time steps to plot. By default, 1.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            By default, 10.
            - If an `int` is provided, the same number is used for all dimensions.
            - If an `Iterable` is provided, each value corresponds to a different dimension.
        n_step : int, optional
            Number of time steps to plot. By default, 1.
        fields : dict, optional
            Dictionary of fields to plot at each time step. Keys are field names. Values can be:
            - a `function` taking (`BSpline` spline, `tuple` of `np.ndarray[np.floating]` XI) and
            returning a `np.ndarray[np.floating]` of shape (`n_step`, number of combinations of XI, field size),
            - a `np.ndarray[np.floating]` defined **on the control points**, of shape (`n_step`, field size, *`ctrl_pts.shape[1:]`),
            which is then interpolated using the B-spline basis functions,
            - a `np.ndarray[np.floating]` defined **on the evaluation grid**, of shape (`n_step`, field size, *grid shape),
            where `grid shape` matches the discretization provided by XI or `n_eval_per_elem`.
            In this case, the field is interpolated in physical space using `scipy.interpolate.griddata`.
        XI : tuple[np.ndarray[np.floating], ...], optional
            Parametric coordinates at which to evaluate the B-spline and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, a regular grid is generated according to `n_eval_per_elem`.
        paraview_sizes: dict, optionnal
            The fields present in this `dict` are overrided by `np.NaN`s.
            The keys must be the fields names and the values must be the fields sizes for paraview.
            By default, {}.

        Returns
        -------
        list[io.Mesh]
            List of `io.Mesh` objects, one for each time step, containing the control mesh geometry 
            and associated fields.

        Notes
        -----
        - The control mesh is constructed by connecting control points along each isoparametric direction.
        - Fields can be provided either as functions evaluated at the Greville abscissae, or as arrays defined on the 
        control points or on a regular parametric grid (in which case they are interpolated at the Greville abscissae).
        - The first axis of the field array or function output corresponds to the time step, even if there is only one.
        - The method is compatible with B-splines of arbitrary dimension.

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 3)  # 3D control points for a 2D surface
        >>> meshes = spline.make_control_poly_meshes(ctrl_pts)
        >>> mesh = meshes[0]
        """
        if XI is None:
            XI = self.linspace(n_eval_per_elem)
        interp_points = self(ctrl_pts, XI).reshape((3, -1)).T
        shape = [xi.size for xi in XI]
        NXI = np.prod(shape)
        NPh = ctrl_pts.shape[0]
        lines = np.empty((0, 2), dtype='int')
        size = np.prod(ctrl_pts.shape[1:])
        inds = np.arange(size).reshape(ctrl_pts.shape[1:])
        for idx in range(self.NPa):
            rng = np.arange(inds.shape[idx])
            lines = np.append(lines, 
                              np.concatenate((np.expand_dims(np.take(inds, rng[ :-1], axis=idx), axis=-1), 
                                              np.expand_dims(np.take(inds, rng[1:  ], axis=idx), axis=-1)), 
                                             axis=-1).reshape((-1, 2)), 
                             axis=0)
        cells = {'line': lines}
        points = np.moveaxis(ctrl_pts, 0, -1).reshape((-1, NPh))
        greville = tuple(self.greville_abscissa())
        n = self.getNbFunc()
        point_data = {}
        for key, value in fields.items():
            if key in paraview_sizes:
                point_data[key] = np.full((n_step, n, paraview_sizes[key]), np.NAN)
            elif callable(value):
                point_data[key] = value(self, greville)
            else:
                value = np.asarray(value)
                if value.ndim>=2 and value.shape[-self.NPa:]==tuple(ctrl_pts.shape[1:]):
                    point_data[key] = value.reshape((n_step, -1, n)).transpose(0, 2, 1)
                elif value.ndim>=2 and value.shape[-self.NPa:]==tuple(shape):
                    paraview_size = value.shape[1]
                    interp_field = griddata(interp_points, value.reshape((n_step*paraview_size, NXI)).T, points, method='linear')
                    point_data[key] = interp_field.reshape((n, n_step, paraview_size)).transpose(1, 0, 2)
                else:
                    raise ValueError(f"Field {key} shape {value.shape} not understood.")
        # make meshes
        meshes = []
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            meshes.append(mesh)

        return meshes
    
    def make_elem_separator_meshes(self, 
                                   ctrl_pts: np.ndarray[np.floating], 
                                   n_eval_per_elem: Union[int, Iterable[int]]=10, 
                                   n_step: int=1, 
                                   fields: dict={}, 
                                   XI: Union[None, tuple[np.ndarray[np.floating], ...]]=None, 
                                   paraview_sizes: dict={}) -> list[io.Mesh]:
        """
        Create meshes representing the boundaries of every element in the B-spline for visualization.

        This method generates a list of `io.Mesh` objects containing the geometry and optional fields
        needed to plot the limits (borders) of all elements from the isoparametric space of the B-spline.
        Supports time-dependent fields and arbitrary dimension.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Array of control points of the B-spline, with shape
            (`NPh`, number of elements for dim 1, ..., number of elements for dim `NPa`),
            where `NPh` is the physical space dimension and `NPa` is the dimension of the
            isoparametric space.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            By default, 10.
            - If an `int` is provided, the same number is used for all dimensions.
            - If an `Iterable` is provided, each value corresponds to a different dimension.
        n_step : int, optional
            Number of time steps to plot. By default, 1.
        fields : dict, optional
            Dictionary of fields to plot at each time step. Keys are field names. Values can be:
            - a `function` taking (`BSpline` spline, `tuple` of `np.ndarray[np.floating]` XI) and
            returning a `np.ndarray[np.floating]` of shape (`n_step`, number of combinations of XI, field size),
            - a `np.ndarray[np.floating]` defined **on the control points**, of shape (`n_step`, field size, *`ctrl_pts.shape[1:]`),
            which is then interpolated using the B-spline basis functions,
            - a `np.ndarray[np.floating]` defined **on the evaluation grid**, of shape (`n_step`, field size, *grid shape),
            where `grid shape` matches the discretization provided by XI or `n_eval_per_elem`.
            In this case, the field is interpolated in physical space using `scipy.interpolate.griddata`.
        XI : tuple[np.ndarray[np.floating], ...], optional
            Parametric coordinates at which to evaluate the B-spline and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, a regular grid is generated according to `n_eval_per_elem`.
        paraview_sizes: dict, optionnal
            The fields present in this `dict` are overrided by `np.NaN`s.
            The keys must be the fields names and the values must be the fields sizes for paraview.
            By default, {}.

        Returns
        -------
        list[io.Mesh]
            List of `io.Mesh` objects, one for each time step, containing the element boundary geometry
            and associated fields.

        Notes
        -----
        - The element boundary mesh is constructed by connecting points along the unique knot values
        in each isoparametric direction, outlining the limits of each element.
        - Fields can be provided either as callable functions, as arrays defined on the control points,
        or as arrays already defined on a regular evaluation grid.
        - When fields are defined on a grid, they are interpolated in the physical space using
        `scipy.interpolate.griddata` with linear interpolation.
        - The first axis of the field array or function output corresponds to the time step, even if there is only one.
        - The method supports B-splines of arbitrary dimension.

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 3)  # 3D control points for a 2D surface
        >>> meshes = spline.make_elem_separator_meshes(ctrl_pts)
        >>> mesh = meshes[0]
        """
        if XI is None:
            XI = self.linspace(n_eval_per_elem)
        interp_points = self(ctrl_pts, XI).reshape((3, -1)).T
        shape = [xi.size for xi in XI]
        NXI = np.prod(shape)
        NPh = ctrl_pts.shape[0]
        knots_uniq = []
        shape_uniq = []
        for basis in self.bases:
            knot = basis.knot
            knot_uniq = np.unique(knot[np.logical_and(knot>=0, knot<=1)])
            knots_uniq.append(knot_uniq)
            shape_uniq.append(knot_uniq.size)
        points = None
        point_data = {key: None for key in fields}
        lines = None
        Size = 0
        n = self.getNbFunc()
        for idx in range(self.NPa):
            basis = self.bases[idx]
            knot_uniq = knots_uniq[idx]
            n_eval = n_eval_per_elem[idx]
            couples = np.concatenate((np.expand_dims(knot_uniq[ :-1], axis=0), 
                                      np.expand_dims(knot_uniq[1:  ], axis=0)), 
                                     axis=0).reshape((2, -1)).T
            for a, b in couples:
                lin =  np.linspace(a, b, n_eval)
                inner_XI = tuple(knots_uniq[:idx] + [lin] + knots_uniq[(idx+1):])
                inner_shape = shape_uniq[:idx] + [lin.size] + shape_uniq[(idx+1):]
                size = np.prod(inner_shape)
                N = self.DN(inner_XI, [0]*self.NPa) # type: ignore
                pts = N @ ctrl_pts.reshape((NPh, -1)).T
                points = pts if points is None else np.vstack((points, pts))
                for key, value in fields.items():
                    if key in paraview_sizes:
                        to_store = np.full((n_step, size, paraview_sizes[key]), np.NAN)
                    elif callable(value):
                        to_store = value(self, inner_XI)
                    else:
                        value = np.asarray(value)
                        if value.ndim>=2 and value.shape[-self.NPa:]==tuple(ctrl_pts.shape[1:]):
                            paraview_size = value.shape[1]
                            arr = value.reshape((n_step, paraview_size, n)).reshape((n_step*paraview_size, n))
                            to_store = (arr @ N.T).reshape((n_step, paraview_size, -1)).transpose(0, 2, 1)
                        elif value.ndim>=2 and value.shape[-self.NPa:]==tuple(shape):
                            paraview_size = value.shape[1]
                            interp_field = griddata(interp_points, value.reshape((n_step*paraview_size, NXI)).T, pts, method='linear')
                            to_store = interp_field.reshape((pts.shape[0], n_step, paraview_size)).transpose(1, 0, 2)
                        else:
                            raise ValueError(f"Field {key} shape {value.shape} not understood.")
                    if point_data[key] is None:
                        point_data[key] = to_store
                    else:
                        point_data[key] = np.concatenate((point_data[key], to_store), axis=1) # type: ignore
                lns = Size + np.arange(size).reshape(inner_shape)
                lns = np.moveaxis(lns, idx, 0).reshape((inner_shape[idx], -1))
                lns = np.concatenate((np.expand_dims(lns[ :-1], axis=-1), 
                                      np.expand_dims(lns[1:  ], axis=-1)), 
                                     axis=-1).reshape((-1, 2))
                lines = lns if lines is None else np.vstack((lines, lns))
                Size += size
        cells = {'line': lines}
        # make meshes
        meshes = []
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i] # type: ignore
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            meshes.append(mesh)
        
        return meshes
    
    def make_elements_interior_meshes(self, 
                                      ctrl_pts: np.ndarray[np.floating], 
                                      n_eval_per_elem: Union[int, Iterable[int]]=10, 
                                      n_step: int=1, 
                                      fields: dict={}, 
                                      XI: Union[None, tuple[np.ndarray[np.floating], ...]]=None) -> list[io.Mesh]:
        """
        Create meshes representing the interior of each element in the B-spline.

        This method generates a list of `io.Mesh` objects containing the geometry and optional fields
        for the interior of all elements, suitable for visualization (e.g., in ParaView). Supports
        time-dependent fields and arbitrary dimension.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Array of control points of the B-spline, with shape
            (`NPh`, number of points for dim 1, ..., number of points for dim `NPa`),
            where `NPh` is the physical space dimension and `NPa` is the dimension of
            the isoparametric space.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            By default, 10.
            - If an `int` is provided, the same number is used for all dimensions.
            - If an `Iterable` is provided, each value corresponds to a different dimension.
        n_step : int, optional
            Number of time steps to plot. By default, 1.
        fields : dict, optional
            Dictionary of fields to plot at each time step. Keys are field names. Values can be:
            - a `function` taking (`BSpline` spline, `tuple` of `np.ndarray[np.floating]` XI) and
            returning a `np.ndarray[np.floating]` of shape (`n_step`, number of combinations of XI, field size),
            - a `np.ndarray[np.floating]` defined **on the control points**, of shape (`n_step`, field size, *`ctrl_pts.shape[1:]`),
            in which case it is interpolated using the B-spline basis functions,
            - a `np.ndarray[np.floating]` defined **directly on the evaluation grid**, of shape (`n_step`, field size, *grid shape),
            where `grid shape` is the shape of the discretization XI (i.e., number of points along each parametric axis).
            By default, `{}` (no fields).
        XI : tuple[np.ndarray[np.floating], ...], optional
            Parametric coordinates at which to evaluate the B-spline and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, a regular grid is generated according to `n_eval_per_elem`.

        Returns
        -------
        list[io.Mesh]
            List of `io.Mesh` objects, one for each time step, containing the element interior geometry
            and associated fields.

        Notes
        -----
        - The interior mesh is constructed by evaluating the B-spline at a regular grid of points
        in the isoparametric space, with connectivity corresponding to lines (1D), quads (2D), or
        hexahedra (3D).
        - Fields can be provided either as arrays (on control points or on the discretization grid) or as functions.
        - Arrays given on control points are automatically interpolated using the B-spline basis functions.
        - Arrays already given on the evaluation grid are used directly without interpolation.
        - The first axis of the field array or function output must correspond to the time step, even if there is only one.
        - The method is compatible with B-splines of arbitrary dimension.

        Examples
        --------
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 3)  # 3D control points for a 2D surface
        >>> # Field given on control points (needs interpolation)
        >>> field_on_ctrl_pts = np.random.rand(1, 1, 4, 3)
        >>> # Field given directly on the evaluation grid (no interpolation)
        >>> field_on_grid = np.random.rand(1, 1, 10, 10)
        >>> meshes = spline.make_elements_interior_meshes(
        ...     ctrl_pts,
        ...     fields={'temperature': field_on_ctrl_pts, 'pressure': field_on_grid}
        ...     XI= # TODO
        ... )
        >>> mesh = meshes[0]
        """
        if XI is None:
            XI = self.linspace(n_eval_per_elem)
        # make points
        N = self.DN(XI)
        NPh = ctrl_pts.shape[0]
        points = N @ ctrl_pts.reshape((NPh, -1)).T
        # make connectivity
        elements = {2: "line", 4: "quad", 8: "hexahedron"}
        shape = [xi.size for xi in XI]
        NXI = np.prod(shape)
        inds = np.arange(NXI).reshape(shape)
        for idx in range(self.NPa):
            rng = np.arange(inds.shape[idx])
            inds = np.concatenate((np.expand_dims(np.take(inds, rng[ :-1], axis=idx), axis=-1), 
                                   np.expand_dims(np.take(inds, rng[1:  ], axis=idx), axis=-1)), axis=-1)
        if self.NPa>=2:
            for i in np.ndindex(*inds.shape[:-2]):
                inds[i] = [[inds[i+(0, 0)], inds[i+(0, 1)]], [inds[i+(1, 1)], inds[i+(1, 0)]]]
        inds = inds.reshape((*inds.shape[:self.NPa], -1))
        inds = inds.reshape((-1, inds.shape[-1]))
        cells = {elements[inds.shape[-1]]: inds}
        # make fields
        n = self.getNbFunc()
        point_data = {}
        for key, value in fields.items():
            if callable(value):
                point_data[key] = value(self, XI)
            else:
                value = np.asarray(value)
                if value.ndim>=2 and value.shape[-self.NPa:]==tuple(ctrl_pts.shape[1:]):
                    # Field given on control points: interpolate with basis functions
                    paraview_size = value.shape[1]
                    arr = value.reshape((n_step * paraview_size, n))
                    point_data[key] = (arr @ N.T).reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
                elif value.ndim>=2 and value.shape[-self.NPa:]==tuple(shape):
                    # Field already given on discretization grid
                    paraview_size = value.shape[1]
                    point_data[key] = value.reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
                else:
                    raise ValueError(f"Field {key} shape {value.shape} not understood.")
        # make meshes
        meshes = []
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step) # type: ignore
            meshes.append(mesh)
            
        return meshes
    
    def saveParaview(
        self, 
        ctrl_pts: np.ndarray[np.floating], 
        path: str, 
        name: str, 
        n_step: int=1, 
        n_eval_per_elem: Union[int, Iterable[int]]=10, 
        fields: Union[dict, None]=None, 
        XI: Union[None, tuple[np.ndarray[np.floating], ...]]=None, 
        groups: Union[dict[str, dict[str, Union[str, int]]], None]=None, 
        make_pvd: bool=True, 
        verbose: bool=True, 
        fields_on_interior_only: Union[bool, Literal['auto'], list[str]]='auto'
        ) -> dict[str, dict[str, Union[str, int]]]:
        """
        Save B-spline visualization data as Paraview files.

        This method creates three types of visualization files:
        - Interior mesh showing the B-spline surface/volume
        - Element borders showing the mesh structure
        - Control points mesh showing the control structure
        
        All files are saved in VTU format with an optional PVD file to group them.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the B-spline geometry.
            Shape: (`NPh`, n1, n2, ...) where:
            - `NPh` is the dimension of the physical space
            - ni is the number of control points in the i-th isoparametric dimension

        path : str
            Directory path where the PV files will be saved

        name : str
            Base name for the output files

        n_step : int, optional
            Number of time steps to save. By default, 1.

        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each isoparametric dimension.
            By default, 10.
            - If an `int` is provided, the same number is used for all dimensions.
            - If an `Iterable` is provided, each value corresponds to a different dimension.

        fields : Union[dict, None], optional
            Fields to visualize at each time step. Dictionary format:
            {
                "field_name": `field_value`
            }
            where `field_value` can be either:
            
            1. A numpy array with shape (`n_step`, `field_size`, `*ctrl_pts.shape[1:]`) where:
            - `n_step`: Number of time steps
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)
            - `*ctrl_pts.shape[1:]`: Same shape as control points (excluding `NPh`)
            
            2. A numpy array with shape (`n_step`, `field_size`, `*grid_shape`) where:
            - `n_step`: Number of time steps
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)
            - `*grid_shape`: Shape of the evaluation grid (number of points along each isoparametric axis)
            
            3. A function that computes field values (`np.ndarray[np.floating]`) at given 
            points from the `BSpline` instance and `XI`, the tuple of arrays containing evaluation 
            points for each dimension (`tuple[np.ndarray[np.floating], ...]`).
            The result should be an array of shape (`n_step`, `n_points`, `field_size`) where:
            - `n_step`: Number of time steps
            - `n_points`: Number of evaluation points (n_xi × n_eta × ...)
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)
            
            By default, None.
        
        XI : tuple[np.ndarray[np.floating], ...], optional
            Parametric coordinates at which to evaluate the B-spline and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, a regular grid is generated according to `n_eval_per_elem`.

        groups : Union[dict[str, dict[str, Union[str, int]]], None], optional
            Nested dictionary specifying file groups for PVD organization. Format:
            {
                "group_name": {
                    "ext": str,     # File extension (e.g., "vtu")
                    "npart": int,   # Number of parts in the group
                    "nstep": int    # Number of timesteps
                }
            }
            The method automatically creates/updates three groups:
            - "interior": For the B-spline surface/volume mesh
            - "elements_borders": For the element boundary mesh
            - "control_points": For the control point mesh
            
            If provided, existing groups are updated; if None, these groups are created.
            By default, None.

        make_pvd : bool, optional
            Whether to create a PVD file grouping all VTU files. By default, True.

        verbose : bool, optional
            Whether to print progress information. By default, True.

        fields_on_interior_only: Union[bool, Literal['auto'], list[str]], optionnal
            Whether to include fields only on the interior mesh (`True`), on all meshes (`False`),
            or on specified field names.
            If set to `'auto'`, fields named `'u'`, `'U'`, `'displacement'` or `'displ'` 
            are included on all meshes while others are only included on the interior mesh.
            By default, 'auto'.

        Returns
        -------
        groups : dict[str, dict[str, Union[str, int]]]
            Updated groups dictionary with information about saved files.

        Notes
        -----
        - Creates three types of VTU files for each time step:
            - {name}_interior_{part}_{step}.vtu
            - {name}_elements_borders_{part}_{step}.vtu
            - {name}_control_points_{part}_{step}.vtu
        - If `make_pvd=True`, creates a PVD file named {name}.pvd
        - Fields can be visualized as scalars or vectors in Paraview
        - The method supports time-dependent visualization through `n_step`

        Examples
        --------
        Save a 2D B-spline visualization:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 4, 4)  # 3D control points
        >>> spline.saveParaview(ctrl_pts, "./output", "bspline")

        Save with a custom field:
        >>> def displacement(spline, XI):
        ...     # Compute displacement field
        ...     return np.random.rand(1, np.prod([x.size for x in XI]), 3)
        >>> fields = {"displacement": displacement}
        >>> spline.saveParaview(ctrl_pts, "./output", "bspline", fields=fields)
        """
        
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if fields is None:
            fields = {}
        
        if groups is None:
            groups = {}
        
        interior = "interior"
        if interior in groups:
            groups[interior]["npart"] += 1
        else:
            groups[interior] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        elements_borders = "elements_borders"
        if elements_borders in groups:
            groups[elements_borders]["npart"] += 1
        else:
            groups[elements_borders] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        control_points = "control_points"
        if control_points in groups:
            groups[control_points]["npart"] += 1
        else:
            groups[control_points] = {"ext": "vtu", "npart": 1, "nstep": n_step}
        
        paraview_sizes = {}
        if fields_on_interior_only is True:
            for key, value in fields.items():
                if callable(value):
                    paraview_sizes[key] = value(self, np.zeros((self.NPa, 1))).shape[2]
                else:
                    paraview_sizes[key] = value.shape[1]
        elif fields_on_interior_only is False:
            pass
        elif fields_on_interior_only=='auto':
            for key, value in fields.items():
                if key not in ['u', 'U', 'displacement', 'displ']:
                    if callable(value):
                        paraview_sizes[key] = value(self, np.zeros((self.NPa, 1))).shape[2]
                    else:
                        paraview_sizes[key] = value.shape[1]
        else:
            for key in fields_on_interior_only:
                value = fields[key]
                if callable(value):
                    paraview_sizes[key] = value(self, np.zeros((self.NPa, 1))).shape[2]
                else:
                    paraview_sizes[key] = value.shape[1]
                
        meshes = self.make_elements_interior_meshes(ctrl_pts, n_eval_per_elem, n_step, fields, XI)
        prefix = os.path.join(path, f"{name}_{interior}_{groups[interior]['npart'] - 1}")
        for time_step, mesh in enumerate(meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(interior, "done")

        meshes = self.make_elem_separator_meshes(ctrl_pts, n_eval_per_elem, n_step, fields, XI, paraview_sizes)
        prefix = os.path.join(path, f"{name}_{elements_borders}_{groups[elements_borders]['npart'] - 1}")
        for time_step, mesh in enumerate(meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(elements_borders, "done")

        meshes = self.make_control_poly_meshes(ctrl_pts, n_eval_per_elem, n_step, fields, XI, paraview_sizes)
        prefix = os.path.join(path, f"{name}_{control_points}_{groups[control_points]['npart'] - 1}")
        for time_step, mesh in enumerate(meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(control_points, "done")
        
        if make_pvd:
            writePVD(os.path.join(path, name), groups)
        
        return groups
    
    def getGeomdl(self, ctrl_pts):
        try:
            from geomdl import BSpline as geomdlBS
        except:
            raise 
        if self.NPa==1:
            curve = geomdlBS.Curve()
            curve.degree = self.bases[0].p
            curve.ctrl_pts = ctrl_pts.T.tolist()
            curve.knotvector = self.bases[0].knot
            return curve
        elif self.NPa==2:
            surf = geomdlBS.Surface()
            surf.degree_u = self.bases[0].p
            surf.degree_v = self.bases[1].p
            surf.ctrl_pts2d = ctrl_pts.transpose((1, 2, 0)).tolist()
            surf.knotvector_u = self.bases[0].knot
            surf.knotvector_v = self.bases[1].knot
            return surf
        elif self.NPa==3:
            vol = geomdlBS.Volume()
            vol.degree_u = self.bases[0].p
            vol.degree_v = self.bases[1].p
            vol.degree_w = self.bases[2].p
            vol.cpsize = ctrl_pts.shape[1:]
            # ctrl_pts format (zeta, xi, eta)
            vol.ctrl_pts = ctrl_pts.transpose(3, 1, 2, 0).reshape((-1, ctrl_pts.shape[0])).tolist()
            vol.knotvector_u = self.bases[0].knot
            vol.knotvector_v = self.bases[1].knot
            vol.knotvector_w = self.bases[2].knot
            return vol
        else:
            raise NotImplementedError("Can only export curves, sufaces or volumes !")
    
    # def plotPV(self, ctrl_pts):
    #     pass
    
    def plotMPL(
        self, 
        ctrl_pts: np.ndarray[np.floating], 
        n_eval_per_elem: Union[int, Iterable[int]]=10, 
        ax: Union[mpl.axes.Axes, None]=None, 
        ctrl_color: str='#1b9e77', 
        interior_color: str='#7570b3', 
        elem_color: str='#666666', 
        border_color: str='#d95f02', 
        language: Union[Literal["english"], Literal["français"]]="english"
        ):
        """
        Plot the B-spline using Matplotlib.

        Creates a visualization of the B-spline geometry showing the control mesh,
        B-spline surface/curve, element borders, and patch borders. Supports plotting
        1D curves and 2D surfaces in 2D space, and 2D surfaces and 3D volumes in 3D space.

        Parameters
        ----------
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the B-spline geometry.
            Shape: (NPh, n1, n2, ...) where:
            - NPh is the dimension of the physical space (2 or 3)
            - ni is the number of control points in the i-th isoparametric dimension

        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for visualizing the B-spline.
            Can be specified as:
            - Single integer: Same number for all dimensions
            - Iterable of integers: Different numbers for each dimension
            By default, 10.

        ax : Union[mpl.axes.Axes, None], optional
            Matplotlib axes for plotting. If None, creates a new figure and axes.
            For 3D visualizations, must be a 3D axes if provided (created with 
            `projection='3d'`).
            Default is None (creates new axes).

        ctrl_color : str, optional
            Color for the control mesh visualization:
            - Applied to control points (markers)
            - Applied to control mesh lines
            Default is '#1b9e77' (green).

        interior_color : str, optional
            Color for the B-spline geometry:
            - For curves: Line color
            - For surfaces: Face color (with transparency)
            - For volumes: Face color of boundary surfaces (with transparency)
            Default is '#7570b3' (purple).

        elem_color : str, optional
            Color for element boundary visualization:
            - Shows internal mesh structure
            - Helps visualize knot locations
            Default is '#666666' (gray).

        border_color : str, optional
            Color for patch boundary visualization:
            - Outlines the entire B-spline patch
            - Helps distinguish patch edges
            Default is '#d95f02' (orange).
        
        language: str, optional
            Language for the plot labels. Can be 'english' or 'français'.
            Default is 'english'.

        Notes
        -----
        Visualization components:
        - Control mesh: Shows control points and their connections
        - B-spline: Shows the actual curve/surface/volume
        - Element borders: Shows the boundaries between elements
        - Patch borders: Shows the outer boundaries of the B-spline

        Supported configurations:
        - 1D B-spline in 2D space (curve)
        - 2D B-spline in 2D space (surface)
        - 2D B-spline in 3D space (surface)
        - 3D B-spline in 3D space (volume)

        For 3D visualization:
        - Surfaces are shown with transparency
        - Volume visualization shows the faces with transparency
        - View angle is automatically set for surfaces based on surface normal

        Examples
        --------
        Plot a 2D curve in 2D space:
        >>> degrees = [2]
        >>> knots = [np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(2, 3)  # 2D control points
        >>> spline.plotMPL(ctrl_pts)

        Plot a 2D surface in 3D space:
        >>> degrees = [2, 2]
        >>> knots = [np.array([0, 0, 0, 1, 1, 1], dtype='float'),
        ...          np.array([0, 0, 0, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> ctrl_pts = np.random.rand(3, 3, 3)  # 3D control points
        >>> spline.plotMPL(ctrl_pts)

        Plot on existing axes with custom colors:
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(projection='3d')
        >>> spline.plotMPL(ctrl_pts, ax=ax, ctrl_color='red', interior_color='blue')
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib import lines
        if language=="english":
            ctrl_mesh = "Control mesh"
            elems_bord = "Elements borders"
            b_spline = "B-spline"
            b_spline_patch = "B-spline patch"
            patch_bord = "Patch borders"
        elif language=="français":
            ctrl_mesh = "Maillage de contrôle"
            elems_bord = "Frontières inter-éléments"
            b_spline = "B-spline"
            b_spline_patch = "Patch B-spline"
            patch_bord = "Frontières inter-patchs"
        else:
            raise NotImplementedError(f"Can't understand language '{language}'. Try 'english' or 'français'.")
        NPh = ctrl_pts.shape[0]
        fig = plt.figure() if ax is None else ax.get_figure()
        if NPh==2:
            ax = fig.add_subplot() if ax is None else ax
            if self.NPa==1:
                ax.plot(ctrl_pts[0], ctrl_pts[1], marker="o", c=ctrl_color, label=ctrl_mesh, zorder=0)
                xi, = self.linspace(n_eval_per_elem=n_eval_per_elem)
                x, y = self.__call__(ctrl_pts, [xi])
                ax.plot(x, y, c=interior_color, label=b_spline, zorder=1)
                xi_elem, = self.linspace(n_eval_per_elem=1)
                x_elem, y_elem = self.__call__(ctrl_pts, [xi_elem])
                ax.scatter(x_elem, y_elem, marker='*', c=elem_color, label=elems_bord, zorder=2) # type: ignore
            elif self.NPa==2:
                xi, eta = self.linspace(n_eval_per_elem=n_eval_per_elem)
                xi_elem, eta_elem = self.linspace(n_eval_per_elem=1)
                x_xi, y_xi = self.__call__(ctrl_pts, [xi_elem, eta])
                x_eta, y_eta = self.__call__(ctrl_pts, [xi, eta_elem])
                x_pol = np.hstack((x_xi[ 0, :: 1], x_eta[:: 1, -1], x_xi[-1, ::-1], x_eta[::-1,  0]))
                y_pol = np.hstack((y_xi[ 0, :: 1], y_eta[:: 1, -1], y_xi[-1, ::-1], y_eta[::-1,  0]))
                xy_pol = np.hstack((x_pol[:, None], y_pol[:, None]))
                ax.add_patch(Polygon(xy_pol, fill=True, edgecolor=None, facecolor=interior_color, alpha=0.5, label=b_spline_patch, zorder=0)) # type: ignore
                ax.plot(ctrl_pts[0, 0, 0], ctrl_pts[1, 0, 0], marker="o", c=ctrl_color, label=ctrl_mesh, zorder=1, ms=plt.rcParams['lines.markersize']/np.sqrt(2))
                ax.add_collection(LineCollection(ctrl_pts.transpose(1, 2, 0), colors=ctrl_color, zorder=1)) # type: ignore
                ax.add_collection(LineCollection(ctrl_pts.transpose(2, 1, 0), colors=ctrl_color, zorder=1)) # type: ignore
                ax.scatter(ctrl_pts[0].ravel(), ctrl_pts[1].ravel(), marker="o", c=ctrl_color, zorder=1, s=0.5*plt.rcParams['lines.markersize']**2) # type: ignore
                ax.plot(x_xi[0, 0], y_xi[0, 0], linestyle="-", c=elem_color, label=elems_bord, zorder=2)
                ax.add_collection(LineCollection(np.array([x_xi, y_xi]).transpose(1, 2, 0)[1:-1], colors=elem_color, zorder=2)) # type: ignore
                ax.add_collection(LineCollection(np.array([x_eta, y_eta]).transpose(2, 1, 0)[1:-1], colors=elem_color, zorder=2)) # type: ignore
                ax.add_patch(Polygon(xy_pol, lw=1.25*plt.rcParams['lines.linewidth'], fill=False, edgecolor=border_color, label=patch_bord, zorder=2)) # type: ignore
            else:
                raise ValueError(f"Can't plot a {self.NPa}D shape in a 2D space.")
            ax.legend()
            ax.set_aspect(1)
        elif NPh==3:
            ax = fig.add_subplot(projection='3d') if ax is None else ax
            if self.NPa==1:
                pass
            elif self.NPa==2:
                xi, eta = self.linspace(n_eval_per_elem=n_eval_per_elem)
                xi_elem, eta_elem = self.linspace(n_eval_per_elem=1)
                x, y, z = self.__call__(ctrl_pts, [xi, eta])
                x_xi, y_xi, z_xi = self.__call__(ctrl_pts, [xi_elem, eta])
                x_eta, y_eta, z_eta = self.__call__(ctrl_pts, [xi, eta_elem])
                ax.plot_surface(x, y, z, rcount=1, ccount=1, edgecolor=None, facecolor=interior_color, alpha=0.5)
                ax.plot_wireframe(ctrl_pts[0], ctrl_pts[1], ctrl_pts[2], color=ctrl_color, zorder=2)
                ax.scatter(ctrl_pts[0], ctrl_pts[1], ctrl_pts[2], color=ctrl_color, zorder=2)
                ax.add_collection(Line3DCollection(np.array([x_xi, y_xi, z_xi]).transpose(1, 2, 0), colors=elem_color, zorder=1)) # type: ignore
                ax.add_collection(Line3DCollection(np.array([x_eta, y_eta, z_eta]).transpose(2, 1, 0), colors=elem_color, zorder=1)) # type: ignore
                # ax.plot_surface(x, y, z, rcount=1, ccount=1, edgecolor=border_color, facecolor=None, alpha=0)
                ctrl_handle = lines.Line2D([], [], color=ctrl_color, marker='o', linestyle='-', label=ctrl_mesh)
                elem_handle = lines.Line2D([], [], color=elem_color, linestyle='-', label=elems_bord)
                border_handle = lines.Line2D([], [], color=border_color, linestyle='-', label=patch_bord)
                ax.legend(handles=[ctrl_handle, elem_handle, border_handle])
                mid_param = [np.array([sum(self.bases[0].span)/2]), np.array([sum(self.bases[1].span)/2])]
                dxi, deta = self.__call__(ctrl_pts, mid_param, k=1)
                nx, ny, nz = np.cross(dxi.ravel(), deta.ravel())
                azim = np.degrees(np.arctan2(ny, nx))
                elev = np.degrees(np.arcsin(nz / np.sqrt(nx**2 + ny**2 + nz**2)))
                ax.view_init(elev=elev + 30, azim=azim)
            elif self.NPa==3:
                XI = self.linspace(n_eval_per_elem=n_eval_per_elem)
                XI_elem = self.linspace(n_eval_per_elem=1)
                for face in range(3):
                    for side in [-1, 0]:
                        XI_face = list(XI)
                        XI_face[face] = np.array([XI[face][side]])
                        X = np.squeeze(np.array(self.__call__(ctrl_pts, XI_face)))
                        ax.plot_surface(*X, rcount=1, ccount=1, edgecolor=None, color=interior_color, alpha=0.5)
                for axis in range(3):
                    ctrl_mesh_axis = np.rollaxis(ctrl_pts, axis + 1, 1).reshape((3, ctrl_pts.shape[axis + 1], -1))
                    ax.add_collection(Line3DCollection(ctrl_mesh_axis.transpose(2, 1, 0), colors=ctrl_color, zorder=2)) # type: ignore
                ax.scatter(*ctrl_pts.reshape((3, -1)), color=ctrl_color, zorder=2)
                for face in range(3):
                    for side in [-1, 0]:
                        for face_i, transpose in zip(sorted([(face + 1)%3, (face + 2)%3]), ((1, 2, 0), (2, 1, 0))):
                            XI_elem_border = list(XI)
                            XI_elem_border[face] = np.array([XI[face][side]])
                            XI_elem_border[face_i] = XI_elem[face_i]
                            X_elem_border = np.squeeze(np.array(self.__call__(ctrl_pts, XI_elem_border)))
                            ax.add_collection(Line3DCollection(X_elem_border.transpose(*transpose), colors=elem_color, zorder=1)) # type: ignore
                ctrl_handle = lines.Line2D([], [], color=ctrl_color, marker='o', linestyle='-', label=ctrl_mesh)
                elem_handle = lines.Line2D([], [], color=elem_color, linestyle='-', label=elems_bord)
                border_handle = lines.Line2D([], [], color=border_color, linestyle='-', label=patch_bord)
                ax.legend(handles=[ctrl_handle, elem_handle, border_handle])
            else:
                raise ValueError(f"Can't plot a {self.NPa}D shape in a 3D space.")
        else:
            raise ValueError(f"Can't plot in a {NPh}D space.")
