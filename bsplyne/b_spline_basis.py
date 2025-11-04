from typing import Iterable, Union
import json, pickle
import numpy as np
import numba as nb
import scipy.sparse as sps
from scipy.special import comb
import matplotlib.pyplot as plt


class BSplineBasis:
    """
    BSpline basis in 1D.

    A class representing a one-dimensional B-spline basis with functionality for evaluation,
    manipulation and visualization of basis functions. Provides methods for basis function
    evaluation, derivatives computation, knot insertion, order elevation, and integration
    point generation.

    Attributes
    ----------
    p : int
        Degree of the polynomials composing the basis.
    knot : np.ndarray[np.floating]
        Knot vector defining the B-spline basis. Contains non-decreasing sequence
        of isoparametric coordinates.
    m : int
        Last index of the knot vector (size - 1).
    n : int
        Last index of the basis functions. When evaluated, returns an array of size
        `n + 1`.
    span : tuple[float, float]
        Interval of definition of the basis `(knot[p], knot[m - p])`.

    Notes
    -----
    The basis functions are defined over the isoparametric space specified by the knot vector.
    Basis function evaluation and manipulation methods use efficient algorithms based on
    Cox-de Boor recursion formulas.

    See Also
    --------
    `numpy.ndarray` : Array type used for knot vector storage
    `scipy.sparse` : Sparse matrix formats used for basis function evaluations
    """

    p: int
    knot: np.ndarray[np.floating]
    m: int
    n: int
    span: tuple[float, float]

    def __init__(self, p: int, knot: Iterable[float]):
        """
        Initialize a B-spline basis with specified degree and knot vector.

        Parameters
        ----------
        p : int
            Degree of the B-spline polynomials.
        knot : Iterable[float]
            Knot vector defining the B-spline basis. Must be a non-decreasing sequence
            of real numbers.

        Returns
        -------
        BSplineBasis
            The initialized `BSplineBasis` instance.

        Notes
        -----
        The knot vector must satisfy these conditions:
        - Size must be at least `p + 2`
        - Must be non-decreasing
        - For non closed B-spline curves, first and last knots must have multiplicity `p + 1`

        The basis functions are defined over the isoparametric space specified by
        the knot vector. The span of the basis is [`knot[p]`, `knot[m - p]`], where
        `m` is the last index of the knot vector.

        Examples
        --------
        Create a quadratic B-spline basis with uniform knot vector:
        >>> basis = BSplineBasis(2, [0., 0., 0., 1., 1., 1.])
        """
        self.p = p
        self.knot = np.array(knot, dtype="float")
        self.m = self.knot.size - 1
        self.n = self.m - self.p - 1
        self.span = (self.knot[self.p], self.knot[self.m - self.p])

    def linspace(self, n_eval_per_elem: int = 10) -> np.ndarray[np.floating]:
        """
        Generate evenly spaced points over the basis span.

        Creates a set of evaluation points by distributing them uniformly within each knot span
        (element) of the basis. Points are evenly spaced within elements but spacing may vary
        between different elements.

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of evaluation points per element. By default, 10.

        Returns
        -------
        xi : np.ndarray[np.floating]
            Array of evenly spaced points in isoparametric coordinates over the basis span.

        Notes
        -----
        The method:
        1. Identifies unique knot spans (elements) in the isoparametric space
        2. Distributes points evenly within each element
        3. Combines points from all elements into a single array

        Examples
        --------
        >>> basis = BSplineBasis(2, [0., 0., 0., 1., 1., 1.])
        >>> basis.linspace(5)
        array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        """
        knot_uniq = np.unique(
            self.knot[
                np.logical_and(self.knot >= self.span[0], self.knot <= self.span[1])
            ]
        )
        xi = np.linspace(knot_uniq[-2], knot_uniq[-1], n_eval_per_elem + 1)
        for i in range(knot_uniq.size - 2, 0, -1):
            xi = np.append(
                np.linspace(
                    knot_uniq[i - 1], knot_uniq[i], n_eval_per_elem, endpoint=False
                ),
                xi,
            )
        return xi

    def linspace_for_integration(
        self,
        n_eval_per_elem: int = 10,
        bounding_box: Union[tuple[float, float], None] = None,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]:
        """
        Generate points and weights for numerical integration over knot spans in the
        isoparametric space. Points are evenly distributed within each element (knot span),
        though spacing may vary between different elements.

        Parameters
        ----------
        n_eval_per_elem : int, optional
            Number of evaluation points per element. By default, 10.
        bounding_box : Union[tuple[float, float], None], optional
            Lower and upper bounds for integration. If `None`, uses the span of the basis.
            By default, None.

        Returns
        -------
        xi : np.ndarray[np.floating]
            Array of integration points in isoparametric coordinates, evenly spaced
            within each element.
        dxi : np.ndarray[np.floating]
            Array of corresponding integration weights, which may vary between elements

        Notes
        -----
        The method generates integration points by:
        1. Identifying unique knot spans (elements) in the isoparametric space
        2. Distributing points evenly within each element
        3. Computing appropriate weights for each point based on the element size

        When `bounding_box` is provided, integration is restricted to that interval,
        and elements are adjusted accordingly.

        Examples
        --------
        >>> basis = BSplineBasis(2, [0, 0, 0, 1, 1, 1])
        >>> xi, dxi = basis.linspace_for_integration(5)
        """
        if bounding_box is None:
            lower, upper = self.span
        else:
            lower, upper = bounding_box
        knot_uniq = np.unique(
            self.knot[
                np.logical_and(self.knot >= self.span[0], self.knot <= self.span[1])
            ]
        )
        xi = []
        dxi = []
        for i in range(knot_uniq.size - 1):
            a = knot_uniq[i]
            b = knot_uniq[i + 1]
            if a < upper and b > lower:
                if a < lower and b > upper:
                    dxi_i_l = (upper - lower) / n_eval_per_elem
                    if (lower - 0.5 * dxi_i_l) < a:
                        dxi_i_u = (upper - a) / n_eval_per_elem
                        if (upper + 0.5 * dxi_i_u) > b:
                            dxi_i = (b - a) / n_eval_per_elem
                        else:
                            b = upper + 0.5 * dxi_i_u
                            dxi_i = dxi_i_u
                    else:
                        a = lower - 0.5 * dxi_i_l
                        dxi_i_u = dxi_i_l
                        if (upper + 0.5 * dxi_i_u) > b:
                            dxi_i = (b - lower) / n_eval_per_elem
                        else:
                            dxi_i = dxi_i_u
                            b = upper + 0.5 * dxi_i_u
                elif a < lower and b > lower:
                    dxi_i_l = (b - lower) / n_eval_per_elem
                    if (lower - 0.5 * dxi_i_l) < a:
                        dxi_i = (b - a) / n_eval_per_elem
                    else:
                        a = lower - 0.5 * dxi_i_l
                        dxi_i = dxi_i_l
                elif a < upper and b > upper:
                    dxi_i_u = (upper - a) / n_eval_per_elem
                    if (upper + 0.5 * dxi_i_u) > b:
                        dxi_i = (b - a) / n_eval_per_elem
                    else:
                        b = upper + 0.5 * dxi_i_u
                        dxi_i = dxi_i_u
                else:
                    dxi_i = (b - a) / n_eval_per_elem
                xi.append(
                    np.linspace(a + 0.5 * dxi_i, b - 0.5 * dxi_i, n_eval_per_elem)
                )
                dxi.append(dxi_i * np.ones(n_eval_per_elem))
        xi = np.hstack(xi)
        dxi = np.hstack(dxi)
        return xi, dxi

    def gauss_legendre_for_integration(
        self,
        n_eval_per_elem: Union[int, None] = None,
        bounding_box: Union[tuple[float, float], None] = None,
    ) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]:
        """
        Generate Gauss-Legendre quadrature points and weights for numerical integration over the B-spline basis.

        Parameters
        ----------
        n_eval_per_elem : Union[int, None], optional
            Number of evaluation points per element. If `None`, takes the value `self.p//2 + 1`.
            By default, None.
        bounding_box : Union[tuple[float, float], None], optional
            Lower and upper bounds for integration. If `None`, uses the span of the basis.
            By default, None.

        Returns
        -------
        xi : np.ndarray[np.floating]
            Array of Gauss-Legendre quadrature points in isoparametric coordinates.
        dxi : np.ndarray[np.floating]
            Array of corresponding integration weights.

        Notes
        -----
        The method generates integration points and weights by:
        1. Identifying unique knot spans (elements) in the isoparametric space
        2. Computing Gauss-Legendre points and weights for each element
        3. Transforming points and weights to account for element size

        When `bounding_box` is provided, integration is restricted to that interval.

        Examples
        --------
        >>> basis = BSplineBasis(2, [0, 0, 0, 1, 1, 1])
        >>> xi, dxi = basis.gauss_legendre_for_integration(3)
        >>> xi  # Gauss-Legendre points
        array([0.11270167, 0.5       , 0.88729833])
        >>> dxi  # Integration weights
        array([0.27777778, 0.44444444, 0.27777778])
        """
        if n_eval_per_elem is None:
            n_eval_per_elem = self.p // 2 + 1
        if bounding_box is None:
            lower, upper = self.span
        else:
            lower, upper = bounding_box
        knot_uniq = np.hstack(
            (
                [lower],
                np.unique(
                    self.knot[np.logical_and(self.knot > lower, self.knot < upper)]
                ),
                [upper],
            )
        )
        points, wheights = np.polynomial.legendre.leggauss(n_eval_per_elem)
        xi = np.hstack(
            [
                (b - a) / 2 * points + (b + a) / 2
                for a, b in zip(knot_uniq[:-1], knot_uniq[1:])
            ]
        )
        dxi = np.hstack(
            [(b - a) / 2 * wheights for a, b in zip(knot_uniq[:-1], knot_uniq[1:])]
        )
        return xi, dxi

    def normalize_knots(self):
        """
        Normalize the knot vector to the interval [0, 1].

        Maps the knot vector to the unit interval by applying an affine transformation that
        preserves the relative spacing between knots. Updates both the knot vector and span
        attributes.

        Examples
        --------
        >>> basis = BSplineBasis(2, [0., 0., 0., 2., 2., 2.])
        >>> basis.normalize_knots()
        >>> basis.knot
        array([0., 0., 0., 1., 1., 1.])
        >>> basis.span
        (0, 1)
        """
        a, b = self.span
        self.knot = (self.knot - a) / (b - a)
        self.span = (0, 1)

    def N(self, XI: np.ndarray[np.floating], k: int = 0) -> sps.coo_matrix:
        """
        Compute the k-th derivative of the B-spline basis functions at specified points.

        Parameters
        ----------
        XI : np.ndarray[np.floating]
            Points in the isoparametric space at which to evaluate the basis functions.
        k : int, optional
            Order of the derivative to compute. By default, 0.

        Returns
        -------
        DN : sps.coo_matrix
            Sparse matrix containing the k-th derivative values. Each row corresponds to an
            evaluation point, each column to a basis function. Shape is (`XI.size`, `n + 1`).

        Notes
        -----
        Uses Cox-de Boor recursion formulas to compute basis function derivatives.
        Returns values in sparse matrix format for efficient storage and computation.

        Examples
        --------
        >>> basis = BSplineBasis(2, [0., 0., 0., 1., 1., 1.])
        >>> basis.N([0., 0.5, 1.]).A  # Evaluate basis functions
        array([[1.  , 0.  , 0.  ],
            [0.25, 0.5 , 0.25],
            [0.  , 0.  , 1.  ]])
        >>> basis.N([0., 0.5, 1.], k=1).A  # Evaluate first derivatives
        array([[-2.,  2.,  0.],
            [-1.,  0.,  1.],
            [ 0., -2.,  2.]])
        """
        vals, row, col = _DN(
            self.p, self.m, self.n, self.knot, np.asarray(XI, dtype=np.float64), k
        )
        DN = sps.coo_matrix((vals, (row, col)), shape=(XI.size, self.n + 1))
        return DN

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the BSplineBasis object.
        """
        return {
            "p": self.p,
            "knot": self.knot.tolist(),
            "m": self.m,
            "n": self.n,
            "span": self.span,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BSplineBasis":
        """
        Creates a BSplineBasis object from a dictionary representation.
        """
        this = cls(data["p"], data["knot"])
        this.m = data["m"]
        this.n = data["n"]
        this.span = data["span"]
        return this

    def save(self, filepath: str) -> None:
        """
        Save the BSplineBasis object to a file.
        Control points are optional.
        Supported extensions: json, pkl
        """
        data = self.to_dict()
        ext = filepath.split(".")[-1]
        if ext == "json":
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        elif ext == "pkl":
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        else:
            raise ValueError(
                f"Unknown extension {ext}. Supported extensions: json, pkl."
            )

    @classmethod
    def load(cls, filepath: str) -> "BSplineBasis":
        """
        Load a BSplineBasis object from a file.
        May return control points if the file contains them.
        Supported extensions: json, pkl
        """
        ext = filepath.split(".")[-1]
        if ext == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
        elif ext == "pkl":
            with open(filepath, "rb") as f:
                data = pickle.load(f)
        else:
            raise ValueError(
                f"Unknown extension {ext}. Supported extensions: json, pkl."
            )
        this = cls.from_dict(data)
        return this

    def plotN(self, k: int = 0, show: bool = True):
        """
        Plot the B-spline basis functions or their derivatives over the span.

        Visualizes each basis function N_i(ξ) or its k-th derivative over its support interval
        using matplotlib. The plot includes proper LaTeX labels and a legend if there are 10 or
        fewer basis functions.

        Parameters
        ----------
        k : int, optional
            Order of derivative to plot. By default, 0 (plots the basis functions themselves).
        show : bool, optional
            Whether to display the plot immediately. Can be useful to add more stuff to the plot.
            By default, True.

        Notes
        -----
        - Uses adaptive sampling with points only in regions where basis functions are non-zero
        - Plots each basis function in a different color with LaTeX-formatted labels
        - Legend is automatically hidden if there are more than 10 basis functions
        - The x-axis represents the isoparametric coordinate ξ

        Examples
        --------
        >>> basis = BSplineBasis(2, [0., 0., 0., 1., 1., 1.])
        >>> basis.plotN()  # Plot basis functions
        >>> basis.plotN(k=1)  # Plot first derivatives
        """
        n_eval_per_elem = 500 // np.unique(self.knot).size
        for idx in range(self.n + 1):
            XI = np.empty(0, dtype="float")
            for i in range(idx, idx + self.p + 1):
                a = self.knot[i]
                b = self.knot[i + 1]
                if a != b:
                    b -= np.finfo("float").eps
                    XI = np.append(XI, np.linspace(a, b, n_eval_per_elem))
            DN_idx = np.empty(0, dtype="float")
            for ind in range(XI.size):
                DN_idx_ind = _funcDNElemOneXi(idx, self.p, self.knot, XI[ind], k)
                DN_idx = np.append(DN_idx, DN_idx_ind)
            label = "$N_{" + str(idx) + "}" + ("'" * k) + "(\\xi)$"
            plt.plot(XI, DN_idx, label=label)
        plt.xlabel("$\\xi$")
        unique_knots, counts = np.unique(self.knot, return_counts=True)
        if unique_knots.size <= 10:
            ylim = plt.ylim()
            y_text = ylim[1] + 0.05 * (ylim[1] - ylim[0])
            id = 0
            for xi, n in zip(unique_knots, counts):
                plt.axvline(xi, color="gray", linestyle=":", linewidth=0.8)
                if n == 1:
                    plt.text(
                        xi,
                        y_text,
                        f"$\\xi_{{{id}}}$",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                else:
                    plt.text(
                        xi,
                        y_text,
                        f"$\\xi_{{{id}-{id + n - 1}}}$",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                id += n
            plt.ylim(ylim[0], y_text + 0.05 * (ylim[1] - ylim[0]))
        if self.n + 1 <= 10:
            plt.legend(loc="best")
        if show:
            plt.show()

    def _funcDElem(self, i, j, new_knot, p):
        """
        Compute the ij value of the knot insertion matrix D.

        Parameters
        ----------
        i : int
            Row index of D.
        j : int
            Column index of D.
        new_knot : numpy.array of float
            New knot vector to use.
        p : int
            Degree of the BSpline.

        Returns
        -------
        D_ij : float
            Value of D at the index ij.

        """
        if p == 0:
            return int(new_knot[i] >= self.knot[j] and new_knot[i] < self.knot[j + 1])
        if self.knot[j + p] != self.knot[j]:
            rec_p = (new_knot[i + p] - self.knot[j]) / (self.knot[j + p] - self.knot[j])
            rec_p *= self._funcDElem(i, j, new_knot, p - 1)
        else:
            rec_p = 0
        if self.knot[j + p + 1] != self.knot[j + 1]:
            rec_j = (self.knot[j + p + 1] - new_knot[i + p]) / (
                self.knot[j + p + 1] - self.knot[j + 1]
            )
            rec_j *= self._funcDElem(i, j + 1, new_knot, p - 1)
        else:
            rec_j = 0
        D_ij = rec_p + rec_j
        return D_ij

    def _D(self, new_knot):
        """
        Compute the `D` matrix used to determine the position of the new control
        points for the knot insertion process. The instance of `BSplineBasis`
        won't be modified here.

        Parameters
        ----------
        new_knot : numpy.array of float
            The new knot vector for the knot insertion.

        Returns
        -------
        D : scipy.sparse.coo_matrix of float
            The matrix `D` such that :
            newCtrlPtsCoordinate = `D` @ ancientCtrlPtsCoordinate.

        """
        new_m = new_knot.size - 1
        new_n = new_m - self.p - 1
        loop1 = new_n + 1
        loop2 = self.p + 1
        nb_val_max = loop1 * loop2
        vals = np.empty(nb_val_max, dtype="float")
        row = np.empty(nb_val_max, dtype="int")
        col = np.empty(nb_val_max, dtype="int")
        nb_not_put = 0
        for ind1 in range(loop1):
            sparse_ind1 = ind1
            i = ind1
            new_knot_i = new_knot[i]
            # find {elem} so that new_knot_i \in [knot_{elem}, knot_{{elem} + 1}[
            elem = _findElem(self.p, self.m, self.n, self.knot, new_knot_i)
            # determine D_ij(new_knot_i) for the values of j where we know D_ij(new_knot_i) not equal to 0
            for ind2 in range(loop2):
                sparse_ind2 = sparse_ind1 * loop2 + ind2
                j = ind2 + elem - self.p
                if j < 0 or j > elem:
                    nb_not_put += 1
                else:
                    sparse_ind = sparse_ind2 - nb_not_put
                    vals[sparse_ind] = self._funcDElem(i, j, new_knot, self.p)
                    row[sparse_ind] = i
                    col[sparse_ind] = j
        if nb_not_put != 0:
            vals = vals[:-nb_not_put]
            row = row[:-nb_not_put]
            col = col[:-nb_not_put]
        D = sps.coo_matrix((vals, (row, col)), shape=(new_n + 1, self.n + 1))
        return D

    def knotInsertion(self, knots_to_add: np.ndarray[np.floating]) -> sps.coo_matrix:
        """
        Insert knots into the B-spline basis and return the transformation matrix.

        Parameters
        ----------
        knots_to_add : np.ndarray[np.floating]
            Array of knots to insert into the knot vector.

        Returns
        -------
        D : sps.coo_matrix
            Transformation matrix such that new control points = `D` @ old control points.

        Notes
        -----
        Updates the basis by:
        - Inserting new knots into the knot vector
        - Incrementing `m` and `n` by the number of inserted knots
        - Computing transformation matrix `D` for control points update

        Examples
        --------
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.knotInsertion(np.array([0.33, 0.67], dtype='float')).A
        array([[1.    , 0.    , 0.    ],
               [0.67  , 0.33  , 0.    ],
               [0.2211, 0.5578, 0.2211],
               [0.    , 0.33  , 0.67  ],
               [0.    , 0.    , 1.    ]])

        The knot vector is modified (as well as n and m) :
        >>> basis.knot
        array([0.  , 0.  , 0.  , 0.33, 0.67, 1.  , 1.  , 1.  ])
        """
        k = knots_to_add.size
        new_knot = np.sort(np.concatenate((self.knot, knots_to_add), dtype="float"))
        D = self._D(new_knot)
        self.m += k
        self.n += k
        self.knot = new_knot
        return D

    def orderElevation(self, t: int) -> sps.coo_matrix:
        """
        Elevate the polynomial degree of the B-spline basis and return the transformation matrix.

        Parameters
        ----------
        t : int
            Amount by which to increase the basis degree. New degree will be current degree plus `t`.

        Returns
        -------
        STD : sps.coo_matrix
            Transformation matrix for control points such that:
            new_control_points = `STD` @ old_control_points

        Notes
        -----
        The method:
        1. Separates B-spline into Bézier segments via knot insertion
        2. Elevates degree of each Bézier segment
        3. Recombines segments into elevated B-spline via knot removal
        4. Updates basis degree, knot vector and other attributes

        Examples
        --------
        Elevate quadratic basis to cubic:
        >>> basis = BSplineBasis(2, np.array([0, 0, 0, 1, 1, 1], dtype='float'))
        >>> basis.orderElevation(1).A
        array([[1.        , 0.        , 0.        ],
               [0.33333333, 0.66666667, 0.        ],
               [0.        , 0.66666667, 0.33333333],
               [0.        , 0.        , 1.        ]])

        The knot vector and the degree are modified (as well as n and m) :
        >>> basis.knot
        array([0., 0., 0., 0., 1., 1., 1., 1.])
        >>> basis.p
        3
        """
        no_dup, counts = np.unique(self.knot, return_counts=True)
        missed = self.p + 1 - counts
        p0 = self.p
        p1 = p0
        p2 = p1 + t
        p3 = p2
        knot0 = self.knot
        knot1 = np.sort(np.concatenate((knot0, np.repeat(no_dup, missed)), axis=0))
        knot2 = np.sort(np.repeat(no_dup, p2 + 1))
        knot3 = np.sort(np.concatenate((knot0, np.repeat(no_dup, t)), axis=0))
        # step 1 : separate the B-spline in bezier curves by knot insertion
        D = self._D(knot1)
        # step 2 : perform the order elevation on every bezier curve
        num_bezier = no_dup.size - 1
        loop1 = num_bezier
        loop2 = p2 + 1
        loop3 = p1 + 1
        nb_val_max = loop1 * loop2 * loop3
        vals = np.empty(nb_val_max, dtype="float")
        row = np.empty(nb_val_max, dtype="int")
        col = np.empty(nb_val_max, dtype="int")
        nb_not_put = 0
        i_offset = 0
        j_offset = 0
        for ind1 in range(loop1):
            sparse_ind1 = ind1
            for ind2 in range(loop2):
                sparse_ind2 = sparse_ind1 * loop2 + ind2
                i = ind2
                inv_denom = 1 / comb(p2, i)  # type: ignore
                for ind3 in range(loop3):
                    sparse_ind3 = sparse_ind2 * loop3 + ind3
                    j = ind3
                    if j < (i - t) or j > i:
                        nb_not_put += 1
                    else:
                        sparse_ind = sparse_ind3 - nb_not_put
                        vals[sparse_ind] = comb(p1, j) * comb(t, i - j) * inv_denom  # type: ignore
                        row[sparse_ind] = i_offset + i
                        col[sparse_ind] = j_offset + j
            i_offset += p2 + 1
            j_offset += p1 + 1
        if nb_not_put != 0:
            vals = vals[:-nb_not_put]
            row = row[:-nb_not_put]
            col = col[:-nb_not_put]
        T = sps.coo_matrix(
            (vals, (row, col)), shape=((p2 + 1) * num_bezier, (p1 + 1) * num_bezier)
        )
        # step 3 : come back to B-spline by removing useless knots
        self.__init__(p2, knot2)
        S = self._D(knot3)
        self.__init__(p3, knot3)
        STD = S @ T @ D
        return STD

    def greville_abscissa(
        self, return_weights: bool = False
    ) -> Union[
        np.ndarray[np.floating], tuple[np.ndarray[np.floating], np.ndarray[np.floating]]
    ]:
        r"""
        Compute the Greville abscissa and optionally their weights for this 1D B-spline basis.

        The Greville abscissa represent the parametric coordinates associated with each
        control point. They are defined as the average of `p` consecutive internal knots.

        Parameters
        ----------
        return_weights : bool, optional
            If `True`, also returns the weights (support lengths) associated with each basis function.
            By default, False.

        Returns
        -------
        greville : np.ndarray[np.floating]
            Array containing the Greville abscissa of size `n + 1`, where `n` is the last index
            of the basis functions in this 1D basis.

        weight : np.ndarray[np.floating], optional
            Only returned if `return_weights` is `True`.
            Array of the same size as `greville`, containing the length of the support of
            each basis function (difference between the end and start knots of its support).

        Notes
        -----
        - The Greville abscissa are computed as the average of `p` consecutive knots:
          for the i-th basis function, its abscissa is
          (knot[i+1] + knot[i+2] + ... + knot[i+p]) / p
        - The weights represent the length of the support of each basis function,
          computed as knot[i+p+1] - knot[i].
        - The number of abscissa equals the number of control points.

        Examples
        --------
        >>> degree = 2
        >>> knot = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
        >>> basis = BSplineBasis(degree, knot)
        >>> greville = basis.greville_abscissa()
        >>> greville
        array([0.  , 0.25, 0.75, 1.  ])

        Compute both abscissa and weights:
        >>> greville, weight = basis.greville_abscissa(return_weights=True)
        >>> weight
        array([0.5, 1. , 1. , 0.5])
        """
        greville = (
            np.convolve(self.knot[1:-1], np.ones(self.p, dtype=int), "valid") / self.p
        )
        if return_weights:
            weight = self.knot[(self.p + 1) :] - self.knot[: -(self.p + 1)]
            return greville, weight
        return greville


# %% fast functions for evaluation


@nb.njit(nb.float64(nb.int64, nb.int64, nb.float64[:], nb.float64), cache=True)
def _funcNElemOneXi(i, p, knot, xi):
    """
    Evaluate the basis function N_i^p(xi) of the BSpline.

    Parameters
    ----------
    i : int
        Index of the basis function wanted.
    p : int
        Degree of the BSpline evaluated.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space at which the BSpline is evaluated.

    Returns
    -------
    N_i : float
        Value of the BSpline basis function N_i^p(xi).

    """
    if p == 0:
        return int(
            (xi >= knot[i] and xi < knot[i + 1])
            or (knot[i + 1] == knot[-1] and xi == knot[i + 1])
        )
    if knot[i + p] != knot[i]:
        rec_p = (xi - knot[i]) / (knot[i + p] - knot[i])
        rec_p *= _funcNElemOneXi(i, p - 1, knot, xi)
    else:
        rec_p = 0
    if knot[i + p + 1] != knot[i + 1]:
        rec_i = (knot[i + p + 1] - xi) / (knot[i + p + 1] - knot[i + 1])
        rec_i *= _funcNElemOneXi(i + 1, p - 1, knot, xi)
    else:
        rec_i = 0
    N_i = rec_p + rec_i
    return N_i


@nb.njit(
    nb.float64(nb.int64, nb.int64, nb.float64[:], nb.float64, nb.int64), cache=True
)
def _funcDNElemOneXi(i, p, knot, xi, k):
    """
    Evaluate the `k`-th derivative of the basis function N_i^p(xi) of the
    BSpline.

    Parameters
    ----------
    i : int
        Index of the basis function wanted.
    p : int
        Degree of the BSpline evaluated.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space at which the BSpline is evaluated.
    k : int
        `k`-th derivative of the BSpline evaluated.

    Raises
    ------
    ValueError
        Can't compute the `k`-th derivative of a B-spline of degree strictly
        less than `k` or if `k`<0.

    Returns
    -------
    DN_i : float
        Value of the `k`-th derivative of the BSpline basis function
        N_i^p(xi).

    """
    if k == 0:
        return _funcNElemOneXi(i, p, knot, xi)
    if p == 0:
        if k >= 0:
            raise ValueError(
                "Impossible to determine the k-th derivative of a B-spline of degree strictly less than k !"
            )
        raise ValueError(
            "Impossible to determine the k-th derivative of a B-spline if k<0 !"
        )
    if knot[i + p] != knot[i]:
        rec_p = p / (knot[i + p] - knot[i])
        rec_p *= _funcDNElemOneXi(i, p - 1, knot, xi, k - 1)
    else:
        rec_p = 0
    if knot[i + p + 1] != knot[i + 1]:
        rec_i = p / (knot[i + p + 1] - knot[i + 1])
        rec_i *= _funcDNElemOneXi(i + 1, p - 1, knot, xi, k - 1)
    else:
        rec_i = 0
    N_i = rec_p - rec_i
    return N_i


@nb.njit(nb.int64(nb.int64, nb.int64, nb.int64, nb.float64[:], nb.float64), cache=True)
def _findElem(p, m, n, knot, xi):
    """
    Find `i` so that `xi` belongs to
    [ `knot`[`i`], `knot`[`i` + 1] [.

    Parameters
    ----------
    p : int
        Degree of the polynomials composing the basis.
    m : int
        Last index of the knot vector.
    n : int
        Last index of the basis.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    xi : float
        Value in the parametric space.

    Raises
    ------
    ValueError
        If the value of `xi` is outside the definition interval
        of the spline.

    Returns
    -------
    i : int
        Index of the first knot of the interval in which `xi` is bounded.

    """
    if xi == knot[m - p]:
        return n
    i = 0
    pastrouve = True
    while i <= n and pastrouve:
        pastrouve = xi < knot[i] or xi >= knot[i + 1]
        i += 1
    if pastrouve:
        raise ValueError("xi is outside the definition interval of the spline !")
        # print("ValueError : xi=", xi, " is outside the definition interval [", knot[p], ", ", knot[m - p], "] of the spline !")
        # return None
    i -= 1
    return i


@nb.njit(
    nb.types.UniTuple.from_types((nb.float64[:], nb.int64[:], nb.int64[:]))(
        nb.int64, nb.int64, nb.int64, nb.float64[:], nb.float64[:], nb.int64
    ),
    cache=True,
)
def _DN(p, m, n, knot, XI, k):
    """
    Compute the `k`-th derivative of the BSpline basis functions for a set
    of values in the parametric space.

    Parameters
    ----------
    p : int
        Degree of the polynomials composing the basis.
    m : int
        Last index of the knot vector.
    n : int
        Last index of the basis.
    knot : numpy.array of float
        Knot vector of the BSpline basis.
    XI : numpy.array of float
        Values in the parametric space at which the BSpline is evaluated.
    k : int
        `k`-th derivative of the BSpline evaluated. The default is 0.

    Returns
    -------
    (vals, row, col) : (numpy.array of float, numpy.array of int, numpy.array of int)
        Values and indices of the `k`-th derivative matrix of the BSpline
        basis functions in the columns for each value of `XI` in the rows.

    """
    loop1 = XI.size
    loop2 = p + 1
    nb_val_max = loop1 * loop2
    vals = np.empty(nb_val_max, dtype="float")
    row = np.empty(nb_val_max, dtype="int")
    col = np.empty(nb_val_max, dtype="int")
    nb_not_put = 0
    for ind1 in range(loop1):  # nb.p
        sparse_ind1 = ind1
        i_xi = ind1
        xi = XI.flat[i_xi]
        # find {elem} so that \xi \in [\xi_{elem}, \xi_{{elem} + 1}[
        elem = _findElem(p, m, n, knot, xi)
        # determine DN_i(\xi) for the values of i where we know DN_i(\xi) not equal to 0
        for ind2 in range(loop2):
            sparse_ind2 = sparse_ind1 * loop2 + ind2
            i = ind2 + elem - p
            if i < 0:
                nb_not_put += 1
            else:
                sparse_ind = sparse_ind2 - nb_not_put
                vals[sparse_ind] = _funcDNElemOneXi(i, p, knot, xi, k)
                row[sparse_ind] = i_xi
                col[sparse_ind] = i
    if nb_not_put != 0:
        vals = vals[:-nb_not_put]
        row = row[:-nb_not_put]
        col = col[:-nb_not_put]
    return (vals, row, col)
