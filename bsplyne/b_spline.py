import os
import re
import xml.dom.minidom

import numpy as np
import scipy.sparse as sps
from wide_product import wide_product
import meshio as io

from bsplyne_lib.b_spline_basis import BSplineBasis

class BSpline:
    """
    B-Spline from a `NPa`-D parametric space into a NPh-D physical space.
    NPh is not an attribute of this class.

    Attributes
    ----------
    NPa : int
        Parametric space dimension.
    bases : numpy.array of BSplineBasis
        `numpy`.`array` containing a `BSplineBasis` instance for each of the 
        `NPa` axis of the parametric space.

    """
    
    def __init__(self, degrees, knots):
        """
        
        Parameters
        ----------
        degrees : numpy.array of int
            Contains the degrees of the B-spline in each parametric dimension.
        knots : list of numpy.array of float
            Contains the knot vectors of the B-spline for each parametric 
            dimension.

        Returns
        -------
        BSpline : BSpline instance
            Contains the ``BSpline`` object created.

        Examples
        --------
        Creation of a 2D shape as a `BSpline` instance :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> BSpline(degrees, knots)

        """
        self.NPa = len(degrees)
        self.bases = np.empty(self.NPa, dtype='object')
        for idx in range(self.NPa):
            p = degrees[idx]
            knot = knots[idx]
            self.bases[idx] = BSplineBasis(p, knot)
    
    @classmethod
    def from_bases(cls, bases):
        """
        Create a ``BSpline`` instance from an array of ``BSplineBasis``.

        Parameters
        ----------
        bases : numpy.ndarray of BSplineBasis
            The array of ``BSplineBasis`` instances.

        Returns
        -------
        BSpline
            Contains the ``BSpline`` object created.
        """
        self = cls([], [])
        self.bases = bases
        self.NPa = self.bases.size
        return self
    
    def getDegrees(self):
        """
        Returns the degree of each basis in the parametric space.

        Parameters
        ----------
        None

        Returns
        -------
        degrees : numpy.array of int
            Contains the degrees of the B-spline.

        Examples
        --------
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getDegrees()
        array([2, 2])

        """
        degrees = np.array([basis.p for basis in self.bases], dtype='int')
        return degrees
    
    def getKnots(self):
        """
        Returns the knot vector of each basis in the parametric space.

        Parameters
        ----------
        None

        Returns
        -------
        knots : list of numpy.array of float
            Contains the knot vectors of the B-spline.

        Examples
        --------
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> spline.getKnots()
        [array([0. , 0. , 0. , 0.5, 1. , 1. , 1. ]),
         array([0. , 0. , 0. , 0.5, 1. , 1. , 1. ])]

        """
        knots = [basis.knot for basis in self.bases]
        return knots

    def getNbFunc(self):
        """
        Compute the number of basis functions of the spline.

        Returns
        -------
        int
            Number of basis functions.
        """
        return np.prod([basis.n + 1 for basis in self.bases])
    
    def get_indices(self, begining=0):
        """
        Create an array containing the indices of the control points of 
        the B-spline.

        Parameters
        ----------
        begining : int, optional
            First index of the arrayof indices, by default 0

        Returns
        -------
        indices : np.array of int
            Indices of the control points in the same shape as the 
            control points.
        """
        indices = np.arange(begining, begining + self.ctrlPts.size).reshape(self.ctrlPts.shape)
        return indices

    def linspace(self, n_eval_per_elem=10):
        """
        Generate `NPa` sets of xi values over the span of each basis.

        Parameters
        ----------
        n_eval_per_elem : numpy.array of int or int, optional
            Number of values per element over each parametric axis, by default 10

        Returns
        -------
        XI : tuple of numpy.array of float
            Set of xi values over each span.
        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        XI = tuple([basis.linspace(n) for basis, n in zip(self.bases, n_eval_per_elem)])
        return XI
    
    def DN(self, XI, k=0):
        """
        Compute the `k`-th derivative of the B-spline basis at the points 
        in the parametric space given as input such that a dot product 
        with the reshaped and transposed control points evaluates the 
        B-spline.

        Parameters
        ----------
        XI : numpy.array of float or tuple of numpy.array of float
            If `numpy`.`array` of `float`, contains the `NPa`-uplets of 
            parametric coordinates as [[xi_0, ...], [eta_a, ...], ...].
            Else, if `tuple` of `numpy`.`array` of `float`, contains the `NPa` 
            parametric coordinates as [[xi_0, ...], [eta_0, ...], ...].
        k : numpy.array of int or int, optional
            If `numpy`.`array` of `int`, or if k is 0, compute the `k`-th 
            derivative of the B-spline basis evaluated on each axis of the 
            parametric space.
            If `int`, compute the `k`-th derivative along every axis. For 
            example, if `k` is 1, compute the gradient, if `k` is 2, compute 
            the hessian, and so on.
            , by default 0

        Returns
        -------
        DN : scipy.sparse.csr_matrix of float or numpy.array of scipy.sparse.csr_matrix of float
            Contains the basis of the B-spline.

        Examples
        --------
        Evaluation of a 2D BSpline basis on these `XI` values : [[0, 0.5], [1, 0]]
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> spline = BSpline(degrees, knots)
        >>> XI = np.array([[0, 0.5], [1, 0]], dtype='float')
        >>> spline.DN(XI, [0, 0]).A
        array([[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
               [0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])
        
        Evaluation of the 2D BSpline basis's derivative along the first axis :
        >>> XI = (np.array([0, 0.5], dtype='float'), 
                  np.array([1], dtype='float'))
        >>> spline.DN(XI, [1, 0]).A
        array([[ 0., -0., -0., -4.,  0.,  0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
               [ 0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        """
        
        if isinstance(XI, np.ndarray):
            fct = wide_product
            XI = XI.reshape((self.NPa, -1))
        else:
            fct = sps.kron
        
        if isinstance(k, int):
            if k==0:
                k = [0]*self.NPa
            else:
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
                        dic[key] = None
                        for idx in range(self.NPa):
                            k_querry = k_arr[idx]
                            if dic[key] is None:
                                dic[key] = dkbasis_dxik[idx, k_querry]
                            else:
                                dic[key] = fct(dic[key], dkbasis_dxik[idx, k_querry])
                    DN[axes] = dic[key]
                return DN
        
        if not isinstance(k, int):
            DN = None
            for idx in range(self.NPa):
                basis = self.bases[idx]
                k_idx = k[idx]
                xi = XI[idx] - np.finfo('float').eps * (XI[idx]==basis.knot[-1])
                DN_elem = basis.N(xi, k=k_idx)
                if DN is None:
                    DN = DN_elem
                else:
                    DN = fct(DN, DN_elem)
            return DN
    
    def __call__(self, ctrlPts, XI, k=0):
        """
        Evaluate the `k`-th derivative of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        XI : numpy.array of float or tuple of numpy.array of float
            If `numpy`.`array` of `float`, contains the `NPa`-uplets of 
            parametric coordinates as [[xi_0, ...], [eta_a, ...], ...].
            Else, if `tuple` of `numpy`.`array` of `float`, contains the `NPa` 
            parametric coordinates as [[xi_0, ...], [eta_0, ...], ...].
        k : numpy.array of int or int, optional
            If `numpy`.`array` of `int`, or if k is 0, compute the `k`-th 
            derivative of the B-spline evaluated on each axis of the parametric 
            space.
            If `int`, compute the `k`-th derivative along every axis. For 
            example, if `k` is 1, compute the gradient, if `k` is 2, compute 
            the hessian, and so on.
            , by default 0

        Returns
        -------
        values : numpy.array of float
            Evaluation of the `k`-th derivative of the B-spline at the 
            parametric space's coordinates given. This array contains the 
            physical space coordinates corresponding as [X, Y, ...]. 
            If `XI` is a tuple, its shape is 
            (NPh, shape of the derivative, len(xi), len(eta), ...).
            Else, its shape is 
            (NPh, shape of the derivative, *`XI`.shape[1:]).

        Examples
        --------
        Evaluation of a 2D BSpline on a grid :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> XI = (np.array([0, 0.5], dtype='float'), 
                  np.array([1], dtype='float'))
        >>> spline(ctrlPts, XI, [0, 0])
        array([[[0.0247786 ],
                [0.69670203]],
        
               [[0.80417637],
                [0.8896871 ]],
        
               [[0.8208673 ],
                [0.51415427]]])
        
        Evaluation of a 2D BSpline on an array of couples :
        >>> XI = np.array([[0, 0.5], [1, 0]], dtype='float')
        >>> spline(ctrlPts, XI, [0, 0])
        array([[0.0247786 , 0.42344584],
               [0.80417637, 0.42489571],
               [0.8208673 , 0.07327555]])

        """
        if isinstance(XI, np.ndarray):
            XI_shape = XI.shape[1:]
        else:
            XI_shape = [xi.size for xi in XI]
        DN = self.DN(XI, k)
        NPh = ctrlPts.shape[0]
        if isinstance(DN, np.ndarray):
            values = np.empty((NPh, *DN.shape, XI_shape))
            slice_before = slice(0, NPh)
            slices_after = [slice(0, xi_shape) for xi_shape in XI_shape]
            for axes in np.ndindex(*DN.shape):
                indices = (slice_before, *axes, *slices_after)
                values[indices] = (DN @ ctrlPts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        else:
            values = (DN @ ctrlPts.reshape((NPh, -1)).T).T.reshape((NPh, *XI_shape))
        return values
    
    def knotInsertion(self, ctrlPts, knots_to_add):
        """
        Add the knots passed in parameter to the knot vector and modify the 
        attributes so that the evaluation of the spline stays the same.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        knots_to_add : list of np.array of float
            Contains the knots to add. It must not contain knots outside of 
            the old knot vector's interval.

        Returns
        -------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)

        Examples
        --------
        Knot insertion on a 2D BSpline in a 3D space :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> knots_to_add = [np.array([0.5, 0.75], dtype='float'), 
                            np.array([], dtype='float')]
        >>> ctrlPts = pline.knotInsertion(ctrlPts, knots_to_add)

        """
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
        NPh = ctrlPts.shape[0]
        pts = (D @ ctrlPts.reshape((NPh, -1)).T).T
        ctrlPts = pts.reshape((NPh, *pts_shape))
        return ctrlPts
    
    def orderElevation(self, ctrlPts, t):
        """
        Performs the order elevation algorithm on every B-spline basis and 
        apply the changes to the control points of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        t : numpy.array of int
            New degree of each B-spline basis will be its current degree plus 
            the value of `t` corresponding.

        Returns
        -------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)

        Examples
        --------
        Order elevation on a 2D BSpline in a 3D space :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> ctrlPts = spline.knotInsertion(ctrlPts, [1, 0])

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
        NPh = ctrlPts.shape[0]
        pts = (STD @ ctrlPts.reshape((NPh, -1)).T).T
        ctrlPts = pts.reshape((NPh, *pts_shape))
        return ctrlPts
    
    def _saveControlPolyParaview(self, ctrlPts, file_prefix, n_step, fields={}):
        """
        Saves a paraview file containing all the data to plot the control 
        polygon of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        NPh = ctrlPts.shape[0]
        lines = np.empty((0, 2), dtype='int')
        size = np.prod(ctrlPts.shape[1:])
        inds = np.arange(size).reshape(ctrlPts.shape[1:])
        for idx in range(self.NPa):
            rng = np.arange(inds.shape[idx])
            lines = np.append(lines, 
                              np.concatenate((np.expand_dims(np.take(inds, rng[ :-1], axis=idx), axis=-1), 
                                              np.expand_dims(np.take(inds, rng[1:  ], axis=idx), axis=-1)), 
                                             axis=-1).reshape((-1, 2)), 
                             axis=0)
        cells = {'line': lines}
        points = np.moveaxis(ctrlPts, 0, -1).reshape((-1, NPh))
        greville = []
        for idx in range(self.NPa):
            basis = self.bases[idx]
            p = basis.p
            knot = basis.knot
            n = ctrlPts.shape[1+idx]
            greville.append(np.convolve(knot[1:-1], np.ones(p, dtype=int), 'valid')/p)
        greville = tuple(greville)
        n = self.getNbFunc()
        point_data = {}
        for key, value in fields.items():
            if callable(value):
                point_data[key] = value(self, greville)
            else:
                point_data[key] = value.reshape((n_step, -1, n)).transpose(0, 2, 1)
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step)
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def _saveElemSeparatorParaview(self, ctrlPts, n_eval_per_elem, file_prefix, n_step=1, fields={}):
        """
        Saves a paraview file containing all the data to plot the limit of 
        every element of the B-spline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        n_eval_per_elem : numpy.array of int
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        NPh = ctrlPts.shape[0]
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
                XI = tuple(knots_uniq[:idx] + [lin] + knots_uniq[(idx+1):])
                shape = shape_uniq[:idx] + [lin.size] + shape_uniq[(idx+1):]
                N = self.DN(XI, [0]*self.NPa)
                pts = N @ ctrlPts.reshape((NPh, -1)).T
                points = pts if points is None else np.vstack((points, pts))
                for key, value in fields.items():
                    if callable(value):
                        to_store = value(self, XI)
                    else:
                        paraview_size = value.shape[1]
                        arr = value.reshape((n_step, paraview_size, n)).reshape((n_step*paraview_size, n))
                        to_store = (arr @ N.T).reshape((n_step, paraview_size, -1)).transpose(0, 2, 1)
                    if point_data[key] is None:
                        point_data[key] = to_store
                    else:
                        point_data[key] = np.concatenate((point_data[key], to_store), axis=1)
                size = np.prod(shape)
                lns = Size + np.arange(size).reshape(shape)
                lns = np.moveaxis(lns, idx, 0).reshape((shape[idx], -1))
                lns = np.concatenate((np.expand_dims(lns[ :-1], axis=-1), 
                                      np.expand_dims(lns[1:  ], axis=-1)), 
                                     axis=-1).reshape((-1, 2))
                lines = lns if lines is None else np.vstack((lines, lns))
                Size += size
        cells = {'line': lines}
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step)
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def _saveElementsInteriorParaview(self, ctrlPts, n_eval_per_elem, file_prefix, n_step=1, fields={}):
        """
        Saves a paraview file containing all the data to plot the interior of 
        the elements of the B-pline.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        n_eval_per_elem : numpy.array of int
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        file_prefix : string
            File name and path where the paraview plot will be saved. 
            A "_", a step number and a ".vtu" will be added.
        n_step : int
            Number of time steps to plot.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 

        Returns
        -------
        None.

        """
        XI = self.linspace(n_eval_per_elem)
        # make points
        N = self.DN(XI)
        NPh = ctrlPts.shape[0]
        points = N @ ctrlPts.reshape((NPh, -1)).T
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
                paraview_size = value.shape[1]
                arr = value.reshape((n_step*paraview_size, n))
                point_data[key] = (arr @ N.T).reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
        # make fields
        # """
        # fields : dict of function or of tuple(int or numpy.array of int, numpy.array of float), default {}
        #     Fields to plot at each time step. The name of the field will 
        #     be the dict key. 
        #     If the value given is a `function`, it must take the spline 
        #     and a `tuple` of parametric points that could be given to 
        #     `self`.`DN` for example. It must return its value for 
        #     each time step and on each combination of parametric points.
        #     `function`(`BSpline` spline, 
        #                `tuple`(`numpy`.`array` of `float`) XI) 
        #     -> `numpy`.`array` of `float` of shape 
        #     (`n_step`, nb combinations of XI, paraview_size) 
        #     where paraview_size is the size of the vector at each control 
        #     point.
        #     If the value given is a tuple (k, pts) : 
        #     - pts : a `numpy`.`array` of `float` containing the value at 
        #         each control point, the shape must be 
        #         (`n_step`, paraview_size, *`self`.`ctrlPts`.`shape`[1:])
        #     - k : the value(s) of k given to the `DN` method. 
        #         If k is a `numpy`.`array`, its shape should be 
        #         (paraview_size, `self`.`NPa`).
        # """
        # NFct = self.getNbFunc()
        # point_data = {}
        # for key, (k, pts) in fields.items():
        #     if callable(value):
        #         point_data[key] = value(self, XI)
        #     else:
        #         paraview_size = pts.shape[1]
        #         if isinstance(k, int):
        #             if k==0:
        #                 arr = pts.reshape((n_step*paraview_size, NFct))
        #                 point_data[key] = (arr @ N.T).reshape((n_step, paraview_size, NXI)).transpose(0, 2, 1)
        #             else:
        #                 DkN = self.DN(XI, k).ravel()
        #                 if paraview_size!=DkN.size:
        #                     if paraview_size==1:
        #                         arr = pts.reshape((n_step, NFct))
        #                         paraview_size = DkN.size
        #                         point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                         for i in range(paraview_size):
        #                             point_data[key][:, :, i] = (arr @ DkN[i].T).reshape((n_step, NXI))
        #                     else:
        #                         raise ValueError("paraview_size is " 
        #                                          + str(paraview_size) 
        #                                          + " but " 
        #                                          + str(DkN.size) 
        #                                          + " derivatives were acquiered !")
        #                 else:
        #                     arr = pts.reshape((n_step, paraview_size, NFct))
        #                     point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                     for i in range(paraview_size):
        #                         point_data[key][:, :, i] = (arr[:, i, :] @ DkN[i].T).reshape((n_step, NXI))
        #         else:
        #             DkN = np.array([self.DN(XI, ki) for ki in k], dtype='object')
        #             if paraview_size!=DkN.size:
        #                 if paraview_size==1:
        #                     arr = pts.reshape((n_step, NFct))
        #                     paraview_size = DkN.size
        #                     point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                     for i in range(paraview_size):
        #                         point_data[key][:, :, i] = (arr @ DkN[i].T).reshape((n_step, NXI))
        #                 else:
        #                     raise ValueError("paraview_size is " 
        #                                         + str(paraview_size) 
        #                                         + " but " 
        #                                         + str(DkN.size) 
        #                                         + " derivatives were acquiered !")
        #             else:
        #                 arr = pts.reshape((n_step, paraview_size, NFct))
        #                 point_data[key] = np.empty((n_step, NXI, paraview_size), dtype='float')
        #                 for i in range(paraview_size):
        #                     point_data[key][:, :, i] = (arr[:, i, :] @ DkN[i].T).reshape((n_step, NXI))
        # save files
        for i in range(n_step):
            point_data_step = {}
            for key, value in point_data.items():
                point_data_step[key] = value[i]
            mesh = io.Mesh(points, cells, point_data_step)
            mesh.write(file_prefix+"_"+str(i)+".vtu")
    
    def saveParaview(self, ctrlPts, path, name, n_step=1, n_eval_per_elem=10, fields={}, groups=None, make_pvd=True, verbose=True):
        """
        Saves a plot as a set of .vtu files with a .pvd file.

        Parameters
        ----------
        ctrlPts : numpy.array of float
            Contains the control points of the B-spline as [X, Y, Z, ...].
            Its shape : (NPh, nb elem for dim 1, ..., nb elem for dim `NPa`)
        path : string
            Path of the directory in which all the files to show in Paraview 
            will be dumped.
        name : string
            Prefix of the files created.
        n_step : int
            Number of time steps to plot.
        n_eval_per_elem : numpy.array of int or int, default 10
            Contains the number of evaluation of the B-spline in each 
            direction of the parametric space for each element.
        fields : dict of function or of numpy.array of float, default {}
            Fields to plot at each time step. The name of the field will 
            be the dict key. 
            If the value given is a `function`, it must take the spline 
            and a `tuple` of parametric points that could be given to 
            `self`.`DN` for example. It must return its value for 
            each time step and on each combination of parametric points.
            `function`(`BSpline` spline, 
                       `tuple`(`numpy`.`array` of `float`) XI) 
            -> `numpy`.`array` of `float` of shape 
            (`n_step`, nb combinations of XI, size for paraview)
            If the value given is a `numpy`.`array` of `float`, the 
            shape must be :
            (`n_step`, size for paraview, *`ctrlPts`.`shape`[1:]) 
        groups : dict of dict, default None
            `dict` (out) of `dict` (in) as :
            - (out) : 
                * "interior" : (in) type of `dict`, 
                * "elements_borders" : (in) type of `dict`, 
                * "control_points" : (in) type of `dict`.
                * other keys from the input that are not checked
            - (in) : 
                * "ext" : name of the extention of the group, 
                * "npart" : number of parts to plot together,
                * "nstep" : number of time steps.
        make_pvd : bool, default True
            If True, create a PVD file for all the data in `groups`.
        verbose : bool, default True
            If True, print the advancement state to the standard output.

        Returns
        -------
        groups : dict of dict
            `dict` (out) of `dict` (in) as :
            - (out) : 
                * "interior" : (in) type of `dict`, 
                * "elements_borders" : (in) type of `dict`, 
                * "control_points" : (in) type of `dict`.
                * other keys from the input that are not checked
            - (in) : 
                * "ext" : name of the extention of the group, 
                * "npart" : number of parts to plot together,
                * "nstep" : number of time steps.

        Examples
        --------
        Save a 2D BSpline in a 3D space in the file file.pvd at the 
        location /path/to/file :
        >>> degrees = np.array([2, 2], dtype='int')
        >>> knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
                     np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
        >>> ctrlPts = np.random.rand(3, 4, 4)
        >>> spline = BSpline(degrees, knots)
        >>> spline.saveParaview(ctrlPts, "/path/to/file", "file")

        """
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        interior = "interior"
        elements_borders = "elements_borders"
        control_points = "control_points"
        if groups is None:
            groups = {interior:         {"ext": "vtu", "npart": 1, "nstep": n_step}, 
                      elements_borders: {"ext": "vtu", "npart": 1, "nstep": n_step}, 
                      control_points  : {"ext": "vtu", "npart": 1, "nstep": n_step}}
        else:
            groups[interior]["npart"] += 1
            groups[elements_borders]["npart"] += 1
            groups[control_points]["npart"] += 1
        
        interior_prefix = os.path.join(path, name+"_"+interior+"_"+str(groups[interior]["npart"] - 1))
        self._saveElementsInteriorParaview(ctrlPts, n_eval_per_elem, interior_prefix, n_step, fields)
        if verbose:
            print(interior, "done")

        elements_borders_prefix = os.path.join(path, name+"_"+elements_borders+"_"+str(groups[elements_borders]["npart"] - 1))
        self._saveElemSeparatorParaview(ctrlPts, n_eval_per_elem, elements_borders_prefix, n_step, fields)
        if verbose:
            print(elements_borders, "done")

        control_points_prefix = os.path.join(path, name+"_"+control_points+"_"+str(groups[control_points]["npart"] - 1))
        self._saveControlPolyParaview(ctrlPts, control_points_prefix, n_step, fields)
        if verbose:
            print(control_points, "done")
        
        if make_pvd:
            _writePVD(os.path.join(path, name), groups)
        
        return groups

def _writePVD(fileName, groups):
    """
    Write PVD file
    Usage: _writePVD("toto", "vtu", groups) 
    generated file: "toto.pvd" 
    
    VTK files must be named as follows:
    groups={"a": {"ext": "vtu", "npart": 1, "nstep": 2}, "b": {"ext": "vtr", "npart": 2, "nstep": 5}}
    =>  toto_b_2_5.vtr  (starts from toto_a_0_0.vtu)
    
    Parameters
    ----------
    fileName : STRING
        mesh files without numbers and extension
    groups : DICT of DICT
        Dict (out) of dict (in) as :
        - (out) : 
            * keys are the names of group to plot, 
            * values are (in) dict, 
        - (in) : 
            * "ext" : name of the extention of the group, 
            * "npart" : number of parts to plot together,
            * "nstep" : number of time steps.

    """
    rep,fname=os.path.split(fileName)
    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)
    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)    
    for name, grp in groups.items():
        for jp in range(grp["npart"]):
            for js in range(grp["nstep"]):
                dataSet = pvd.createElementNS("VTK", "DataSet")
                dataSet.setAttribute("timestep", str(js))
                dataSet.setAttribute("group", name)
                dataSet.setAttribute("part", str(jp))
                dataSet.setAttribute("file", fname+"_"+name+"_"+str(jp)+"_"+str(js)+"."+grp["ext"])
                dataSet.setAttribute("name", name)
                collection.appendChild(dataSet)
    outFile = open(fileName+".pvd", 'w')
    pvd.writexml(outFile, newl='\n')
    print("VTK: "+ fileName +".pvd written")
    outFile.close()