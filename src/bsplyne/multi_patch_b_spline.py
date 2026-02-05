# %%
import os
from itertools import permutations
from typing import Iterable, Union, Literal

import numpy as np
import numba as nb
import meshio as io
from scipy.spatial import cKDTree

from .b_spline import BSpline
from .b_spline_basis import BSplineBasis
from .save_utils import writePVD, merge_meshes
from .parallel_utils import parallel_blocks

# from .save_YETI import Domain, write


# union-find algorithm for connectivity
@nb.njit(nb.int32(nb.int32[:], nb.int32), cache=True)
def _find(parent, x):
    if parent[x] != x:
        parent[x] = _find(parent, parent[x])
    return parent[x]


@nb.njit(nb.void(nb.int32[:], nb.int32[:], nb.int32, nb.int32), cache=True)
def _union(parent, rank, x, y):
    rootX = _find(parent, x)
    rootY = _find(parent, y)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


@nb.njit(nb.int32[:](nb.int32[:, :], nb.int64), cache=True)
def _get_unique_nodes_inds(nodes_couples, nb_nodes):
    parent = np.arange(nb_nodes, dtype=np.int32)
    rank = np.zeros(nb_nodes, dtype=np.int32)
    for a, b in nodes_couples:
        _union(parent, rank, a, b)
    unique_nodes_inds = np.empty(nb_nodes, dtype=np.int32)
    for i in range(nb_nodes):
        unique_nodes_inds[i] = _find(parent, i)
    return unique_nodes_inds


class MultiPatchBSplineConnectivity:
    """
    Contains all the methods to link multiple B-spline patches.
    It uses 3 representations of the data :
      - a unique representation, possibly common with other meshes, containing
        only unique nodes indices,
      - a unpacked representation containing duplicated nodes indices,
      - a separated representation containing duplicated nodes indices,
        separated between patches. It is here for user friendliness.

    Attributes
    ----------
    unique_nodes_inds : np.ndarray[np.integer]
        The indices of the unique representation needed to create the unpacked one.
    shape_by_patch : np.ndarray[np.integer]
        The shape of the separated nodes by patch.
    nb_nodes : int
        The total number of unpacked nodes.
    nb_unique_nodes : int
        The total number of unique nodes.
    nb_patchs : int
        The number of patches.
    npa : int
        The dimension of the parametric space of the B-splines.
    """

    unique_nodes_inds: np.ndarray
    shape_by_patch: np.ndarray
    nb_nodes: int
    nb_unique_nodes: int
    nb_patchs: int
    npa: int

    def __init__(self, unique_nodes_inds, shape_by_patch, nb_unique_nodes):
        """

        Parameters
        ----------
        unique_nodes_inds : np.ndarray[np.integer]
            The indices of the unique representation needed to create the unpacked one.
        shape_by_patch : np.ndarray[np.integer]
            The shape of the separated nodes by patch.
        nb_unique_nodes : int
            The total number of unique nodes.
        """
        self.unique_nodes_inds = unique_nodes_inds
        self.shape_by_patch = shape_by_patch
        self.nb_nodes = np.sum(np.prod(self.shape_by_patch, axis=1))
        self.nb_unique_nodes = nb_unique_nodes  # np.unique(self.unique_nodes_inds).size
        self.nb_patchs, self.npa = self.shape_by_patch.shape

    @classmethod
    def from_nodes_couples(cls, nodes_couples, shape_by_patch):
        """
        Create the connectivity from a list of couples of unpacked nodes.

        Parameters
        ----------
        nodes_couples : np.ndarray[np.integer]
            Couples of indices of unpacked nodes that are considered the same.
            Its shape should be (# of couples, 2)
        shape_by_patch : np.ndarray[np.integer]
            The shape of the separated nodes by patch.

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        nb_nodes = np.sum(np.prod(shape_by_patch, axis=1))
        unique_nodes_inds = _get_unique_nodes_inds(
            nodes_couples.astype(np.int32), np.int64(nb_nodes)
        )
        different_unique_nodes_inds, inverse = np.unique(
            unique_nodes_inds, return_inverse=True
        )
        unique_nodes_inds -= np.cumsum(
            np.diff(np.concatenate(([-1], different_unique_nodes_inds))) - 1
        )[inverse]
        nb_unique_nodes = np.unique(unique_nodes_inds).size
        return cls(unique_nodes_inds, shape_by_patch, nb_unique_nodes)

    @classmethod
    def from_separated_ctrlPts(
        cls, separated_ctrlPts, eps=1e-10, return_nodes_couples: bool = False
    ):
        """
        Create the connectivity from a list of control points given as
        a separated field by comparing every couple of points.

        Parameters
        ----------
        separated_ctrlPts : list of np.ndarray[np.floating]
            Control points of every patch to be compared in the separated
            representation. Every array is of shape :
            (``NPh``, nb elem for dim 1, ..., nb elem for dim ``npa``)
        eps : float, optional
            Maximum distance between two points to be considered the same, by default 1e-10
        return_nodes_couples : bool, optional
            If `True`, returns the `nodes_couples` created, by default False

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        NPh = separated_ctrlPts[0].shape[0]
        assert np.all(
            [ctrlPts.shape[0] == NPh for ctrlPts in separated_ctrlPts[1:]]
        ), "Physical spaces must contain the same number of dimensions !"
        shape_by_patch = np.array(
            [ctrlPts.shape[1:] for ctrlPts in separated_ctrlPts], dtype="int"
        )

        ctrlPts = np.hstack([pts.reshape((NPh, -1)) for pts in separated_ctrlPts])
        tree = cKDTree(ctrlPts.T)
        matches = tree.query_ball_tree(tree, r=eps)
        connex = set()
        for match in matches:
            if len(match) > 1:
                connex.add(tuple(sorted(match)))
        nodes_couples = []
        for tup in connex:
            tup = np.array(tup)
            nodes_couples.append(np.stack((tup[:-1], tup[1:])).T)

        # nodes_couples = []
        # previous_pts = separated_ctrlPts[0].reshape((NPh, -1))
        # previous_inds_counter = previous_pts.shape[1]
        # previous_inds = np.arange(previous_inds_counter)
        # for ctrlPts in separated_ctrlPts[1:]:
        #     # create current pts and inds
        #     current_pts = ctrlPts.reshape((NPh, -1))
        #     current_inds_counter = previous_inds_counter + current_pts.shape[1]
        #     current_inds = np.arange(previous_inds_counter, current_inds_counter)
        #     # get couples
        #     dist = np.linalg.norm(
        #         previous_pts[:, :, None] - current_pts[:, None, :], axis=0
        #     )
        #     previous_inds_inds, current_inds_inds = (dist < eps).nonzero()
        #     nodes_couples.append(
        #         np.hstack(
        #             (
        #                 previous_inds[previous_inds_inds, None],
        #                 current_inds[current_inds_inds, None],
        #             )
        #         )
        #     )
        #     # add current to previous for next iteration
        #     previous_pts = np.hstack((previous_pts, current_pts))
        #     previous_inds_counter = current_inds_counter
        #     previous_inds = np.hstack((previous_inds, current_inds))
        if len(nodes_couples) > 0:
            nodes_couples = np.vstack(nodes_couples)
        else:
            nodes_couples = np.empty((0, 2), dtype="int")
        if return_nodes_couples:
            return cls.from_nodes_couples(nodes_couples, shape_by_patch), nodes_couples
        else:
            return cls.from_nodes_couples(nodes_couples, shape_by_patch)

    def unpack(self, unique_field):
        """
        Extract the unpacked representation from a unique representation.

        Parameters
        ----------
        unique_field : np.ndarray
            The unique representation. Its shape should be :
            (field, shape, ..., `self`.`nb_unique_nodes`)

        Returns
        -------
        unpacked_field : np.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        unpacked_field = unique_field[..., self.unique_nodes_inds]
        return unpacked_field

    def pack(self, unpacked_field, method="mean"):
        """
        Extract the unique representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : np.ndarray
            The unpacked representation. Its shape should be :
            (field, shape, ..., `self`.`nb_nodes`)
        method: str
            The method used to group values that could be different

        Returns
        -------
        unique_nodes : np.ndarray
            The unique representation. Its shape is :
            (field, shape, ..., `self`.`nb_unique_nodes`)
        """
        field_shape = unpacked_field.shape[:-1]
        unique_field = np.zeros(
            (*field_shape, self.nb_unique_nodes), dtype=unpacked_field.dtype
        )
        if method == "last":
            unique_field[..., self.unique_nodes_inds] = unpacked_field
        elif method == "first":
            unique_field[..., self.unique_nodes_inds[::-1]] = unpacked_field[..., ::-1]
        elif method == "mean":
            np.add.at(unique_field.T, self.unique_nodes_inds, unpacked_field.T)
            counts = np.zeros(self.nb_unique_nodes, dtype="uint")
            np.add.at(counts, self.unique_nodes_inds, 1)
            unique_field /= counts
        else:
            raise NotImplementedError(
                f"Method {method} is not implemented ! Consider using 'first' or 'mean'."
            )
        return unique_field

    def separate(self, unpacked_field):
        """
        Extract the separated representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : np.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)

        Returns
        -------
        separated_field : list of np.ndarray
            The separated representation. Every array is of shape :
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)
        """
        field_shape = unpacked_field.shape[:-1]
        separated_field = []
        ind = 0
        for patch_shape in self.shape_by_patch:
            next_ind = ind + np.prod(patch_shape)
            separated_field.append(
                unpacked_field[..., ind:next_ind].reshape((*field_shape, *patch_shape))
            )
            ind = next_ind
        return separated_field

    def agglomerate(self, separated_field):
        """
        Extract the unpacked representation from a separated representation.

        Parameters
        ----------
        separated_field : list of np.ndarray
            The separated representation. Every array is of shape :
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)

        Returns
        -------
        unpacked_field : np.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        field_shape = separated_field[0].shape[: -self.npa]
        assert np.all(
            [f.shape[: -self.npa] == field_shape for f in separated_field]
        ), "Every patch must have the same field shape !"
        unpacked_field = np.concatenate(
            [f.reshape((*field_shape, -1)) for f in separated_field], axis=-1
        )
        return unpacked_field

    def unique_field_indices(self, field_shape, representation="separated"):
        """
        Get the unique, unpacked or separated representation of a field's unique indices.

        Parameters
        ----------
        field_shape : tuple of int
            The shape of the field. For example, if it is a vector field, `field_shape`
            should be (3,). If it is a second order tensor field, it should be (3, 3).
        representation : str, optional
            The user must choose between `"unique"`, `"unpacked"`, and `"separated"`.
            It corresponds to the type of representation to get, by default "separated"

        Returns
        -------
        unique_field_indices : np.ndarray[np.integer] or list of np.ndarray[np.integer]
            The unique, unpacked or separated representation of a field's unique indices.
            If unique, its shape is (*`field_shape`, `self`.`nb_unique_nodes`).
            If unpacked, its shape is : (*`field_shape`, `self`.`nb_nodes`).
            If separated, every array is of shape : (*`field_shape`, nb elem for dim 1, ..., nb elem for dim `npa`).
        """
        nb_indices = np.prod(field_shape) * self.nb_unique_nodes
        unique_field_indices_as_unique_field = np.arange(
            nb_indices, dtype="int"
        ).reshape((*field_shape, self.nb_unique_nodes))
        if representation == "unique":
            unique_field_indices = unique_field_indices_as_unique_field
            return unique_field_indices
        elif representation == "unpacked":
            unique_field_indices = self.unpack(unique_field_indices_as_unique_field)
            return unique_field_indices
        elif representation == "separated":
            unique_field_indices = self.separate(
                self.unpack(unique_field_indices_as_unique_field)
            )
            return unique_field_indices
        else:
            raise ValueError(
                f'Representation "{representation}" not recognised. Representation must either be "unique", "unpacked", or "separated" !'
            )

    def get_duplicate_unpacked_nodes_mask(self):
        """
        Returns a boolean mask indicating which nodes in the unpacked representation are duplicates.

        Returns
        -------
        duplicate_nodes_mask : np.ndarray
            Boolean mask of shape (nb_nodes,) where True indicates a node is duplicated
            across multiple patches and False indicates it appears only once.
        """
        _, inverse, counts = np.unique(
            self.unique_nodes_inds, return_inverse=True, return_counts=True
        )
        duplicate_nodes_mask = counts[inverse] > 1
        return duplicate_nodes_mask

    def extract_exterior_borders(self, splines):
        """
        Extract exterior borders from B-spline patches.

        Parameters
        ----------
        splines : list[BSpline]
            Array of B-spline patches to extract borders from.

        Returns
        -------
        border_connectivity : MultiPatchBSplineConnectivity
            Connectivity information for the border patches.
        border_splines : list[BSpline]
            Array of B-spline patches representing the borders.
        border_unique_to_self_unique_connectivity : np.ndarray[np.integer]
            Array mapping border unique nodes to original unique nodes.

        Raises
        ------
        AssertionError
            If parametric space dimension is less than 2.
        """
        if self.npa <= 1:
            raise AssertionError(
                "The parametric space must be at least 2D to extract borders !"
            )
        duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
        duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
        separated_unique_nodes_inds = self.unique_field_indices(())
        arr = np.arange(self.npa).tolist()
        border_splines = []
        border_unique_nodes_inds = []
        border_shape_by_patch = []
        for i in range(self.nb_patchs):
            spline = splines[i]
            duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
            unique_nodes_inds_spline = separated_unique_nodes_inds[i]
            shape_by_patch_spline = self.shape_by_patch[i]
            for axis in range(self.npa):
                bases = np.hstack((spline.bases[(axis + 1) :], spline.bases[:axis]))
                axes = arr[axis:-1] + arr[:axis]
                border_shape_by_patch_spline = np.hstack(
                    (shape_by_patch_spline[(axis + 1) :], shape_by_patch_spline[:axis])
                )
                if not np.take(duplicate_nodes_mask_spline, 0, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases[::-1])
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = (
                        np.take(unique_nodes_inds_spline, 0, axis=axis)
                        .transpose(axes[::-1])
                        .ravel()
                    )
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[
                        ::-1
                    ][None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {0} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
                if not np.take(duplicate_nodes_mask_spline, -1, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases)
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = (
                        np.take(unique_nodes_inds_spline, -1, axis=axis)
                        .transpose(axes)
                        .ravel()
                    )
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[
                        None
                    ]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {-1} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
        border_splines = np.array(border_splines, dtype="object")
        border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
        border_shape_by_patch = np.concatenate(border_shape_by_patch)
        border_unique_to_self_unique_connectivity, inverse = np.unique(
            border_unique_nodes_inds, return_inverse=True
        )
        border_unique_nodes_inds -= np.cumsum(
            np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity)))
            - 1
        )[inverse]
        border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
        border_connectivity = self.__class__(
            border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes
        )
        return (
            border_connectivity,
            border_splines,
            border_unique_to_self_unique_connectivity,
        )

    def extract_interior_borders(self, splines):
        """
        Extract interior borders from B-spline patches where nodes are shared between patches.

        Parameters
        ----------
        splines : list[BSpline]
            Array of B-spline patches to extract borders from.

        Returns
        -------
        border_connectivity : MultiPatchBSplineConnectivity
            Connectivity information for the border patches.
        border_splines : list[BSpline]
            Array of B-spline patches representing the borders.
        border_unique_to_self_unique_connectivity : np.ndarray[np.integer]
            Array mapping border unique nodes to original unique nodes.

        Raises
        ------
        AssertionError
            If parametric space dimension is less than 2.
        """
        if self.npa <= 1:
            raise AssertionError(
                "The parametric space must be at least 2D to extract borders !"
            )
        duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
        duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
        separated_unique_nodes_inds = self.unique_field_indices(())
        arr = np.arange(self.npa).tolist()
        border_splines = []
        border_unique_nodes_inds = []
        border_shape_by_patch = []
        for i in range(self.nb_patchs):
            spline = splines[i]
            duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
            unique_nodes_inds_spline = separated_unique_nodes_inds[i]
            shape_by_patch_spline = self.shape_by_patch[i]
            for axis in range(self.npa):
                bases = np.hstack((spline.bases[(axis + 1) :], spline.bases[:axis]))
                axes = arr[axis:-1] + arr[:axis]
                border_shape_by_patch_spline = np.hstack(
                    (shape_by_patch_spline[(axis + 1) :], shape_by_patch_spline[:axis])
                )
                if np.take(duplicate_nodes_mask_spline, 0, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases[::-1])
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = (
                        np.take(unique_nodes_inds_spline, 0, axis=axis)
                        .transpose(axes[::-1])
                        .ravel()
                    )
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[
                        ::-1
                    ][None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {0} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
                if np.take(duplicate_nodes_mask_spline, -1, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases)
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = (
                        np.take(unique_nodes_inds_spline, -1, axis=axis)
                        .transpose(axes)
                        .ravel()
                    )
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[
                        None
                    ]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {-1} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
        border_splines = np.array(border_splines, dtype="object")
        border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
        border_shape_by_patch = np.concatenate(border_shape_by_patch)
        border_unique_to_self_unique_connectivity, inverse = np.unique(
            border_unique_nodes_inds, return_inverse=True
        )
        border_unique_nodes_inds -= np.cumsum(
            np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity)))
            - 1
        )[inverse]
        border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
        border_connectivity = self.__class__(
            border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes
        )
        return (
            border_connectivity,
            border_splines,
            border_unique_to_self_unique_connectivity,
        )

    # def extract_exterior_surfaces(self, splines):
    #     if self.npa!=3:
    #         raise AssertionError("The parametric space must be 3D to extract surfaces !")
    #     duplicate_unpacked_nodes_mask = self.get_duplicate_unpacked_nodes_mask()
    #     duplicate_separated_nodes_mask = self.separate(duplicate_unpacked_nodes_mask)
    #     separated_unique_nodes_inds = self.unique_field_indices(())
    #     arr = np.arange(self.npa).tolist()
    #     border_splines = []
    #     border_unique_nodes_inds = []
    #     border_shape_by_patch = []
    #     for i in range(self.nb_patchs):
    #         spline = splines[i]
    #         duplicate_nodes_mask_spline = duplicate_separated_nodes_mask[i]
    #         unique_nodes_inds_spline = separated_unique_nodes_inds[i]
    #         shape_by_patch_spline = self.shape_by_patch[i]
    #
    #         # surface 1
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=0).all():
    #             bspline_border = BSpline.from_bases(spline.bases[:0:-1])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=0).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[:0:-1][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 1 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 2
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=0).all():
    #             bspline_border = BSpline.from_bases(spline.bases[1:])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=0).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[1:][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 2 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 3
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=1).all():
    #             bspline_border = BSpline.from_bases(spline.bases[::2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=1).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[::2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 3 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 4
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=1).all():
    #             bspline_border = BSpline.from_bases(spline.bases[::-2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=1).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[::-2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 4 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 5
    #         if not np.take(duplicate_nodes_mask_spline, 0, axis=2).all():
    #             bspline_border = BSpline.from_bases(spline.bases[1::-1])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=2).T.ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[1::-1][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 5 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #         # surface 6
    #         if not np.take(duplicate_nodes_mask_spline, -1, axis=2).all():
    #             bspline_border = BSpline.from_bases(spline.bases[:2])
    #             border_splines.append(bspline_border)
    #             unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=2).ravel()
    #             border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
    #             border_shape_by_patch_spline = shape_by_patch_spline[:2][None]
    #             border_shape_by_patch.append(border_shape_by_patch_spline)
    #             print(f"Surface 6 of patch {i} uses nodes {unique_nodes_inds_spline_border}")
    #     border_splines = np.array(border_splines, dtype='object')
    #     border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
    #     border_shape_by_patch = np.concatenate(border_shape_by_patch)
    #     border_unique_to_self_unique_connectivity, inverse = np.unique(border_unique_nodes_inds, return_inverse=True)
    #     border_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity))) - 1)[inverse]
    #     border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
    #     border_connectivity = self.__class__(border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes)
    #     return border_connectivity, border_splines, border_unique_to_self_unique_connectivity

    def subset(self, splines, patches_to_keep):
        """
        Create a subset of the multi-patch B-spline connectivity by keeping only selected patches.

        Parameters
        ----------
        splines : list[BSpline]
            Array of B-spline patches to subset.
        patches_to_keep : np.ndarray[np.integer]
            Indices of patches to keep in the subset.

        Returns
        -------
        new_connectivity : MultiPatchBSplineConnectivity
            New connectivity object containing only the selected patches.
        new_splines : list[BSpline]
            Array of B-spline patches for the selected patches.
        new_unique_to_self_unique_connectivity : np.ndarray[np.integer]
            Array mapping new unique nodes to original unique nodes.
        """
        new_splines = splines[patches_to_keep]
        separated_unique_nodes_inds = self.unique_field_indices(())
        new_unique_nodes_inds = np.concatenate(
            [separated_unique_nodes_inds[patch].flat for patch in patches_to_keep]
        )
        new_shape_by_patch = self.shape_by_patch[patches_to_keep]
        new_unique_to_self_unique_connectivity, inverse = np.unique(
            new_unique_nodes_inds, return_inverse=True
        )
        new_unique_nodes_inds -= np.cumsum(
            np.diff(np.concatenate(([-1], new_unique_to_self_unique_connectivity))) - 1
        )[inverse]
        new_nb_unique_nodes = np.unique(new_unique_nodes_inds).size
        new_connectivity = self.__class__(
            new_unique_nodes_inds, new_shape_by_patch, new_nb_unique_nodes
        )
        return new_connectivity, new_splines, new_unique_to_self_unique_connectivity

    def make_control_poly_meshes(
        self,
        splines: Iterable[BSpline],
        separated_ctrl_pts: Iterable[np.ndarray[np.floating]],
        n_eval_per_elem: Union[Iterable[int], int] = 10,
        n_step: int = 1,
        unique_fields: dict = {},
        separated_fields: Union[dict, None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        paraview_sizes: dict = {},
    ) -> list[io.Mesh]:

        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem] * self.npa

        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]

        if XI_list is None:
            XI_list = [None] * self.nb_patchs

        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError(
                    "To handle functions as fields, use separated_fields !"
                )
            else:
                if value.shape[-1] != self.nb_unique_nodes:
                    raise ValueError(
                        f"Field {key} of unique_fields is in a wrong format (not unique format)."
                    )
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]

        step_meshes = np.empty((self.nb_patchs, n_step), dtype=object)
        for patch in range(self.nb_patchs):
            step_meshes[patch] = splines[patch].make_control_poly_meshes(
                separated_ctrl_pts[patch],
                n_eval_per_elem=n_eval_per_elem,
                n_step=n_step,
                fields=separated_fields[patch],
                XI=XI_list[patch],
                paraview_sizes=paraview_sizes,
            )
            for mesh in step_meshes[patch]:
                mesh.point_data["patch_id"] = np.full(
                    mesh.points.shape[0], patch, dtype=int
                )
        meshes = []
        for time_step_meshes in step_meshes.T:
            meshes.append(merge_meshes(time_step_meshes))
        return meshes

    def make_elem_separator_meshes(
        self,
        splines: Iterable[BSpline],
        separated_ctrl_pts: Iterable[np.ndarray[np.floating]],
        n_eval_per_elem: Union[Iterable[int], int] = 10,
        n_step: int = 1,
        unique_fields: dict = {},
        separated_fields: Union[dict, None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        paraview_sizes: dict = {},
        parallel: bool = True,
        verbose: bool = True,
    ) -> list[io.Mesh]:

        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem] * self.npa

        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]

        if XI_list is None:
            XI_list = [None] * self.nb_patchs

        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError(
                    "To handle functions as fields, use separated_fields !"
                )
            else:
                if value.shape[-1] != self.nb_unique_nodes:
                    raise ValueError(
                        f"Field {key} of unique_fields is in a wrong format (not unique format)."
                    )
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]

        funcs = [splines[i].make_elem_separator_meshes for i in range(self.nb_patchs)]
        all_args = [
            (
                separated_ctrl_pts[i],
                n_eval_per_elem,
                n_step,
                separated_fields[i],
                XI_list[i],
                paraview_sizes,
            )
            for i in range(self.nb_patchs)
        ]

        elem_separator_meshes = parallel_blocks(
            funcs,
            all_args,
            verbose=verbose,
            pbar_title="Making elements separators meshes",
            disable_parallel=not parallel,
        )

        for patch, step_meshes in enumerate(elem_separator_meshes):
            for timestep, mesh in enumerate(step_meshes):
                mesh.point_data["patch_id"] = np.full(
                    mesh.points.shape[0], patch, dtype=int
                )
        meshes = []
        for timestep, patch_meshes in enumerate(zip(*elem_separator_meshes)):
            meshes.append(merge_meshes(patch_meshes))

        return meshes

    def make_elements_interior_meshes(
        self,
        splines: Iterable[BSpline],
        separated_ctrl_pts: Iterable[np.ndarray[np.floating]],
        n_eval_per_elem: Union[Iterable[int], int] = 10,
        n_step: int = 1,
        unique_fields: dict = {},
        separated_fields: Union[dict, None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        parallel: bool = True,
        verbose: bool = True,
    ) -> list[io.Mesh]:
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem] * self.npa

        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]

        if XI_list is None:
            XI_list = [None] * self.nb_patchs

        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError(
                    "To handle functions as fields, use separated_fields !"
                )
            else:
                if value.shape[-1] != self.nb_unique_nodes:
                    raise ValueError(
                        f"Field {key} of unique_fields is in a wrong format (not unique format)."
                    )
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]

        funcs = [
            splines[i].make_elements_interior_meshes for i in range(self.nb_patchs)
        ]
        all_args = [
            (
                separated_ctrl_pts[i],
                n_eval_per_elem,
                n_step,
                separated_fields[i],
                XI_list[i],
            )
            for i in range(self.nb_patchs)
        ]

        interior_meshes = parallel_blocks(
            funcs,
            all_args,
            verbose=verbose,
            pbar_title="Making interior meshes",
            disable_parallel=not parallel,
        )

        for patch, step_meshes in enumerate(interior_meshes):
            for timestep, mesh in enumerate(step_meshes):
                mesh.point_data["patch_id"] = np.full(
                    mesh.points.shape[0], patch, dtype=int
                )
        meshes = []
        for timestep, patch_meshes in enumerate(zip(*interior_meshes)):
            meshes.append(merge_meshes(patch_meshes))

        return meshes

    def make_all_meshes(
        self,
        splines: Iterable[BSpline],
        separated_ctrl_pts: Iterable[np.ndarray[np.floating]],
        n_step: int = 1,
        n_eval_per_elem: Union[int, Iterable[int]] = 10,
        unique_fields: dict = {},
        separated_fields: Union[list[dict], None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        verbose: bool = True,
        fields_on_interior_only: Union[bool, Literal["auto"], list[str]] = "auto",
        disable_parallel: bool = False,
    ) -> tuple[list[io.Mesh], list[io.Mesh], list[io.Mesh]]:
        """
        Generate all mesh representations (interior, element borders, and control points) for a multipatch B-spline geometry.

        This method creates three types of meshes for visualization or analysis:
        - The interior mesh representing the B-spline surface or volume.
        - The element separator mesh showing the borders between elements.
        - The control polygon mesh showing the control structure.

        Parameters
        ----------
        splines : Iterable[BSpline]
            List of B-spline patches to process.
        separated_ctrl_pts : Iterable[np.ndarray[np.floating]]
            Control points for each patch in separated representation.
        n_step : int, optional
            Number of time steps to generate. By default, 1.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each parametric dimension.
            If an `int` is provided, the same number is used for all dimensions.
            If an `Iterable` is provided, each value corresponds to a different dimension.
            By default, 10.
        unique_fields : dict, optional
            Fields in unique representation to visualize. By default, `{}`.
            Keys are field names, values are arrays (not callables nor FE fields).
        separated_fields : Union[list[dict], None], optional
            Fields to visualize at each time step.
            List of `self.nb_patchs` dictionaries (one per patch) of format:
            {
                "field_name": `field_value`
            }
            where `field_value` can be either:
            1. A `np.ndarray` with shape (`n_step`, `field_size`, `self.shape_by_patch[patch]`)
            2. A `np.ndarray` with shape (`n_step`, `field_size`, `*grid_shape`)
            3. A function that computes field values (`np.ndarray[np.floating]`) at given points from the `BSpline` instance and `XI`.
            By default, None.
        XI_list : Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]], optional
            Parametric coordinates at which to evaluate the B-spline patches and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, regular grids are generated according to `n_eval_per_elem`.
            By default, None.
        verbose : bool, optional
            Whether to print progress information. By default, True.
        fiels_on_interior_only: Union[bool, Literal['auto'], list[str]], optionnal
            Whether to include fields only on the interior mesh (`True`), on all meshes (`False`),
            or on specified field names.
            If set to `'auto'`, fields named `'u'`, `'U'`, `'displacement'` or `'displ'`
            are included on all meshes while others are only included on the interior mesh.
            By default, 'auto'.
        disable_parallel : bool, optional
            Wether to disable parallel execution. By default, False.

        Returns
        -------
        tuple[list[io.Mesh], list[io.Mesh], list[io.Mesh]]
            Tuple containing three lists of `io.Mesh` objects:
            - Interior meshes for each time step.
            - Element separator meshes for each time step.
            - Control polygon meshes for each time step.

        Raises
        ------
        NotImplementedError
            If a callable is passed in `unique_fields`.
        ValueError
            If a field in `unique_fields` does not have the correct shape.

        Notes
        -----
        - Fields can be visualized as scalars or vectors.
        - Supports time-dependent visualization through `n_step`.
        - Fields in `unique_fields` must be arrays; to use callables, use `separated_fields`.

        Examples
        --------
        >>> interior, borders, control = connectivity.make_all_meshes(splines, separated_ctrl_pts)
        """

        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem] * self.npa

        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]

        if XI_list is None:
            XI_list = [None] * self.nb_patchs

        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError(
                    "To handle functions as fields, use separated_fields !"
                )
            else:
                if value.shape[-1] != self.nb_unique_nodes:
                    raise ValueError(
                        f"Field {key} of unique_fields is in a wrong format (not unique format)."
                    )
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]

        paraview_sizes = {}
        fields = separated_fields[0]
        if fields_on_interior_only is True:
            for key, value in fields.items():
                if callable(value):
                    paraview_sizes[key] = value(
                        splines[0], np.zeros((splines[0].NPa, 1))
                    ).shape[2]
                else:
                    paraview_sizes[key] = value.shape[1]
        elif fields_on_interior_only is False:
            pass
        elif fields_on_interior_only == "auto":
            for key, value in fields.items():
                if key not in ["u", "U", "displacement", "displ"]:
                    if callable(value):
                        paraview_sizes[key] = value(
                            splines[0], np.zeros((splines[0].NPa, 1))
                        ).shape[2]
                    else:
                        paraview_sizes[key] = value.shape[1]
        else:
            for key in fields_on_interior_only:
                value = fields[key]
                if callable(value):
                    paraview_sizes[key] = value(
                        splines[0], np.zeros((splines[0].NPa, 1))
                    ).shape[2]
                else:
                    paraview_sizes[key] = value.shape[1]

        elem_interior_meshes = self.make_elements_interior_meshes(
            splines,
            separated_ctrl_pts,
            n_eval_per_elem,
            n_step,
            separated_fields=separated_fields,
            XI_list=XI_list,
            verbose=verbose,
            parallel=(not disable_parallel),
        )
        if verbose:
            print("interior done")

        elem_separator_meshes = self.make_elem_separator_meshes(
            splines,
            separated_ctrl_pts,
            n_eval_per_elem,
            n_step,
            separated_fields=separated_fields,
            XI_list=XI_list,
            paraview_sizes=paraview_sizes,
            parallel=(not disable_parallel),
        )
        if verbose:
            print("elements borders done")

        control_poly_meshes = self.make_control_poly_meshes(
            splines,
            separated_ctrl_pts,
            n_eval_per_elem,
            n_step,
            separated_fields=separated_fields,
            XI_list=XI_list,
            paraview_sizes=paraview_sizes,
        )
        if verbose:
            print("control points done")

        return elem_interior_meshes, elem_separator_meshes, control_poly_meshes

    def save_paraview(
        self,
        splines: Iterable[BSpline],
        separated_ctrl_pts: Iterable[np.ndarray[np.floating]],
        path: str,
        name: str,
        n_step: int = 1,
        n_eval_per_elem: Union[int, Iterable[int]] = 10,
        unique_fields: dict = {},
        separated_fields: Union[list[dict], None] = None,
        XI_list: Union[None, Iterable[tuple[np.ndarray[np.floating], ...]]] = None,
        groups: Union[dict[str, dict[str, Union[str, int]]], None] = None,
        make_pvd: bool = True,
        verbose: bool = True,
        fields_on_interior_only: Union[bool, Literal["auto"], list[str]] = "auto",
        disable_parallel: bool = False,
    ):
        """
        Save multipatch B-spline visualization data as Paraview files.

        This method generates three types of visualization files for a multipatch B-spline geometry:
        - Interior mesh showing the B-spline surface/volume
        - Element borders showing the mesh structure
        - Control points mesh showing the control structure

        All files are saved in VTU format, with an optional PVD file to group them for Paraview.

        Parameters
        ----------
        splines : Iterable[BSpline]
            List of B-spline patches to save.
        separated_ctrl_pts : Iterable[np.ndarray[np.floating]]
            Control points for each patch in separated representation.
        path : str
            Directory path where the files will be saved.
        name : str
            Base name for the output files.
        n_step : int, optional
            Number of time steps to save. By default, 1.
        n_eval_per_elem : Union[int, Iterable[int]], optional
            Number of evaluation points per element for each parametric dimension.
            If an `int` is provided, the same number is used for all dimensions.
            If an `Iterable` is provided, each value corresponds to a different dimension.
            By default, 10.
        unique_fields : dict, optional
            Fields in unique representation to save. By default, `{}`.
            Keys are field names, values are arrays (not callables nor FE fields).
        separated_fields : Union[list[dict], None], optional
            Fields to visualize at each time step.
            List of `self.nb_patchs` dictionaries (one per patch) of format:
            {
                "field_name": `field_value`
            }
            where `field_value` can be either:

            1. A numpy array with shape (`n_step`, `field_size`, `self.shape_by_patch[patch]`) where:
            - `n_step`: Number of time steps
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)
            - `self.shape_by_patch[patch]`: Same shape as the patch's control points grid (excluding `NPh`)

            2. A numpy array with shape (`n_step`, `field_size`, `*grid_shape`) where:
            - `n_step`: Number of time steps
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)
            - `*grid_shape`: Shape of the evaluation grid (number of points along each parametric axis)

            3. A function that computes field values (`np.ndarray[np.floating]`) at given
            points from the `BSpline` instance and `XI`, the tuple of arrays containing evaluation
            points for each dimension (`tuple[np.ndarray[np.floating], ...]`).
            The result should be an array of shape (`n_step`, `n_points`, `field_size`) where:
            - `n_step`: Number of time steps
            - `n_points`: Number of evaluation points (n_xi × n_eta × ...)
            - `field_size`: Size of the field at each point (1 for scalar, 3 for vector)

            By default, None.
        XI_list : Iterable[tuple[np.ndarray[np.floating], ...]], optional
            Parametric coordinates at which to evaluate the B-spline patches and fields.
            If not `None`, overrides the `n_eval_per_elem` parameter.
            If `None`, regular grids are generated according to `n_eval_per_elem`.
        groups : Union[dict[str, dict[str, Union[str, int]]], None], optional
            Nested dictionary specifying file groups for PVD organization. Format:
            {
                "group_name": {
                    "ext": str,     # File extension (e.g., "vtu")
                    "npart": int,   # Number of parts in the group
                    "nstep": int    # Number of timesteps
                }
            }
            If provided, existing groups are updated; if `None`, groups are created automatically.
            By default, `None`.
        make_pvd : bool, optional
            Whether to create a PVD file grouping all VTU files. By default, `True`.
        verbose : bool, optional
            Whether to print progress information. By default, `True`.
        fields_on_interior_only: Union[bool, Literal['auto'], list[str]], optionnal
            Whether to include fields only on the interior mesh (`True`), on all meshes (`False`),
            or on specified field names.
            If set to `'auto'`, fields named `'u'`, `'U'`, `'displacement'` or `'displ'`
            are included on all meshes while others are only included on the interior mesh.
            By default, 'auto'.
        disable_parallel : bool, optional
            Whether to disable the parallel execution. By default, False.

        Returns
        -------
        groups : dict[str, dict[str, Union[str, int]]]
            Updated groups dictionary with information about saved files.

        Raises
        ------
        NotImplementedError
            If a callable is passed in `unique_fields`.
        ValueError
            If the multiprocessing pool is not running and cannot be restarted.

        Notes
        -----
        - Creates three types of VTU files for each time step:
            - {name}_interior_{part}_{step}.vtu
            - {name}_elements_borders_{part}_{step}.vtu
            - {name}_control_points_{part}_{step}.vtu
        - If `make_pvd=True`, creates a PVD file named {name}.pvd.
        - Fields can be visualized as scalars or vectors in Paraview.
        - The method supports time-dependent visualization through `n_step`.
        - Fields in `unique_fields` must be arrays; to use callables, use `separated_fields`.

        Examples
        --------
        Save a multipatch B-spline visualization:
        >>> connectivity.save_paraview(splines, separated_ctrl_pts, "./output", "multipatch")

        Save with a custom separated field on a 2 patches multipatch:
        >>> fields = [{"temperature": np.random.rand(1, 4, 4)}, {"temperature": np.random.rand(1, 7, 3)}]
        >>> connectivity.save_paraview(splines, separated_ctrl_pts, "./output", "multipatch", separated_fields=fields)
        """

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

        elem_interior_meshes, elem_separator_meshes, control_poly_meshes = (
            self.make_all_meshes(
                splines,
                separated_ctrl_pts,
                n_step,
                n_eval_per_elem,
                unique_fields,
                separated_fields,
                XI_list,
                verbose,
                fields_on_interior_only,
                disable_parallel=disable_parallel,
            )
        )

        prefix = os.path.join(
            path, f"{name}_{interior}_{groups[interior]['npart'] - 1}"
        )
        for time_step, mesh in enumerate(elem_interior_meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(interior, "saved")

        prefix = os.path.join(
            path, f"{name}_{elements_borders}_{groups[elements_borders]['npart'] - 1}"
        )
        for time_step, mesh in enumerate(elem_separator_meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(elements_borders, "saved")

        prefix = os.path.join(
            path, f"{name}_{control_points}_{groups[control_points]['npart'] - 1}"
        )
        for time_step, mesh in enumerate(control_poly_meshes):
            mesh.write(f"{prefix}_{time_step}.vtu")
        if verbose:
            print(control_points, "saved")

        if make_pvd:
            writePVD(os.path.join(path, name), groups)

        return groups


#     def save_YETI(self, splines, separated_ctrl_pts, path, name):
#         if self.npa==2:
#             el_type = "U3"
#         elif self.npa==3:
#             el_type = "U1"
#         else:
#             raise NotImplementedError("Can only save surfaces or volumes !")
#         objects_list = []
#         for patch in range(self.nb_patchs):
#             geomdl_patch = splines[patch].getGeomdl(separated_ctrl_pts[patch])
#             obj = Domain.DefaultDomain(geometry=geomdl_patch,
#                                        id_dom=patch,
#                                        elem_type=el_type)
#             objects_list.append(obj)
#         write.write_files(objects_list, os.path.join(path, name))


# %%
class CouplesBSplineBorder:

    def __init__(
        self,
        spline1_inds,
        spline2_inds,
        axes1,
        axes2,
        front_sides1,
        front_sides2,
        transpose_2_to_1,
        flip_2_to_1,
        NPa,
    ):
        self.spline1_inds = spline1_inds
        self.spline2_inds = spline2_inds
        self.axes1 = axes1
        self.axes2 = axes2
        self.front_sides1 = front_sides1
        self.front_sides2 = front_sides2
        self.transpose_2_to_1 = transpose_2_to_1
        self.flip_2_to_1 = flip_2_to_1
        self.NPa = NPa
        self.nb_couples = self.flip_2_to_1.shape[0]

    @classmethod
    def extract_border_pts(cls, field, axis, front_side, field_dim=1, offset=0):
        npa = field.ndim - field_dim
        base_face = np.hstack((np.arange(axis + 1, npa), np.arange(axis)))
        if not front_side:
            base_face = base_face[::-1]
        border_field = field.transpose(
            axis + field_dim, *np.arange(field_dim), *(base_face + field_dim)
        )[(-(1 + offset) if front_side else offset)]
        return border_field

    @classmethod
    def extract_border_spline(cls, spline, axis, front_side):
        base_face = np.hstack((np.arange(axis + 1, spline.NPa), np.arange(axis)))
        if not front_side:
            base_face = base_face[::-1]
        degrees = spline.getDegrees()
        knots = spline.getKnots()
        border_degrees = [degrees[i] for i in base_face]
        border_knots = [knots[i] for i in base_face]
        border_spline = BSpline(border_degrees, border_knots)
        return border_spline

    @classmethod
    def transpose_and_flip(cls, field, transpose, flip, field_dim=1):
        field = field.transpose(*np.arange(field_dim), *(transpose + field_dim))
        for i in range(flip.size):
            if flip[i]:
                field = np.flip(field, axis=(i + field_dim))
        return field

    @classmethod
    def transpose_and_flip_knots(cls, knots, spans, transpose, flip):
        new_knots = []
        for i in range(flip.size):
            if flip[i]:
                new_knots.append(sum(spans[i]) - knots[transpose[i]][::-1])
            else:
                new_knots.append(knots[transpose[i]])
        return new_knots

    @classmethod
    def transpose_and_flip_back_knots(cls, knots, spans, transpose, flip):
        transpose_back = np.argsort(transpose)
        flip_back = flip[transpose_back]
        return cls.transpose_and_flip_knots(knots, spans, transpose_back, flip_back)

    @classmethod
    def transpose_and_flip_spline(cls, spline, transpose, flip):
        spans = spline.getSpans()
        knots = spline.getKnots()
        degrees = spline.getDegrees()
        for i in range(flip.size):
            p = degrees[transpose[i]]
            knot = knots[transpose[i]]
            if flip[i]:
                knot = sum(spans[i]) - knot[::-1]
            spline.bases[i] = BSplineBasis(p, knot)
        return spline

    @classmethod
    def from_splines(cls, separated_ctrl_pts, splines):
        NPa = splines[0].NPa
        assert np.all(
            [sp.NPa == NPa for sp in splines]
        ), "Every patch should have the same parametric space dimension !"
        NPh = separated_ctrl_pts[0].shape[0]
        assert np.all(
            [ctrl_pts.shape[0] == NPh for ctrl_pts in separated_ctrl_pts]
        ), "Every patch should have the same physical space dimension !"
        npatch = len(splines)
        all_flip = np.unpackbits(
            np.arange(2 ** (NPa - 1), dtype="uint8")[:, None],
            axis=1,
            count=(NPa - 1 - 8),
            bitorder="little",
        )[:, ::-1].astype("bool")
        all_transpose = np.array(list(permutations(np.arange(NPa - 1))))
        spline1_inds = []
        spline2_inds = []
        axes1 = []
        axes2 = []
        front_sides1 = []
        front_sides2 = []
        transpose_2_to_1 = []
        flip_2_to_1 = []
        for spline1_ind in range(npatch):
            spline1 = splines[spline1_ind]
            ctrl_pts1 = separated_ctrl_pts[spline1_ind]
            # print(f"sp1 {spline1_ind}")
            for spline2_ind in range(spline1_ind + 1, npatch):
                spline2 = splines[spline2_ind]
                ctrl_pts2 = separated_ctrl_pts[spline2_ind]
                # print(f"|sp2 {spline2_ind}")
                for axis1 in range(spline1.NPa):
                    degrees1 = np.hstack(
                        (
                            spline1.getDegrees()[(axis1 + 1) :],
                            spline1.getDegrees()[:axis1],
                        )
                    )
                    knots1 = (
                        spline1.getKnots()[(axis1 + 1) :] + spline1.getKnots()[:axis1]
                    )
                    # print(f"||ax1 {axis1}")
                    for axis2 in range(spline2.NPa):
                        degrees2 = np.hstack(
                            (
                                spline2.getDegrees()[(axis2 + 1) :],
                                spline2.getDegrees()[:axis2],
                            )
                        )
                        knots2 = (
                            spline2.getKnots()[(axis2 + 1) :]
                            + spline2.getKnots()[:axis2]
                        )
                        spans2 = (
                            spline2.getSpans()[(axis2 + 1) :]
                            + spline2.getSpans()[:axis2]
                        )
                        # print(f"|||ax2 {axis2}")
                        for front_side1 in [False, True]:
                            pts1 = cls.extract_border_pts(ctrl_pts1, axis1, front_side1)
                            # print(f"||||{'front' if front_side1 else 'back '} side1")
                            for front_side2 in [False, True]:
                                pts2 = cls.extract_border_pts(
                                    ctrl_pts2, axis2, front_side2
                                )
                                # print(f"|||||{'front' if front_side2 else 'back '} side2")
                                for transpose in all_transpose:
                                    # print(f"||||||transpose {transpose}")
                                    if (
                                        degrees1 == [degrees2[i] for i in transpose]
                                    ).all():
                                        # print(f"||||||same degrees {degrees1}")
                                        if list(pts1.shape[1:]) == [
                                            pts2.shape[1:][i] for i in transpose
                                        ]:
                                            # print(f"||||||same shapes {pts1.shape[1:]}")
                                            if np.all(
                                                [
                                                    knots1[i].size
                                                    == knots2[transpose[i]].size
                                                    for i in range(NPa - 1)
                                                ]
                                            ):
                                                # print(f"||||||same knots sizes {[knots1[i].size for i in range(NPa - 1)]}")
                                                for flip in all_flip:
                                                    # print(f"|||||||flip {flip}")
                                                    if np.all(
                                                        [
                                                            (k1 == k2).all()
                                                            for k1, k2 in zip(
                                                                knots1,
                                                                cls.transpose_and_flip_knots(
                                                                    knots2,
                                                                    spans2,
                                                                    transpose,
                                                                    flip,
                                                                ),
                                                            )
                                                        ]
                                                    ):
                                                        # print(f"|||||||same knots {knots1}")
                                                        pts2_turned = (
                                                            cls.transpose_and_flip(
                                                                pts2, transpose, flip
                                                            )
                                                        )
                                                        if np.allclose(
                                                            pts1, pts2_turned
                                                        ):
                                                            # print("_________________GOGOGO_________________")
                                                            spline1_inds.append(
                                                                spline1_ind
                                                            )
                                                            spline2_inds.append(
                                                                spline2_ind
                                                            )
                                                            axes1.append(axis1)
                                                            axes2.append(axis2)
                                                            front_sides1.append(
                                                                front_side1
                                                            )
                                                            front_sides2.append(
                                                                front_side2
                                                            )
                                                            transpose_2_to_1.append(
                                                                transpose
                                                            )
                                                            flip_2_to_1.append(flip)
        spline1_inds = np.array(spline1_inds, dtype="int")
        spline2_inds = np.array(spline2_inds, dtype="int")
        axes1 = np.array(axes1, dtype="int")
        axes2 = np.array(axes2, dtype="int")
        front_sides1 = np.array(front_sides1, dtype="bool")
        front_sides2 = np.array(front_sides2, dtype="bool")
        transpose_2_to_1 = np.array(transpose_2_to_1, dtype="int")
        flip_2_to_1 = np.array(flip_2_to_1, dtype="bool")
        return cls(
            spline1_inds,
            spline2_inds,
            axes1,
            axes2,
            front_sides1,
            front_sides2,
            transpose_2_to_1,
            flip_2_to_1,
            NPa,
        )

    def append(self, other):
        if self.NPa != other.NPa:
            raise ValueError(
                f"operands could not be concatenated with parametric spaces of dimensions {self.NPa} and {other.NPa}"
            )
        self.spline1_inds = np.concatenate(
            (self.spline1_inds, other.spline1_inds), axis=0
        )
        self.spline2_inds = np.concatenate(
            (self.spline2_inds, other.spline2_inds), axis=0
        )
        self.axes1 = np.concatenate((self.axes1, other.axes1), axis=0)
        self.axes2 = np.concatenate((self.axes2, other.axes2), axis=0)
        self.front_sides1 = np.concatenate(
            (self.front_sides1, other.front_sides1), axis=0
        )
        self.front_sides2 = np.concatenate(
            (self.front_sides2, other.front_sides2), axis=0
        )
        self.transpose_2_to_1 = np.concatenate(
            (self.transpose_2_to_1, other.transpose_2_to_1), axis=0
        )
        self.flip_2_to_1 = np.concatenate((self.flip_2_to_1, other.flip_2_to_1), axis=0)
        self.nb_couples += other.nb_couples

    def get_operator_allxi1_to_allxi2(self, spans1, spans2, couple_ind):
        ax1 = self.axes1[couple_ind]
        ax2 = self.axes2[couple_ind]
        front1 = self.front_sides1[couple_ind]
        front2 = self.front_sides2[couple_ind]
        transpose = self.transpose_2_to_1[couple_ind]
        flip = self.flip_2_to_1[couple_ind]

        A = np.zeros((self.NPa, self.NPa), dtype="float")
        A[ax2, ax1] = -1 if front1 == front2 else 1
        arr = np.arange(self.NPa)
        j1 = np.hstack((arr[(ax1 + 1) :], arr[:ax1]))
        j2 = np.hstack((arr[(ax2 + 1) :], arr[:ax2]))
        A[j2[transpose], j1] = [-1 if f else 1 for f in flip]
        b = np.zeros(self.NPa, dtype="float")
        b[ax2] = (int(front1) + int(front2)) * (1 if front2 else -1)
        b[j2[transpose]] = [1 if f else 0 for f in flip]

        alpha1, beta1 = np.array(spans1).T
        M1, p1 = np.diag(1 / (beta1 - alpha1)), -alpha1 / (beta1 - alpha1)
        alpha2, beta2 = np.array(spans2).T
        M2, p2 = np.diag(beta2 - alpha2), alpha2
        b = p2 + M2 @ b + M2 @ A @ p1
        A = M2 @ A @ M1

        return A, b

    def get_connectivity(self, shape_by_patch):
        indices = []
        start = 0
        for shape in shape_by_patch:
            end = start + np.prod(shape)
            indices.append(np.arange(start, end).reshape(shape))
            start = end
        nodes_couples = []
        for i in range(self.nb_couples):
            border_inds1 = self.__class__.extract_border_pts(
                indices[self.spline1_inds[i]],
                self.axes1[i],
                self.front_sides1[i],
                field_dim=0,
            )
            border_inds2 = self.__class__.extract_border_pts(
                indices[self.spline2_inds[i]],
                self.axes2[i],
                self.front_sides2[i],
                field_dim=0,
            )
            border_inds2_turned_and_fliped = self.__class__.transpose_and_flip(
                border_inds2, self.transpose_2_to_1[i], self.flip_2_to_1[i], field_dim=0
            )
            nodes_couples.append(
                np.hstack(
                    (
                        border_inds1.reshape((-1, 1)),
                        border_inds2_turned_and_fliped.reshape((-1, 1)),
                    )
                )
            )
        if len(nodes_couples) > 0:
            nodes_couples = np.vstack(nodes_couples)
        return MultiPatchBSplineConnectivity.from_nodes_couples(
            nodes_couples, shape_by_patch
        )

    def get_borders_couples(self, separated_field, offset=0):
        field_dim = separated_field[0].ndim - self.NPa
        borders1 = []
        borders2_turned_and_fliped = []
        for i in range(self.nb_couples):
            border1 = self.__class__.extract_border_pts(
                separated_field[self.spline1_inds[i]],
                self.axes1[i],
                self.front_sides1[i],
                offset=offset,
                field_dim=field_dim,
            )
            borders1.append(border1)
            border2 = self.__class__.extract_border_pts(
                separated_field[self.spline2_inds[i]],
                self.axes2[i],
                self.front_sides2[i],
                offset=offset,
                field_dim=field_dim,
            )
            border2_turned_and_fliped = self.__class__.transpose_and_flip(
                border2,
                self.transpose_2_to_1[i],
                self.flip_2_to_1[i],
                field_dim=field_dim,
            )
            borders2_turned_and_fliped.append(border2_turned_and_fliped)
        return borders1, borders2_turned_and_fliped

    def get_borders_couples_splines(self, splines):
        borders1 = []
        borders2_turned_and_fliped = []
        for i in range(self.nb_couples):
            border1 = self.__class__.extract_border_spline(
                splines[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i]
            )
            borders1.append(border1)
            border2 = self.__class__.extract_border_spline(
                splines[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i]
            )
            border2_turned_and_fliped = self.__class__.transpose_and_flip_spline(
                border2, self.transpose_2_to_1[i], self.flip_2_to_1[i]
            )
            borders2_turned_and_fliped.append(border2_turned_and_fliped)
        return borders1, borders2_turned_and_fliped

    def compute_border_couple_DN(
        self,
        couple_ind: int,
        splines: list[BSpline],
        XI1_border: list[np.ndarray],
        k1: list[int],
    ):
        spline1 = splines[self.spline1_inds[couple_ind]]
        ax1 = self.axes1[couple_ind]
        front1 = self.front_sides1[couple_ind]
        spline2 = splines[self.spline2_inds[couple_ind]]
        ax2 = self.axes2[couple_ind]
        front2 = self.front_sides2[couple_ind]
        XI1 = (
            XI1_border[(self.NPa - 1 - ax1) :]
            + [np.array([spline1.bases[ax1].span[int(front1)]])]
            + XI1_border[: (self.NPa - 1 - ax1)]
        )
        transpose_back = np.argsort(self.transpose_2_to_1[couple_ind])
        flip_back = self.flip_2_to_1[couple_ind][transpose_back]
        spans = spline2.getSpans()[(ax2 + 1) :] + spline2.getSpans()[:ax2]
        XI2_border = [
            (
                (sum(spans[i]) - XI1_border[transpose_back[i]])
                if flip_back[i]
                else XI1_border[transpose_back[i]]
            )
            for i in range(self.NPa - 1)
        ]
        XI2 = (
            XI2_border[(self.NPa - 1 - ax2) :]
            + [np.array([spline2.bases[ax2].span[int(front2)]])]
            + XI2_border[: (self.NPa - 1 - ax2)]
        )
        k2 = k1[: self.axes1[couple_ind]] + k1[(self.axes1[couple_ind] + 1) :]
        k2 = [k2[i] for i in transpose_back]
        k2 = (
            k2[: self.axes2[couple_ind]]
            + [k1[self.axes1[couple_ind]]]
            + k2[self.axes2[couple_ind] :]
        )
        DN1 = spline1.DN(XI1, k=k1)
        DN2 = spline2.DN(XI2, k=k2)
        return DN1, DN2

    def compute_border_couple_DN(
        self,
        couple_ind: int,
        splines: list[BSpline],
        XI1_border: list[np.ndarray],
        k1: list[int],
    ):
        spline1 = splines[self.spline1_inds[couple_ind]]
        spans1 = spline1.getSpans()
        ax1 = self.axes1[couple_ind]
        front1 = self.front_sides1[couple_ind]
        XI1 = (
            XI1_border[(self.NPa - 1 - ax1) :]
            + [np.array([spline1.bases[ax1].span[int(front1)]])]
            + XI1_border[: (self.NPa - 1 - ax1)]
        )
        DN1 = spline1.DN(XI1, k=k1)

        spline2 = splines[self.spline2_inds[couple_ind]]
        spans2 = spline2.getSpans()
        ax2 = self.axes2[couple_ind]
        front2 = self.front_sides2[couple_ind]
        transpose = self.transpose_2_to_1[couple_ind]
        A, b = self.get_operator_allxi1_to_allxi2(spans1, spans2, couple_ind)
        XI2 = []
        for i in range(self.NPa):
            j = np.argmax(np.abs(A[i]))
            XI2.append(A[i, j] * XI1[j] + b[i])

        k = int(sum(k1))
        DN2 = spline2.DN(XI2, k=k)
        if k != 0:
            AT = 1
            for i in range(k):
                AT = np.tensordot(AT, A, 0)
            AT = AT.transpose(*2 * np.arange(k), *(2 * np.arange(k) + 1))
            DN2 = np.tensordot(DN2, AT, k)
            i1 = np.repeat(np.arange(self.NPa), k1)
            DN2 = DN2[tuple(i1.tolist())]
        return DN1, DN2
