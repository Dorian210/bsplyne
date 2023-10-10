# %%
from itertools import permutations
from functools import lru_cache

import numpy as np
import scipy.sparse as sps
from tqdm import trange

from .b_spline import BSpline

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
    unique_nodes_inds : numpy.ndarray of int
        The indices of the unique representation needed to create the unpacked one.
    shape_by_patch : numpy.ndarray of int
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
        unique_nodes_inds : numpy.ndarray of int
            The indices of the unique representation needed to create the unpacked one.
        shape_by_patch : numpy.ndarray of int
            The shape of the separated nodes by patch.
        nb_unique_nodes : int
            The total number of unique nodes.
        """
        self.unique_nodes_inds = unique_nodes_inds
        self.shape_by_patch = shape_by_patch
        self.nb_nodes = np.sum(np.prod(self.shape_by_patch, axis=1))
        self.nb_unique_nodes = nb_unique_nodes #np.unique(self.unique_nodes_inds).size
        self.nb_patchs, self.npa = self.shape_by_patch.shape
    
    @classmethod
    def from_nodes_couples(cls, nodes_couples, shape_by_patch):
        """
        Create the connectivity from a list of couples of unpacked nodes.

        Parameters
        ----------
        nodes_couples : numpy.ndarray of int
            Couples of indices of unpacked nodes that are considered the same.
            Its shape should be (# of couples, 2)
        shape_by_patch : numpy.ndarray of int
            The shape of the separated nodes by patch.

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        dic = {}
        for a, b in nodes_couples:
            if a in dic:
                dic[a].append(b)
            else:
                dic[a] = [b]
            if b in dic:
                dic[b].append(a)
            else:
                dic[b] = [a]
        nb_nodes = np.sum(np.prod(shape_by_patch, axis=1))
        unique_nodes_inds = np.arange(nb_nodes)
        for key, values in dic.items():
            for v in values:
                unique_nodes_inds[v] = unique_nodes_inds[key]
        different_unique_nodes_inds, inverse = np.unique(unique_nodes_inds, return_inverse=True)
        unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], different_unique_nodes_inds))) - 1)[inverse]
        nb_unique_nodes = np.unique(unique_nodes_inds).size
        return cls(unique_nodes_inds, shape_by_patch, nb_unique_nodes)
    
    @classmethod
    def from_separated_ctrlPts(cls, separated_ctrlPts, eps=1e-10):
        """
        Create the connectivity from a list of control points given as 
        a separated field by comparing every couple of points.

        Parameters
        ----------
        separated_ctrlPts : list of numpy.ndarray of float
            Control points of every patch to be compared in the separated 
            representation. Every array is of shape : 
            (``NPh``, nb elem for dim 1, ..., nb elem for dim ``npa``)
        eps : float, optional
            Maximum distance between two points to be considered the same, by default 1e-10

        Returns
        -------
        MultiPatchBSplineConnectivity
            Instance of `MultiPatchBSplineConnectivity` created.
        """
        NPh = separated_ctrlPts[0].shape[0]
        assert np.all([ctrlPts.shape[0]==NPh for ctrlPts in separated_ctrlPts[1:]]), "Physical spaces must contain the same number of dimensions !"
        shape_by_patch = np.array([ctrlPts.shape[1:] for ctrlPts in separated_ctrlPts], dtype='int')
        nodes_couples = []
        previous_pts = separated_ctrlPts[0].reshape((NPh, -1))
        previous_inds_counter = previous_pts.shape[1]
        previous_inds = np.arange(previous_inds_counter)
        for ctrlPts in separated_ctrlPts[1:]:
            # create current pts and inds
            current_pts = ctrlPts.reshape((NPh, -1))
            current_inds_counter = previous_inds_counter + current_pts.shape[1]
            current_inds = np.arange(previous_inds_counter, current_inds_counter)
            # get couples
            dist = np.linalg.norm(previous_pts[:, :, None] - current_pts[:, None, :], axis=0)
            previous_inds_inds, current_inds_inds = (dist<eps).nonzero()
            nodes_couples.append(np.hstack((previous_inds[previous_inds_inds, None], current_inds[current_inds_inds, None])))
            # add current to previous for next iteration
            previous_pts = np.hstack((previous_pts, current_pts))
            previous_inds_counter = current_inds_counter
            previous_inds = np.hstack((previous_inds, current_inds))
        if len(nodes_couples)>0:
            nodes_couples = np.vstack(nodes_couples)
        else:
            nodes_couples = np.empty((0, 2), dtype='int')
        return cls.from_nodes_couples(nodes_couples, shape_by_patch)
    
    def unpack(self, unique_field):
        """
        Extract the unpacked representation from a unique representation.

        Parameters
        ----------
        unique_field : numpy.ndarray
            The unique representation. Its shape should be :
            (field, shape, ..., `self`.`nb_unique_nodes`)

        Returns
        -------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        unpacked_field = unique_field[..., self.unique_nodes_inds]
        return unpacked_field
    
#     def unpack_patch_jacobians(self, field_size):
#         patch_jacobians = []
#         ind = 0
#         for patch_shape in self.shape_by_patch:
#             nb_row = np.prod(patch_shape)
#             next_ind = ind + nb_row
#             row = np.arange(nb_row)
#             col = self.unique_nodes_inds[ind:next_ind]
#             data = np.ones(nb_row)
#             mat = sps.coo_matrix((data, (row, col)), shape=(nb_row, self.nb_unique_nodes))
#             patch_jacobians.append(sps.block_diag([mat]*field_size))
#             ind = next_ind
#         return patch_jacobians
    
    def pack(self, unpacked_field):
        """
        Extract the unique representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape should be :
            (field, shape, ..., `self`.`nb_nodes`)

        Returns
        -------
        unique_nodes : numpy.ndarray
            The unique representation. Its shape is :
            (field, shape, ..., `self`.`nb_unique_nodes`)
        """
        field_shape = unpacked_field.shape[:-1]
        unique_field = np.zeros((*field_shape, self.nb_unique_nodes), dtype=unpacked_field.dtype)
        unique_field[..., self.unique_nodes_inds[::-1]] = unpacked_field[..., ::-1]
        return unique_field
    
    def separate(self, unpacked_field):
        """
        Extract the separated representation from an unpacked representation.

        Parameters
        ----------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)

        Returns
        -------
        separated_field : list of numpy.ndarray
            The separated representation. Every array is of shape : 
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)
        """
        field_shape = unpacked_field.shape[:-1]
        separated_field = []
        ind = 0
        for patch_shape in self.shape_by_patch:
            next_ind = ind + np.prod(patch_shape)
            separated_field.append(unpacked_field[..., ind:next_ind].reshape((*field_shape, *patch_shape)))
            ind = next_ind
        return separated_field
    
    def agglomerate(self, separated_field):
        """
        Extract the unpacked representation from a separated representation.

        Parameters
        ----------
        separated_field : list of numpy.ndarray
            The separated representation. Every array is of shape : 
            (field, shape, ..., nb elem for dim 1, ..., nb elem for dim `npa`)

        Returns
        -------
        unpacked_field : numpy.ndarray
            The unpacked representation. Its shape is :
            (field, shape, ..., `self`.`nb_nodes`)
        """
        field_shape = separated_field[0].shape[:-self.npa]
        assert np.all([f.shape[:-self.npa]==field_shape for f in separated_field]), "Every patch must have the same field shape !"
        unpacked_field = np.concatenate([f.reshape((*field_shape, -1)) for f in separated_field], axis=-1)
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
        unique_field_indices : numpy.ndarray of int or list of numpy.ndarray of int
            The unique, unpacked or separated representation of a field's unique indices.
            If unique, its shape is (*`field_shape`, `self`.`nb_unique_nodes`).
            If unpacked, its shape is : (*`field_shape`, `self`.`nb_nodes`).
            If separated, every array is of shape : (*`field_shape`, nb elem for dim 1, ..., nb elem for dim `npa`).
        """
        nb_indices = np.prod(field_shape)*self.nb_unique_nodes
        unique_field_indices_as_unique_field = np.arange(nb_indices, dtype='int').reshape((*field_shape, self.nb_unique_nodes))
        if representation=="unique":
            unique_field_indices = unique_field_indices_as_unique_field
            return unique_field_indices
        elif representation=="unpacked":
            unique_field_indices = self.unpack(unique_field_indices_as_unique_field)
            return unique_field_indices
        elif representation=="separated":
            unique_field_indices = self.separate(self.unpack(unique_field_indices_as_unique_field))
            return unique_field_indices
        else:
            raise ValueError(f'Representation "{representation}" not recognised. Representation must either be "unique", "unpacked", or "separated" !')
    
    def get_duplicate_unpacked_nodes_mask(self):
        unique, inverse, counts = np.unique(self.unique_nodes_inds, return_inverse=True, return_counts=True)
        duplicate_nodes_mask = np.zeros(self.nb_nodes, dtype='bool')
        duplicate_nodes_mask[counts[inverse]>1] = True
        return duplicate_nodes_mask
    
    def extract_exterior_borders(self, splines):
        if self.npa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
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
                bases = np.hstack((spline.bases[(axis + 1):], spline.bases[:axis]))
                axes = arr[axis:-1] + arr[:axis]
                border_shape_by_patch_spline = np.hstack((shape_by_patch_spline[(axis + 1):], shape_by_patch_spline[:axis]))
                if not np.take(duplicate_nodes_mask_spline, 0, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases[::-1])
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, 0, axis=axis).transpose(axes[::-1]).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[::-1][None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {0} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
                if not np.take(duplicate_nodes_mask_spline, -1, axis=axis).all():
                    bspline_border = BSpline.from_bases(bases)
                    border_splines.append(bspline_border)
                    unique_nodes_inds_spline_border = np.take(unique_nodes_inds_spline, -1, axis=axis).transpose(axes).ravel()
                    border_unique_nodes_inds.append(unique_nodes_inds_spline_border)
                    border_shape_by_patch_spline_border = border_shape_by_patch_spline[None]
                    border_shape_by_patch.append(border_shape_by_patch_spline_border)
                    # print(f"side {-1} of axis {axis} of patch {i} uses nodes {unique_nodes_inds_spline_border}")
        border_splines = np.array(border_splines, dtype='object')
        border_unique_nodes_inds = np.concatenate(border_unique_nodes_inds)
        border_shape_by_patch = np.concatenate(border_shape_by_patch)
        border_unique_to_self_unique_connectivity, inverse = np.unique(border_unique_nodes_inds, return_inverse=True)
        border_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], border_unique_to_self_unique_connectivity))) - 1)[inverse]
        border_nb_unique_nodes = np.unique(border_unique_nodes_inds).size
        border_connectivity = self.__class__(border_unique_nodes_inds, border_shape_by_patch, border_nb_unique_nodes)
        return border_connectivity, border_splines, border_unique_to_self_unique_connectivity
    
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
        new_splines = splines[patches_to_keep]
        separated_unique_nodes_inds = self.unique_field_indices(())
        new_unique_nodes_inds = np.concatenate([separated_unique_nodes_inds[patch].flat for patch in patches_to_keep])
        new_shape_by_patch = self.shape_by_patch[patches_to_keep]
        new_unique_to_self_unique_connectivity, inverse = np.unique(new_unique_nodes_inds, return_inverse=True)
        new_unique_nodes_inds -= np.cumsum(np.diff(np.concatenate(([-1], new_unique_to_self_unique_connectivity))) - 1)[inverse]
        new_nb_unique_nodes = np.unique(new_unique_nodes_inds).size
        new_connectivity = self.__class__(new_unique_nodes_inds, new_shape_by_patch, new_nb_unique_nodes)
        return new_connectivity, new_splines, new_unique_to_self_unique_connectivity
    
    def save_paraview(self, splines, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.npa
        
        if verbose:
            iterator = trange(self.nb_patchs, desc=("Saving " + name))
        else:
            iterator = range(self.nb_patchs)
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.nb_patchs)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.separate(self.unpack(value))
                for patch in range(self.nb_patchs):
                    separated_fields[patch][key] = separated_value[patch]
        
        groups = {}
        for patch in iterator:
            groups = splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                                      path, 
                                                      name, 
                                                      n_step=n_step, 
                                                      n_eval_per_elem=n_eval_per_elem, 
                                                      fields=separated_fields[patch], 
                                                      groups=groups, 
                                                      make_pvd=((patch + 1)==self.nb_patchs), 
                                                      verbose=False)
        

if __name__=='__main__':
    from bsplyne_lib import new_cube
    
    cube1, cube1_ctrlPts = new_cube([0.5, 0.5, 0.5], [0, 0, 1], 1)
    cube2, cube2_ctrlPts = new_cube([1.5, 0.5, 0.5], [1, 0, 0], 1)
    splines = [cube1, cube2]
    separated_ctrlPts = [cube1_ctrlPts, cube2_ctrlPts]
    
    connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrlPts)
    
    border_connectivity, border_splines, border_unique_to_self_unique_connectivity = connectivity.extract_exterior_borders(splines)
    
    print(connectivity.unique_field_indices((1,)))
    print(border_connectivity.unique_field_indices((1,)))
    
    
# %%


class CouplesBSplineBorder:
    
    def __init__(self, spline1_inds, spline2_inds, axes1, axes2, front_sides1, front_sides2, transpose_2_to_1, flip_2_to_1, NPa):
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
        border_field = field.transpose(axis + field_dim, *np.arange(field_dim), *(base_face + field_dim))[(-(1 + offset) if front_side else offset)]
        return border_field
    
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
    def from_splines(cls, separated_ctrl_pts, splines):
        NPa = splines[0].NPa
        assert np.all([sp.NPa==NPa for sp in splines]), "Every patch should have the same parametric space dimension !"
        NPh = separated_ctrl_pts[0].shape[0]
        assert np.all([ctrl_pts.shape[0]==NPh for ctrl_pts in separated_ctrl_pts]), "Every patch should have the same physical space dimension !"
        npatch = len(splines)
        all_flip = np.unpackbits(np.arange(2**(NPa - 1), dtype='uint8')[:, None], axis=1, count=(NPa - 1 - 8), bitorder='little')[:, ::-1].astype('bool')
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
                    degrees1 = np.hstack((spline1.getDegrees()[(axis1 + 1):], spline1.getDegrees()[:axis1]))
                    knots1 = spline1.getKnots()[(axis1 + 1):] + spline1.getKnots()[:axis1]
                    # print(f"||ax1 {axis1}")
                    for axis2 in range(spline2.NPa):
                        degrees2 = np.hstack((spline2.getDegrees()[(axis2 + 1):], spline2.getDegrees()[:axis2]))
                        knots2 = spline2.getKnots()[(axis2 + 1):] + spline2.getKnots()[:axis2]
                        spans2 = spline2.getSpans()[(axis2 + 1):] + spline2.getSpans()[:axis2]
                        # print(f"|||ax2 {axis2}")
                        for front_side1 in [False, True]:
                            pts1 = cls.extract_border_pts(ctrl_pts1, axis1, front_side1)
                            # print(f"||||{'front' if front_side1 else 'back '} side1")
                            for front_side2 in [False, True]:
                                pts2 = cls.extract_border_pts(ctrl_pts2, axis2, front_side2)
                                # print(f"|||||{'front' if front_side2 else 'back '} side2")
                                for transpose in all_transpose:
                                    # print(f"||||||transpose {transpose}")
                                    if (degrees1==[degrees2[i] for i in transpose]).all():
                                        # print(f"||||||same degrees {degrees1}")
                                        if list(pts1.shape[1:])==[pts2.shape[1:][i] for i in transpose]:
                                            # print(f"||||||same shapes {pts1.shape[1:]}")
                                            if np.all([knots1[i].size==knots2[transpose[i]].size for i in range(NPa - 1)]):
                                                # print(f"||||||same knots sizes {[knots1[i].size for i in range(NPa - 1)]}")
                                                for flip in all_flip:
                                                    # print(f"|||||||flip {flip}")
                                                    if np.all([(k1==k2).all() for k1, k2 in zip(knots1, cls.transpose_and_flip_knots(knots2, spans2, transpose, flip))]):
                                                        # print(f"|||||||same knots {knots1}")
                                                        pts2_turned = cls.transpose_and_flip(pts2, transpose, flip)
                                                        if np.allclose(pts1, pts2_turned):
                                                            # print("_________________GOGOGO_________________")
                                                            spline1_inds.append(spline1_ind)
                                                            spline2_inds.append(spline2_ind)
                                                            axes1.append(axis1)
                                                            axes2.append(axis2)
                                                            front_sides1.append(front_side1)
                                                            front_sides2.append(front_side2)
                                                            transpose_2_to_1.append(transpose)
                                                            flip_2_to_1.append(flip)
        spline1_inds = np.array(spline1_inds, dtype='int')
        spline2_inds = np.array(spline2_inds, dtype='int')
        axes1 = np.array(axes1, dtype='int')
        axes2 = np.array(axes2, dtype='int')
        front_sides1 = np.array(front_sides1, dtype='bool')
        front_sides2 = np.array(front_sides2, dtype='bool')
        transpose_2_to_1 = np.array(transpose_2_to_1, dtype='int')
        flip_2_to_1 = np.array(flip_2_to_1, dtype='bool')
        return cls(spline1_inds, spline2_inds, axes1, axes2, front_sides1, front_sides2, transpose_2_to_1, flip_2_to_1, NPa)
    
    def get_connectivity(self, shape_by_patch):
        indices = []
        start = 0
        for shape in shape_by_patch:
            end = start + np.prod(shape)
            indices.append(np.arange(start, end).reshape(shape))
            start = end
        nodes_couples = []
        for i in range(self.nb_couples):
            border_inds1 = self.__class__.extract_border_pts(indices[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i], field_dim=0)
            border_inds2 = self.__class__.extract_border_pts(indices[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i], field_dim=0)
            border_inds2_turned_and_fliped = self.__class__.transpose_and_flip(border_inds2, self.transpose_2_to_1[i], self.flip_2_to_1[i], field_dim=0)
            nodes_couples.append(np.hstack((border_inds1.reshape((-1, 1)), border_inds2_turned_and_fliped.reshape((-1, 1)))))
        if len(nodes_couples)>0:
            nodes_couples = np.vstack(nodes_couples)
        return MultiPatchBSplineConnectivity.from_nodes_couples(nodes_couples, shape_by_patch)
    
    def get_borders_couples(self, separated_field, offset=0):
        field_dim = separated_field[0].ndim - self.NPa
        borders1 = []
        borders2_turned_and_fliped = []
        for i in range(self.nb_couples):
            border1 = self.__class__.extract_border_pts(separated_field[self.spline1_inds[i]], self.axes1[i], self.front_sides1[i], offset=offset, field_dim=field_dim)
            borders1.append(border1)
            border2 = self.__class__.extract_border_pts(separated_field[self.spline2_inds[i]], self.axes2[i], self.front_sides2[i], offset=offset, field_dim=field_dim)
            border2_turned_and_fliped = self.__class__.transpose_and_flip(border2, self.transpose_2_to_1[i], self.flip_2_to_1[i], field_dim=field_dim)
            borders2_turned_and_fliped.append(border2_turned_and_fliped)
        return borders1, borders2_turned_and_fliped

if __name__=="__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from bsplyne import BSpline, MultiPatchBSplineConnectivity
    
    def plot_multipatch(ddl, connectivity):
        if ddl.shape[0]==2:
            fig, ax = plt.subplots()
            line_col = LineCollection
        elif ddl.shape[0]==3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            line_col = Line3DCollection
        else:
            raise NotImplementedError(f"Physical space must be 2D or 3D, not {ddl.shape[0]}D !")
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        pts = connectivity.separate(connectivity.unpack(ddl))
        indices = connectivity.unique_field_indices(())
        for i, (p, txt) in enumerate(zip(pts[::-1], indices[::-1])):
            if 1==1:
                ax.scatter(*p)
                for j, t in enumerate(txt.flat):
                    ax.text(*[x.flat[j] for x in p], t, size=20, zorder=10, color='k')
                p = np.moveaxis(p, 0, -1)
                for _ in range(connectivity.npa):
                    ax.add_collection(line_col(p.reshape((-1, p.shape[-2], p.shape[-1])), colors=colors[i]))
                    p = np.moveaxis(p, 0, -2)
        plt.show()
    
    pts1 = np.array(np.meshgrid([0, 1], [0, 1, 2], [0, 1], indexing='ij')).transpose(0, 3, 2, 1)
    # pts1 = np.concatenate((pts1, np.zeros_like(pts1[0])[None]))
    sp1 = BSpline([1, 2, 1], [np.array([0, 0, 1, 1], dtype='float'), np.array([0, 0, 0, 1, 1, 1], dtype='float'), np.array([0, 0, 1, 1], dtype='float')])
    pts2 = np.array(np.meshgrid([0, 2], [0, 1, 2], [0, 1], indexing='ij'))
    # pts2 = np.concatenate((pts2, np.zeros_like(pts2[0])[None]))
    sp2 = BSpline([1, 2, 1], [np.array([0, 0, 1, 1], dtype='float'), np.array([0, 0, 0, 1, 1, 1], dtype='float'), np.array([0, 0, 1, 1], dtype='float')])
    pts = [pts1, pts2]
    splines = [sp1, sp2]
    
    couples = CouplesBSplineBorder.from_splines(pts, splines)
    shape_by_patch = np.array([p.shape[1:] for p in pts], dtype='int')
    connectivity = couples.get_connectivity(shape_by_patch)
    
    # connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(pts)
    
    ddl = connectivity.pack(connectivity.agglomerate(pts))
    
    plot_multipatch(ddl, connectivity)

# %%


# TODO : extract border, displace border, DN multipatch, knot insertion, order elevation
class MultiPatchBSpline:
    
    def __init__(self, splines, couples=None, connectivity=None):
        self.splines = splines
        self.npatch = len(self.splines)
        assert np.all([s.NPa==self.splines[0].NPa for s in self.splines[1:]]), "The parametric space should be of the same dimension on every patch"
        self.NPa = self.splines[0].NPa
        # assert np.all([s.NPh==self.splines[0].NPh for s in self.splines[1:]]), "The physical space should be of the same dimension on every patch"
        # self.NPh = self.splines[0].NPh
        # if couples is None:
        #     self.couples = CouplesBSplineBorder.from_splines(splines)
        # else:
        #     self.couples = couples
        if connectivity is None:
            self.connectivity = self.couples.get_connectivity(self.splines)
        else:
            self.connectivity = connectivity
    
    def get_border(self):
        if self.NPa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_coo_mask = self.get_duplicate_coo_mask()
        border_splines = []
        border_coo_inds = []
        ind = 0
        for i in range(self.npatch):
            s = self.splines[i]
            degrees = s.getDegrees()
            knots = np.array(s.getKnots(), dtype='object')
            coo = s.ctrlPts
            next_ind = ind + self.coo_sizes[i]
            coo_inds_s = self.coo_inds[ind:next_ind].reshape(coo.shape)
            duplicate_coo_mask_s = duplicate_coo_mask[ind:next_ind].reshape(coo.shape)
            ind = next_ind
            for axis in range(self.NPa):
                d = np.delete(degrees, axis)
                k = np.delete(knots, axis, axis=0)
                for side in range(2):
                    if not np.take(duplicate_coo_mask_s, -side, axis=(axis+1)).all():
                        border_splines.append(BSpline(np.take(coo, -side, axis=(axis+1)), d, k))
                        border_coo_inds.append(np.take(coo_inds_s, -side, axis=(axis+1)).flat)
        border_splines = np.array(border_splines, dtype='object')
        border_coo_inds = np.concatenate(border_coo_inds)
        border = self.__class__(border_splines, border_coo_inds, self.ndof)
        return border
    
    def move_border(self):
        # TODO with IDW
        raise NotImplementedError()
    
    def save_paraview(self, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if verbose:
            iterator = trange(self.npatch, desc=("Saving " + name))
        else:
            iterator = range(self.npatch)
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.npatch)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.connectivity.separate(self.connectivity.unpack(value))
                for patch in range(self.npatch):
                    separated_fields[patch][key] = separated_value[patch]
        
        groups = {}
        for patch in iterator:
            groups = self.splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                                      path, 
                                                      name, 
                                                      n_step=n_step, 
                                                      n_eval_per_elem=n_eval_per_elem, 
                                                      fields=separated_fields[patch], 
                                                      groups=groups, 
                                                      make_pvd=((patch + 1)==self.npatch), 
                                                      verbose=False)
    
    def save_stl(self):
        raise NotImplementedError()

if __name__=="__main__":
    from bsplyne_lib import BSpline, new_cube

    cube = new_cube([0.5, 0.5, 0.5], [0, 0, 1], 1)
    cube_degrees = cube.getDegrees()
    cube_knots = cube.getKnots()
    orientation = np.ones(3, dtype='float')/np.sqrt(3)
    length = 10
    splines = [cube]
    for axis in range(3):
        to_extrude = np.take(cube.ctrlPts, -1, axis=(axis + 1))
        ctrlPts = np.concatenate((to_extrude[:, None], 
                                (to_extrude + length*orientation[:, None, None])[:, None]), axis=1)
        degrees = np.array([1] + [cube_degrees[i] for i in range(3) if i!=axis], dtype='int')
        knots = [np.array([0, 0, 1, 1], dtype='float')] + [cube_knots[i] for i in range(3) if i!=axis]
        splines.append(BSpline(ctrlPts, degrees, knots))
    splines = np.array(splines, dtype='object')
    connectivity = MultiPatchBSplineConnectivity.from_splines(splines)
    couples = None
    volume = MultiPatchBSpline(splines, couples, connectivity)
    dof = volume.connectivity.get_dof(volume.splines)
    U_pts = volume.connectivity.dof_to_pts(dof + 1*np.random.rand(dof.size))
    fields = {"U": U_pts[None]}
    volume.save_paraview("./out_tests", "MultiPatch", fields=fields)


# %%

if __name__=="__main__":
    from bsplyne_lib import new_cube
    from bsplyne_lib.geometries_in_3D import _rotation_matrix
    
    length = 2.
    C1 = new_cube([length/4, length/4, length/4], [0, 0, 1], length/2)
    C2 = new_cube([length/4, length/4, 3*length/4], [0, 0, 1], length/2)
    center2 = np.array([length/4, length/4, 3*length/4])[:, None, None, None]
    C2.ctrlPts = np.tensordot(_rotation_matrix([0, 0, 1], np.pi*0/2), C2.ctrlPts - center2, 1) + center2
    splines = np.array([C1, C2], dtype='object')
    to_insert = [np.array([0.25, 0.5, 0.75], dtype='float'), 
                 np.array([0.25, 0.5, 0.75], dtype='float'), 
                 np.array([0.25, 0.5, 0.75], dtype='float')]
    to_elevate = [1, 1, 1]
    for sp in splines:
        sp.orderElevation(to_elevate)
        sp.knotInsertion(to_insert)
    volume = MultiPatchBSpline(splines)
    print(volume.splines[0].ctrlPts.shape)
    print(volume.couples.spline1_inds, 
          volume.couples.spline2_inds, 
          volume.couples.axes1, 
          volume.couples.axes2, 
          volume.couples.front_sides1, 
          volume.couples.front_sides2, 
          volume.couples.transpose_2_to_1, 
          volume.couples.flip_2_to_1)
    print(volume.connectivity.coo_inds)
    dof = volume.connectivity.get_dof(volume.splines)
    U_pts = volume.connectivity.dof_to_pts(dof + 1 + 0.1*np.random.rand(dof.size))
    fields = {"U": U_pts[None]}
    volume.save_paraview("./out_tests", "MultiPatch", fields=fields)
    

# %%

if __name__=="__main__":
    import numpy as np
    from stl import mesh
    from bsplyne_lib import new_quarter_strut

    spline = new_quarter_strut([0, 0, 0], [0, 0, 1], 1, 10)

    tri = []
    XI = spline.linspace(n_eval_per_elem=[10, 1, 100])
    shape = [xi.size for xi in XI]
    for axis in range(3):
        XI_axis = [xi for xi in XI]
        shape_axis = [shape[i] for i in range(len(shape)) if i!=axis]
        XI_axis[axis] = np.zeros(1)
        pts_l = spline(tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        XI_axis[axis] = np.ones(1)
        pts_r = spline(tuple(XI_axis), [0, 0, 0]).reshape([3] + shape_axis)
        for pts in [pts_l, pts_r]:
            A = pts[:,  :-1,  :-1].reshape((3, -1)).T[:, None, :]
            B = pts[:,  :-1, 1:  ].reshape((3, -1)).T[:, None, :]
            C = pts[:, 1:  ,  :-1].reshape((3, -1)).T[:, None, :]
            D = pts[:, 1:  , 1:  ].reshape((3, -1)).T[:, None, :]
            tri1 = np.concatenate((A, B, C), axis=1)
            tri2 = np.concatenate((D, C, B), axis=1)
            tri.append(np.concatenate((tri1, tri2), axis=0))
    tri = np.concatenate(tri, axis=0)
    data = np.empty(tri.shape[0], dtype=mesh.Mesh.dtype)
    data['vectors'] = tri
    m = mesh.Mesh(data, remove_empty_areas=True)

    m.save('new_stl_file.stl')

# %%

class MultiPatchBSpline:
    
    def __init__(self, splines, connectivity):
        self.splines = splines
        self.npatch = len(self.splines)
        assert np.all([s.NPa==self.splines[0].NPa for s in self.splines[1:]]), "The parametric space should be of the same dimension on every patch"
        self.NPa = self.splines[0].NPa
        assert np.all([s.NPh==self.splines[0].NPh for s in self.splines[1:]]), "The physical space should be of the same dimension on every patch"
        self.NPh = self.splines[0].NPh
        self.connectivity = connectivity
    
    def get_border(self):
        if self.NPa<=1:
            raise AssertionError("The parametric space must be at least 2D to extract borders !")
        duplicate_coo_mask = self.get_duplicate_coo_mask()
        border_splines = []
        border_coo_inds = []
        ind = 0
        for i in range(self.npatch):
            s = self.splines[i]
            degrees = s.getDegrees()
            knots = np.array(s.getKnots(), dtype='object')
            coo = s.ctrlPts
            next_ind = ind + self.coo_sizes[i]
            coo_inds_s = self.coo_inds[ind:next_ind].reshape(coo.shape)
            duplicate_coo_mask_s = duplicate_coo_mask[ind:next_ind].reshape(coo.shape)
            ind = next_ind
            for axis in range(self.NPa):
                d = np.delete(degrees, axis)
                k = np.delete(knots, axis, axis=0)
                for side in range(2):
                    if not np.take(duplicate_coo_mask_s, -side, axis=(axis+1)).all():
                        border_splines.append(BSpline(np.take(coo, -side, axis=(axis+1)), d, k))
                        border_coo_inds.append(np.take(coo_inds_s, -side, axis=(axis+1)).flat)
        border_splines = np.array(border_splines, dtype='object')
        border_coo_inds = np.concatenate(border_coo_inds)
        border = self.__class__(border_splines, border_coo_inds, self.ndof)
        return border
    
    def move_border(self):
        # TODO with IDW
        raise NotImplementedError()
    
    def save_paraview(self, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=10, unique_fields={}, separated_fields=None, verbose=True):
        if type(n_eval_per_elem) is int:
            n_eval_per_elem = [n_eval_per_elem]*self.NPa
        
        if verbose:
            iterator = trange(self.npatch, desc=("Saving " + name))
        else:
            iterator = range(self.npatch)
        
        if separated_fields is None:
            separated_fields = [{} for _ in range(self.npatch)]
        
        for key, value in unique_fields.items():
            if callable(value):
                raise NotImplementedError("To handle functions as fields, use separated_fields !")
            else:
                separated_value = self.connectivity.separate(self.connectivity.unpack(value))
                for patch in range(self.npatch):
                    separated_fields[patch][key] = separated_value[patch]
        
        groups = {}
        for patch in iterator:
            groups = self.splines[patch].saveParaview(separated_ctrl_pts[patch], 
                                                      path, 
                                                      name, 
                                                      n_step=n_step, 
                                                      n_eval_per_elem=n_eval_per_elem, 
                                                      fields=separated_fields[patch], 
                                                      groups=groups, 
                                                      make_pvd=((patch + 1)==self.npatch), 
                                                      verbose=False)
    
    def save_stl(self):
        raise NotImplementedError()