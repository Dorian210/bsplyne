"""
.. include:: ../README.md
"""
from .b_spline_basis import BSplineBasis
BSplineBasis.__module__ = "bsplyne.b_spline_basis"
from .b_spline import BSpline
BSpline.__module__ = "bsplyne.b_spline"
from .multi_patch_b_spline import MultiPatchBSplineConnectivity, CouplesBSplineBorder
MultiPatchBSplineConnectivity.__module__ = "bsplyne.multi_patch_b_spline"
CouplesBSplineBorder.__module__ = "bsplyne.multi_patch_b_spline"
from .geometries_in_3D import (_scale_rotate_translate, 
                               new_quarter_circle, 
                               new_circle, 
                               new_disk, 
                               new_degenerated_disk, 
                               new_quarter_pipe, 
                               new_pipe, 
                               new_quarter_cylinder, 
                               new_cylinder, 
                               new_degenerated_cylinder, 
                               new_closed_circle, 
                               new_closed_disk, 
                               new_closed_pipe, 
                               new_closed_cylinder, 
                               new_quarter_strut, 
                               new_cube)
_scale_rotate_translate.__module__ = "bsplyne.geometries_in_3D"
new_quarter_circle.__module__ = "bsplyne.geometries_in_3D"
new_circle.__module__ = "bsplyne.geometries_in_3D"
new_disk.__module__ = "bsplyne.geometries_in_3D"
new_degenerated_disk.__module__ = "bsplyne.geometries_in_3D"
new_quarter_pipe.__module__ = "bsplyne.geometries_in_3D"
new_pipe.__module__ = "bsplyne.geometries_in_3D"
new_quarter_cylinder.__module__ = "bsplyne.geometries_in_3D"
new_cylinder.__module__ = "bsplyne.geometries_in_3D"
new_degenerated_cylinder.__module__ = "bsplyne.geometries_in_3D"
new_closed_circle.__module__ = "bsplyne.geometries_in_3D"
new_closed_disk.__module__ = "bsplyne.geometries_in_3D"
new_closed_pipe.__module__ = "bsplyne.geometries_in_3D"
new_closed_cylinder.__module__ = "bsplyne.geometries_in_3D"
new_quarter_strut.__module__ = "bsplyne.geometries_in_3D"
new_cube.__module__ = "bsplyne.geometries_in_3D"
