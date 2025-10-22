# %% Imports
import numpy as np

from .b_spline import BSpline

# %% 3D transformations
def _rotation_matrix(axis, angle):
    P = np.expand_dims(axis, axis=1)@np.expand_dims(axis, axis=0)
    I = np.eye(3)
    Q = np.cross(I, axis)
    R = P + np.cos(angle)*(I - P) + np.sin(angle)*Q
    return R

def _scale_matrix(scale_vector):
    S = np.diag(scale_vector)
    return S

def scale_rotate_translate(pts, scale_vector, axis, angle, translation_vector):
    """
    Applies a scale, rotation and translation to a set of points.

    Parameters
    ----------
    pts : array_like
        The points to be transformed.
    scale_vector : array_like
        The vector to scale by.
    axis : array_like
        The axis to rotate around.
    angle : float
        The angle to rotate by in radians.
    translation_vector : array_like
        The vector to translate by.

    Returns
    -------
    new_pts : array_like
        The transformed points.
    """
    S = _scale_matrix(scale_vector)
    R = _rotation_matrix(axis, angle)
    new_pts = (np.tensordot(R@S, pts, 1).T + translation_vector).T
    return new_pts

# %% inner points computation
def _IDW(known_points, unknown_points, data, exposant=2):
    dim = known_points.shape[0]
    distances = np.empty((dim, unknown_points.shape[1], known_points.shape[1]))
    for d in range(dim):
        distances[d] = np.expand_dims(unknown_points[d], axis=-1) - known_points[d]
    distances = np.linalg.norm(distances, axis=0)
    ID = distances**(-exposant)
    ID = (ID.T/ID.sum(1)).T
    data_mean = np.mean(data, axis=-1)
    return (np.tensordot(ID, data.T - data_mean, 1) + data_mean).T

def _find_inner_ctrlPts(degrees, knotVects, ctrlPts, exposant=2):
    greville = []
    for idx in range(len(degrees)):
        p = degrees[idx]
        knot = knotVects[idx]
        n = ctrlPts.shape[1+idx]
        greville.append(np.array([sum([knot[i+k] for k in range(p)])/p for i in range(p-1, n-1+p)]))
    parametricPts = np.array(np.meshgrid(*greville, indexing='ij'))
    notFound = np.zeros(parametricPts.shape[1:], dtype='bool')
    notFound[tuple([slice(1, -1) for i in range(notFound.ndim)])] = True
    found = np.logical_not(notFound)
    known_points = parametricPts[:, found]
    unknown_points = parametricPts[:, notFound]
    data = ctrlPts[:, found]
    ctrlPts[:, notFound] = _IDW(known_points, unknown_points, data, exposant=exposant)
    return ctrlPts

# %% Constants for B-Spline circles
_p = 2
_knot = np.array([0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1], dtype='float')
_C = np.array([-41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
               -41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
               -50557440/36473*np.sqrt(np.sqrt(2) + 2)/np.pi**3 - 92505600/36473*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (26527/2334272)*np.pi + (18163200/36473)/np.pi**3 + (103925760/36473)*np.sqrt(2)/np.pi**3, 
               -91238400/36473*np.sqrt(2)/np.pi**3 - (41034240/36473)/np.pi**3 - 65605/2334272*np.pi + (52646400/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (70632960/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3, 
               (1/16)*np.pi, 
               0], dtype='float')

# %% 1D parametric space
def _base_quarter_circle():
    degrees = np.array([_p], dtype='int')
    knots = [_knot]
    Z = np.zeros_like(_C)
    ctrlPts = np.array([_C, _C[::-1], Z])
    quarter_circle = BSpline(degrees, knots)
    return quarter_circle, ctrlPts

def new_quarter_circle(center, normal, radius):
    """
    Creates a B-spline quarter circle from a given center, normal and radius.

    Parameters
    ----------
    center : array_like
        The center of the quarter circle.
    normal : array_like
        The normal vector of the quarter circle.
    radius : float
        The radius of the quarter circle.

    Returns
    -------
    quarter_circle : BSpline
        The quarter circle.
    ctrlPts : array_like
        The control points of the quarter circle.
    """
    quarter_circle, ctrlPts = _base_quarter_circle()

    scale_vector = np.array([radius, radius, 1], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the quarter circle
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # The vector to translate by
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_circle, ctrlPts

def _base_circle():
    degrees = np.array([_p], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in _knot[:-_p]] 
                    + [1/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [2/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [3/4 + float(k)/4 for k in _knot[_p:]], dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.concatenate((np.array([ _C[:-1]      ,  _C[::-1][:-1], Z[:-1]]), 
                              np.array([-_C[::-1][:-1],  _C[:-1]      , Z[:-1]]), 
                              np.array([-_C[:-1]      , -_C[::-1][:-1], Z[:-1]]), 
                              np.array([ _C[::-1]     , -_C           , Z     ])), axis=1)
    circle = BSpline(degrees, knots)
    return circle, ctrlPts

def new_circle(center, normal, radius):
    """
    Create a B-spline circle in 3D space.

    Parameters
    ----------
    center : array_like
        The center of the circle.
    normal : array_like
        The normal vector of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    circle : BSpline
        The circle.
    ctrlPts : array_like
        The control points of the circle.
    """
    circle, ctrlPts = _base_circle()

    # The vector to scale by
    scale_vector = np.array([radius, radius, 1], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the circle
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # The vector to translate by
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return circle, ctrlPts

# %% 2D parametric space
def _base_disk():
    q = 1
    degrees = np.array([_p, q], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in _knot[:-_p]] 
                    + [1/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [2/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [3/4 + float(k)/4 for k in _knot[_p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.concatenate((np.array([ _C[:-1]      ,  _C[::-1][:-1], Z[:-1]]), 
                              np.array([-_C[::-1][:-1],  _C[:-1]      , Z[:-1]]), 
                              np.array([-_C[:-1]      , -_C[::-1][:-1], Z[:-1]]), 
                              np.array([ _C[::-1]     , -_C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts

def new_disk(center, normal, radius):
    """
    Creates a B-spline disk from a given center, normal and radius.

    Parameters
    ----------
    center : array_like
        The center of the disk.
    normal : array_like
        The normal vector of the disk.
    radius : float
        The radius of the disk.

    Returns
    -------
    disk : BSpline
        The disk.
    ctrlPts : array_like
        The control points of the disk.
    """
    disk, ctrlPts = _base_disk()

    # The vector to scale by
    scale_vector = np.array([radius, radius, 1], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the disk
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # The vector to translate by
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def _base_degenerated_disk():
    degrees = np.array([_p, _p], dtype='int')
    knots = [_knot, _knot]
    Z = np.zeros_like(_C)
    n = _C.size
    ctrlPts = np.empty((3, n, n), dtype='float')
    ctrlPts[:,  0,  :] = [+_C[:: 1], +_C[::-1], Z]
    ctrlPts[:, -1,  :] = [-_C[::-1], -_C[:: 1], Z]
    ctrlPts[:,  :,  0] = [+_C[:: 1], -_C[::-1], Z]
    ctrlPts[:,  :, -1] = [-_C[::-1], +_C[:: 1], Z]
    ctrlPts = _find_inner_ctrlPts(degrees, knots, ctrlPts)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts

def new_degenerated_disk(center, normal, radius):
    """
    Creates a B-spline degenerated disk from a given center, normal and radius.
    The disk is degenerated as it is created by "blowing" a square into a circle.

    Parameters
    ----------
    center : array_like
        The center of the degenerated disk.
    normal : array_like
        The normal vector of the degenerated disk.
    radius : float
        The radius of the degenerated disk.

    Returns
    -------
    disk : BSpline
        The degenerated disk.
    ctrlPts : array_like
        The control points of the degenerated disk.
    """
    disk, ctrlPts = _base_degenerated_disk()

    # The vector to scale by
    scale_vector = np.array([radius, radius, 1], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')
    
    # The normal vector of the degenerated disk
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # The vector to translate by
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def _base_quarter_pipe():
    q = 1
    degrees = np.array([_p, q], dtype='int')
    knots = [_knot, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.array([_C, _C[::-1], Z])
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    quarter_pipe = BSpline(degrees, knots)
    return quarter_pipe, ctrlPts

def new_quarter_pipe(center_front, orientation, radius, length):
    """
    Creates a B-spline quarter pipe from a given center, orientation, radius and length.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the quarter pipe.
    orientation : array_like
        The normal vector of the quarter pipe.
    radius : float
        The radius of the quarter pipe.
    length : float
        The length of the quarter pipe.

    Returns
    -------
    quarter_pipe : BSpline
        The quarter pipe.
    ctrlPts : array_like
        The control points of the quarter pipe.
    """
    quarter_pipe, ctrlPts = _base_quarter_pipe()

    # The vector to scale by
    scale_vector = np.array([radius, radius, length], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the quarter pipe
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # The vector to translate by
    translation_vector = np.array(center_front, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_pipe, ctrlPts

def _base_pipe():
    q = 1
    degrees = np.array([_p, q], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in _knot[:-_p]] 
                    + [1/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [2/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [3/4 + float(k)/4 for k in _knot[_p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.concatenate((np.array([ _C[:-1]      ,  _C[::-1][:-1], Z[:-1]]), 
                              np.array([-_C[::-1][:-1],  _C[:-1]      , Z[:-1]]), 
                              np.array([-_C[:-1]      , -_C[::-1][:-1], Z[:-1]]), 
                              np.array([ _C[::-1]     , -_C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    pipe = BSpline(degrees, knots)
    return pipe, ctrlPts

def new_pipe(center_front, orientation, radius, length):
    """
    Creates a B-spline pipe from a given center, orientation, radius and length.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the pipe.
    orientation : array_like
        The normal vector of the pipe.
    radius : float
        The radius of the pipe.
    length : float
        The length of the pipe.

    Returns
    -------
    pipe : BSpline
        The pipe.
    ctrlPts : array_like
        The control points of the pipe.
    """
    pipe, ctrlPts = _base_pipe()

    # The vector to scale by
    scale_vector = np.array([radius, radius, length], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')
    
    # The normal vector of the pipe
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # The vector to translate by
    translation_vector = np.array(center_front, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return pipe, ctrlPts

# %% 3D parametric space
def _base_quarter_cylinder():
    q = 1
    r = 1
    degrees = np.array([_p, q, r], dtype='int')
    knots = [_knot, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.array([_C, _C[::-1], Z])
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    quarter_cylinder = BSpline(degrees, knots)
    return quarter_cylinder, ctrlPts

def new_quarter_cylinder(center_front, orientation, radius, length):
    """
    Creates a B-spline quarter cylinder from a given center, orientation, radius and length.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the quarter cylinder.
    orientation : array_like
        The normal vector of the quarter cylinder.
    radius : float
        The radius of the quarter cylinder.
    length : float
        The length of the quarter cylinder.

    Returns
    -------
    quarter_cylinder : BSpline
        The quarter cylinder.
    ctrlPts : array_like
        The control points of the quarter cylinder.
    """
    quarter_cylinder, ctrlPts = _base_quarter_cylinder()

    # The vector to scale by
    scale_vector = np.array([radius, radius, length], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the quarter cylinder
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # The vector to translate by
    translation_vector = np.array(center_front, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return quarter_cylinder, ctrlPts

def _base_cylinder():
    q = 1
    r = 1
    degrees = np.array([_p, q, r], dtype='int')
    knots = [np.array([0/4 + float(k)/4 for k in _knot[:-_p]] 
                    + [1/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [2/4 + float(k)/4 for k in _knot[_p:-_p]] 
                    + [3/4 + float(k)/4 for k in _knot[_p:]], dtype='float'), 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    Z = np.zeros_like(_C)
    ctrlPts = np.concatenate((np.array([ _C[:-1]      ,  _C[::-1][:-1], Z[:-1]]), 
                              np.array([-_C[::-1][:-1],  _C[:-1]      , Z[:-1]]), 
                              np.array([-_C[:-1]      , -_C[::-1][:-1], Z[:-1]]), 
                              np.array([ _C[::-1]     , -_C           , Z     ])), axis=1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts

def new_cylinder(center_front, orientation, radius, length):
    """
    Creates a B-spline cylinder from a given center, orientation, radius and length.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the cylinder.
    orientation : array_like
        The normal vector of the cylinder.
    radius : float
        The radius of the cylinder.
    length : float
        The length of the cylinder.

    Returns
    -------
    cylinder : BSpline
        The cylinder.
    ctrlPts : array_like
        The control points of the cylinder.
    """
    cylinder, ctrlPts = _base_cylinder()

    # The vector to scale by
    scale_vector = np.array([radius, radius, length], dtype='float')

    # The unit vector pointing upwards
    e_z = np.array([0, 0, 1], dtype='float')

    # The normal vector of the cylinder
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # The vector to translate by
    translation_vector = np.array(center_front, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return cylinder, ctrlPts

def _base_degenerated_cylinder():
    q = 1
    degrees = np.array([_p, _p, q], dtype='int')
    knots = [_knot, _knot, np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    Z = np.zeros_like(_C)
    n = _C.size
    ctrlPts = np.empty((3, n, n), dtype='float')
    ctrlPts[:,  0,  :] = [+_C[:: 1], +_C[::-1], Z]
    ctrlPts[:, -1,  :] = [-_C[::-1], -_C[:: 1], Z]
    ctrlPts[:,  :,  0] = [+_C[:: 1], -_C[::-1], Z]
    ctrlPts[:,  :, -1] = [-_C[::-1], +_C[:: 1], Z]
    ctrlPts = _find_inner_ctrlPts(degrees[:-1], knots[:-1], ctrlPts)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, ctrlPts), axis=-1)
    ctrlPts[2, :, :, 1] = 1
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts

def new_degenerated_cylinder(center_front, orientation, radius, length):
    """
    Creates a B-spline cylinder from a given center, orientation, radius and length.
    The cylinder is degenerated as it is created by "blowing" a square into a circle 
    before extruding it into a cylinder.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the cylinder.
    orientation : array_like
        The normal vector of the cylinder.
    radius : float
        The radius of the cylinder.
    length : float
        The length of the cylinder.

    Returns
    -------
    cylinder : BSpline
        The cylinder.
    ctrlPts : array_like
        The control points of the cylinder.
    """
    cylinder, ctrlPts = _base_degenerated_cylinder()

    # Scale the control points
    scale_vector = np.array([radius, radius, length], dtype='float')

    # Rotate the control points
    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # Translate the control points
    translation_vector = np.array(center_front, dtype='float')
    
    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return cylinder, ctrlPts

# %% Constants for closed knot vector B-Spline circles
_p_closed = 2
_knot_closed = 1/8*np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='float')
_a = -(226560/751)/np.pi**3 + (176640/751)*np.sqrt(2)/np.pi**3
_b = -(579840/751)/np.pi**3 + (403200/751)*np.sqrt(2)/np.pi**3
_C_closed = np.array([[_a, _b], [_a, -_b], [-_b, _a], [_b, _a], [-_a, -_b], [-_a, _b], [_b, -_a], [-_b, -_a], [_a, _b], [_a, -_b]], dtype='float')

# %% 1D parametric space
def _base_closed_circle():
    degrees = np.array([_p_closed], dtype='int')
    knots = [_knot_closed]
    Z = np.zeros_like(_C_closed)
    ctrlPts = _C_closed
    circle = BSpline(degrees, knots)
    return circle, ctrlPts

def new_closed_circle(center, normal, radius):
    """
    Creates a B-spline circle from a given center, normal and radius.
    The circle is closed as it the junction between the start and the end of the 
    parametric space spans enough elements to conserve the C^(p-1) continuity.

    Parameters
    ----------
    center : array_like
        The center of the circle.
    normal : array_like
        The normal vector of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    circle : BSpline
        The circle.
    ctrlPts : array_like
        The control points of the circle.
    """
    # Get the base circle
    circle, ctrlPts = _base_closed_circle()

    # Scale the control points by the radius
    scale_vector = np.array([radius, radius, 1], dtype='float')

    # Rotate the control points by the angle between the normal vector and the up vector
    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # Translate the control points by the center vector
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return circle, ctrlPts

# %% 2D parametric space
def _base_closed_disk():
    q = 1
    degrees = np.array([_p_closed, q], dtype='int')
    knots = [_knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    ctrlPts = np.expand_dims(_C_closed, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    disk = BSpline(degrees, knots)
    return disk, ctrlPts

def new_closed_disk(center, normal, radius):
    """
    Creates a B-spline disk from a given center, normal and radius.
    The disk is closed as it the junction between the start and the end of the 
    parametric space spans enough elements to conserve the C^(p-1) continuity.

    Parameters
    ----------
    center : array_like
        The center of the disk.
    normal : array_like
        The normal vector of the disk.
    radius : float
        The radius of the disk.

    Returns
    -------
    disk : BSpline
        The disk.
    ctrlPts : array_like
        The control points of the disk.
    """
    disk, ctrlPts = _base_closed_disk()

    # Scale the control points by the radius
    scale_vector = np.array([radius, radius, 1], dtype='float')

    # Rotate the control points by the angle between the normal vector and the up vector
    e_z = np.array([0, 0, 1], dtype='float')
    normal = np.array(normal, dtype='float')
    normal /= np.linalg.norm(normal)
    axis = np.cross(e_z, normal)
    angle = np.arccos(np.dot(e_z, normal))

    # Translate the control points by the center vector
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return disk, ctrlPts

def _base_closed_pipe():
    q = 1
    degrees = np.array([_p_closed, q], dtype='int')
    knots = [_knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float')]
    ctrlPts = np.expand_dims(_C_closed, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    pipe = BSpline(degrees, knots)
    return pipe, ctrlPts

def new_closed_pipe(center_front, orientation, radius, length):
    """
    Creates a B-spline closed pipe from a given center, orientation, radius and length.
    The pipe is closed as the junction between the start and the end of the 
    parametric space along the circular direction spans enough elements to conserve the
    C^(p-1) continuity.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the pipe.
    orientation : array_like
        The normal vector of the pipe.
    radius : float
        The radius of the pipe.
    length : float
        The length of the pipe.

    Returns
    -------
    pipe : BSpline
        The pipe.
    ctrlPts : array_like
        The control points of the pipe.
    """
    pipe, ctrlPts = _base_closed_pipe()

    # Scale the control points by the radius and length
    scale_vector = np.array([radius, radius, length], dtype='float')

    # Rotate the control points by the angle between the normal vector and the up vector
    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # Translate the control points by the center vector
    translation_vector = np.array(center_front, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)

    return pipe, ctrlPts

# %% 3D parametric space
def _base_closed_cylinder():
    q = 1
    r = 1
    degrees = np.array([_p_closed, q, r], dtype='int')
    knots = [_knot_closed, 
             np.array([0]*(q+1) + [1]*(q+1), dtype='float'), 
             np.array([0]*(r+1) + [1]*(r+1), dtype='float')]
    ctrlPts = np.expand_dims(_C_closed, axis=-1)
    ctrlPts = np.concatenate((ctrlPts, np.zeros_like(ctrlPts)), axis=-1)
    ctrlPts = np.expand_dims(ctrlPts, axis=-1)
    ctrlPts_m = ctrlPts.copy()
    ctrlPts_m[2] = 0
    ctrlPts_p = ctrlPts.copy()
    ctrlPts_p[2] = 1
    ctrlPts = np.concatenate((ctrlPts_m, ctrlPts_p), axis=-1)
    cylinder = BSpline(degrees, knots)
    return cylinder, ctrlPts

def new_closed_cylinder(center_front, orientation, radius, length):
    """
    Creates a B-spline closed cylinder from a given center, orientation, radius and length.
    The cylinder is closed as the junction between the start and the end of the 
    parametric space along the circular direction spans enough elements to conserve the
    C^(p-1) continuity.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the closed cylinder.
    orientation : array_like
        The normal vector of the closed cylinder.
    radius : float
        The radius of the closed cylinder.
    length : float
        The length of the closed cylinder.

    Returns
    -------
    closed_cylinder : BSpline
        The closed cylinder.
    ctrlPts : array_like
        The control points of the closed cylinder.
    """
    cylinder, ctrlPts = _base_closed_cylinder()

    # Scale the control points by the radius and length
    scale_vector = np.array([radius, radius, length], dtype='float')
    
    # Rotate the control points by the angle between the normal vector and the up vector
    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))
    
    # Translate the control points by the center vector
    translation_vector = np.array(center_front, dtype='float')
    
    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)
    
    return cylinder, ctrlPts

# %% strut elements

def new_quarter_strut(center_front, orientation, radius, length):
    """
    Creates a B-spline quarter strut from a given center, orientation, radius and length.

    Parameters
    ----------
    center_front : array_like
        The center of the front of the quarter strut.
    orientation : array_like
        The normal vector of the quarter strut.
    radius : float
        The radius of the quarter strut.
    length : float
        The length of the quarter strut.

    Returns
    -------
    quarter_strut : BSpline
        The quarter strut.
    ctrlPts : array_like
        The control points of the quarter strut.
    """
    degrees = np.array([2, 1, 1], dtype='int')
    knots = [np.array([0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1], dtype='float'), 
            np.array([0, 0, 1, 1], dtype='float'), 
            np.array([0, 0, 1, 1], dtype='float')]
    C = np.array([-41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
                  -41687040/36473*np.sqrt(2)/np.pi**3 - (4884480/36473)/np.pi**3 - 221/72946*np.pi + (14069760/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3 + (50933760/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3, 
                  -50557440/36473*np.sqrt(np.sqrt(2) + 2)/np.pi**3 - 92505600/36473*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (26527/2334272)*np.pi + (18163200/36473)/np.pi**3 + (103925760/36473)*np.sqrt(2)/np.pi**3, 
                  -91238400/36473*np.sqrt(2)/np.pi**3 - (41034240/36473)/np.pi**3 - 65605/2334272*np.pi + (52646400/36473)*np.sqrt(2 - np.sqrt(2))/np.pi**3 + (70632960/36473)*np.sqrt(np.sqrt(2) + 2)/np.pi**3, 
                  (1/16)*np.pi, 
                  0], dtype='float')*radius

    tmp = radius*np.ones_like(C)
    front = np.array([C, C[::-1], tmp])
    tmp = np.zeros_like(front)
    front = np.concatenate((front[..., None], tmp[..., None]), axis=-1)

    tmp = length - radius*np.ones_like(C)
    back = np.array([C, C[::-1], tmp])
    tmp = np.zeros_like(back)
    tmp[-1] = length
    back = np.concatenate((back[..., None], tmp[..., None]), axis=-1)

    ctrlPts = np.concatenate((front[..., None], back[..., None]), axis=-1)

    spline = BSpline(degrees, knots)
    ctrlPts = spline.orderElevation(ctrlPts, [0, 0, 1])
    ctrlPts = spline.knotInsertion(ctrlPts, [np.array([]), np.array([]), np.array([0.25, 0.5, 0.75], dtype='float')])

    p = np.arctan(3 - 2*np.sqrt(2))
    k = np.linspace(p, np.pi/2 - p, C.size)
    x = radius*np.sqrt(2)*((0.5 + 0.25*np.sqrt(2))*np.cos(k) + (0.25*np.sqrt(2) - 0.5)*np.sin(k))
    y = radius*np.sqrt(2)*((0.5 + 0.25*np.sqrt(2))*np.sin(k) + (0.25*np.sqrt(2) - 0.5)*np.cos(k))
    z = radius*(0.5*np.cos(k) + 0.5*np.sin(k))

    p = 0
    k = np.linspace(p, np.pi/2 - p, C.size)
    x = radius*np.cos(k)
    y = radius*np.sin(k)
    z = radius/np.sqrt(2)*(np.cos(k) + np.sin(k))
    
    # x = radius*np.linspace(np.sqrt(2), 0, C.size)
    # y = radius*np.linspace(0, np.sqrt(2), C.size)
    # ctrlPts[0, :, 0,  0] = x
    # ctrlPts[1, :, 0,  0] = y
    ctrlPts[2, :, 0,  0] = z
    # ctrlPts[0, :, 0, -1] = x
    # ctrlPts[1, :, 0, -1] = y
    ctrlPts[2, :, 0, -1] = length - z

    scale_vector = np.array([1, 1, 1], dtype='float')

    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    translation_vector = np.array(center_front, dtype='float')

    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)
    
    return spline, ctrlPts

# %% cube

def new_cube(center, orientation, side_length):
    """
    Creates a B-spline cube from a given center, orientation and side length.

    Parameters
    ----------
    center : array_like
        The center of the cube.
    orientation : array_like
        The normal vector of the cube.
    side_length : float
        The side length of the cube.

    Returns
    -------
    cube : BSpline
        The cube.
    ctrlPts : array_like
        The control points of the cube.
    """
    degrees = np.array([1]*3, dtype='int')
    knots = [np.array([0, 0, 1, 1], dtype='float')]*3
    # Create a 3D grid of control points
    ctrlPts = np.array(np.meshgrid(*([np.array([-0.5, 0.5])]*3), indexing='ij'))

    spline = BSpline(degrees, knots)
    
    # Scale the control points by the side length
    scale_vector = np.array([side_length]*3, dtype='float')

    # Rotate the control points by the angle between the normal vector and the up vector
    e_z = np.array([0, 0, 1], dtype='float')
    orientation = np.array(orientation, dtype='float')
    orientation /= np.linalg.norm(orientation)
    axis = np.cross(e_z, orientation)
    angle = np.arccos(np.dot(e_z, orientation))

    # Translate the control points by the center vector
    translation_vector = np.array(center, dtype='float')

    # Apply the transformations to the control points
    ctrlPts = scale_rotate_translate(ctrlPts, scale_vector, axis, angle, translation_vector)
    
    return spline, ctrlPts
