import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from bsplyne_lib import MultiPatchBSplineConnectivity

def test___init__():
    unique_nodes_inds = 
    shape_by_patch
    degrees = np.array([2, 2], dtype='int')
    knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
             np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
    spline = BSpline(degrees, knots)
    assert (    spline.NPa==len(knots) 
            and len(spline.bases)==len(knots) 
            and np.all([spline.bases[i].p==degrees[i] for i in range(spline.NPa)]) 
            and np.all([np.allclose(spline.bases[i].knot, knots[i]) for i in range(spline.NPa)]))
    