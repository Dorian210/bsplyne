import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from bsplyne_lib import BSplineBasis

def test___init__():
    p = 2
    knot = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
    basis = BSplineBasis(p, knot)
    assert (basis.p==p 
            and np.all(basis.knot==knot) 
            and basis.m==knot.size - 1 
            and basis.n==basis.m - p - 1 
            and basis.span==(basis.knot[basis.p], basis.knot[basis.m - basis.p]))

def test_N():
    p = 2
    knot = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
    basis = BSplineBasis(p, knot)
    XI = np.linspace(0, 1, 11)
    N = np.array([(XI<=0.5)*( 4*XI**2 - 4*XI + 1)                                 , 
                  (XI<=0.5)*(-6*XI**2 + 4*XI + 0) + (XI>0.5)*( 2*XI**2 - 4*XI + 2), 
                  (XI<=0.5)*( 2*XI**2 + 0*XI + 0) + (XI>0.5)*(-6*XI**2 + 8*XI - 2), 
                                                    (XI>0.5)*( 4*XI**2 - 4*XI + 1)], dtype='float')
    DN = np.array([(XI<=0.5)*(  8*XI - 4)                        , 
                   (XI<=0.5)*(-12*XI + 4) + (XI>0.5)*(  4*XI - 4), 
                   (XI<=0.5)*(  4*XI + 0) + (XI>0.5)*(-12*XI + 8), 
                                            (XI>0.5)*(  8*XI - 4)], dtype='float')
    assert np.allclose(N.T, basis.N(XI).A) and np.allclose(DN.T, basis.N(XI, 1).A)

def test_knotInsertion():
    p = 2
    knot = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
    basis = BSplineBasis(p, knot)
    ctrlPts = np.array([[0, 1, 1, 0], [0, 1, 2, 3]], dtype='float')
    XI = np.linspace(0, 1, 11)
    pts_before = (basis.N(XI) @ ctrlPts.T).T
    knots_to_add = np.array([0.5, 0.75], dtype='float')
    D = basis.knotInsertion(knots_to_add)
    ctrlPts = (D@ctrlPts.T).T
    pts_after = (basis.N(XI) @ ctrlPts.T).T
    assert np.allclose(pts_before, pts_after)

def test_orderElevation():
    p = 2
    knot = np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')
    basis = BSplineBasis(p, knot)
    ctrlPts = np.array([[0, 1, 1, 0], [0, 1, 2, 3]], dtype='float')
    XI = np.linspace(0, 1, 11)
    pts_before = (basis.N(XI) @ ctrlPts.T).T
    t = 2
    STD = basis.orderElevation(t)
    ctrlPts = (STD@ctrlPts.T).T
    pts_after = (basis.N(XI) @ ctrlPts.T).T
    assert np.allclose(pts_before, pts_after)
