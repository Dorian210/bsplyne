import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
from bsplyne_lib import BSpline

def test___init__():
    degrees = np.array([2, 2], dtype='int')
    knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
             np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
    ctrlPts = np.random.rand(3, 4, 4)
    spline = BSpline(ctrlPts, degrees, knots)
    assert (    spline.NPa==len(knots) 
            and len(spline.bases)==len(knots) 
            and np.all([spline.bases[i].p==degrees[i] for i in range(spline.NPa)]) 
            and np.all([np.allclose(spline.bases[i].knot, knots[i]) for i in range(spline.NPa)]) 
            and spline.NPh==ctrlPts.shape[0] 
            and np.allclose(spline.ctrlPts, ctrlPts))

def test_DN():
    degrees = np.array([2, 2], dtype='int')
    knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
             np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
    ctrlPts = np.random.rand(3, 4, 4)
    spline = BSpline(ctrlPts, degrees, knots)

    exp = 1e-5
    size = [11]*spline.NPa
    XI_grid = [np.linspace(0, 1-exp, s) for s in size]
    XI = np.array(np.meshgrid(*XI_grid, indexing='ij'))
    dXI_grid = [exp*(0.9*np.random.rand(s) + 0.1) for s in size]
    dXI = np.array(np.meshgrid(*dXI_grid, indexing='ij'))
    k = np.zeros(spline.NPa)
    res = True
    for axis in range(spline.NPa):
        XI_dXI = XI.copy()
        XI_dXI[axis] += dXI[axis]
        dk = k.copy()
        dk[axis] = 1
        residu = spline.DN(XI_dXI, k) - spline.DN(XI, k) - spline.DN(XI, dk).multiply(dXI[axis].reshape((-1, 1)))
        res = res or np.all(np.linalg.norm(residu.A, axis=1)<10*np.abs(dXI[axis].ravel()**2))
    assert res

def test_DN():
    degrees = np.array([2, 2], dtype='int')
    knots = [np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float'), 
             np.array([0, 0, 0, 0.5, 1, 1, 1], dtype='float')]
    ctrlPts = np.random.rand(3, 4, 4)
    spline = BSpline(ctrlPts, degrees, knots)

    exp = 1e-5
    size = [11]*spline.NPa
    XI_grid = [np.linspace(0, 1-exp, s) for s in size]
    dXI_grid = [exp*(0.9*np.random.rand(s) + 0.1) for s in size]
    dXI = np.array(np.meshgrid(*dXI_grid, indexing='ij'))
    k = np.zeros(spline.NPa)
    res = True
    for axis in range(spline.NPa):
        XI_dXI_grid = [xi.copy() for xi in XI_grid]
        XI_dXI_grid[axis] += dXI_grid[axis]
        dk = k.copy()
        dk[axis] = 1
        residu = spline.DN(XI_dXI_grid, k) - spline.DN(XI_grid, k) - spline.DN(XI_grid, dk).multiply(dXI[axis].reshape((-1, 1)))
        res = res or np.all(np.linalg.norm(residu.A, axis=1)<10*np.abs(dXI[axis].ravel()**2))
    assert res