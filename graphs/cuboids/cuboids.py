# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import new_cube, BSpline
np.random.seed(4)
c3D, pts3D = new_cube([0, 0, 0], [0, 0, 1], 1)
pts3D = c3D.orderElevation(pts3D, [1, 1, 1])
pts3D += np.random.normal(0, 5e-2, size=pts3D.shape)
c3D.saveParaview(pts3D, "./", "cuboid", n_eval_per_elem=20)
c2D = BSpline(c3D.getDegrees()[:-1], c3D.getKnots()[:-1])
pts2D = pts3D[..., -1]
c2D.saveParaview(pts2D, "./", "squaroid", n_eval_per_elem=20)
c1D = BSpline(c2D.getDegrees()[:-1], c2D.getKnots()[:-1])
pts1D = pts2D[..., -1]
c1D.saveParaview(pts1D, "./", "linoid", n_eval_per_elem=20)