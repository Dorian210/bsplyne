# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import BSpline

import os

def b_spline_from_cubic_bezier_pts(pts): # pts : (2, n)
    nb_elem = pts.shape[1]//3
    p = 3
    knot = np.concatenate(([0], np.repeat(np.linspace(0, 1, nb_elem + 1), p), [1]))
    spline = BSpline([p], [knot])
    return spline

ctrl_pts1 = np.array([[7.42466e+0, 1.85180e+2], 
                      [4.14907e+1, 8.99694e+1], 
                      [1.00451e+2, 2.09637e+1], 
                      [1.95225e+2, 1.57228e+1], 
                      [2.89998e+2, 1.09186e+1], 
                      [3.37604e+2, 1.19668e+2], 
                      [2.86941e+2, 2.19246e+2], 
                      [2.35842e+2, 3.19697e+2], 
                      [1.34517e+2, 3.84335e+2], 
                      [8.64754e+1, 3.42408e+2], 
                      [3.75600e+1, 2.99607e+2], 
                      [6.33280e+1, 2.17062e+2], 
                      [9.43368e+1, 1.77318e+2], 
                      [1.24909e+2, 1.35391e+2], 
                      [2.16189e+2, 3.23191e+1], 
                      [3.80405e+2, 3.53763e+1], 
                      [5.43310e+2, 3.84335e+1], 
                      [5.48114e+2, 1.13554e+2], 
                      [5.47678e+2, 1.50240e+2], 
                      [5.47241e+2, 1.86927e+2], 
                      [5.21910e+2, 2.45450e+2], 
                      [4.55088e+2, 2.72965e+2], 
                      [3.88703e+2, 2.98733e+2], 
                      [3.86082e+2, 2.39336e+2], 
                      [4.55088e+2, 2.68598e+2], 
                      [5.24530e+2, 2.97423e+2], 
                      [5.29334e+2, 3.41098e+2], 
                      [5.25404e+2, 3.97438e+2], 
                      [5.21473e+2, 4.53341e+2], 
                      [4.60329e+2, 5.27587e+2], 
                      [3.86956e+2, 5.28024e+2], 
                      [3.13583e+2, 5.29771e+2], 
                      [3.20134e+2, 4.72558e+2], 
                      [3.19697e+2, 4.60329e+2], 
                      [3.19260e+2, 4.48973e+2], 
                      [3.18824e+2, 4.03552e+2], 
                      [3.74727e+2, 3.59878e+2]], dtype='float').T

ctrl_pts2 = np.array([[2.66414e+1, 4.60329e+2], 
                      [9.17164e+0, 4.93521e+2], 
                      [4.89154e+1, 5.38069e+2], 
                      [1.31460e+2, 5.10118e+2], 
                      [2.14005e+2, 4.82603e+2], 
                      [2.82137e+2, 3.70796e+2], 
                      [3.47212e+2, 2.41083e+2], 
                      [4.13160e+2, 1.11370e+2], 
                      [4.65570e+2, 6.15810e+1], 
                      [5.32828e+2, 2.53312e+1]], dtype='float').T

spline1 = b_spline_from_cubic_bezier_pts(ctrl_pts1)
spline2 = b_spline_from_cubic_bezier_pts(ctrl_pts2)

xi = np.linspace(0, 1, 1000)
x1, y1 = spline1(ctrl_pts1, [xi])
x2, y2 = spline2(ctrl_pts2, [xi])
fig, ax = plt.subplots()
ls = ":"
ax.scatter(ctrl_pts1[0], -ctrl_pts1[1], c='#999999')
ax.plot(ctrl_pts1[0], -ctrl_pts1[1], '#999999', linestyle=ls)
ax.scatter(ctrl_pts2[0], -ctrl_pts2[1], c='#999999')
ax.plot(ctrl_pts2[0], -ctrl_pts2[1], '#999999', linestyle=ls)
ax.plot(x1, -y1, '#000000')
ax.plot(x2, -y2, '#000000')
ax.set_aspect(1)
ax.set_axis_off()
fig.savefig(os.path.join(os.getcwd(), "logo.png"), dpi=300, bbox_inches='tight', pad_inches=0)

# %%
import scipy.sparse as sps
nb_pts1 = 25
p1 = 5
knot1 = np.concatenate(([0]*p1, np.linspace(0, 1, nb_pts1 - p1 + 1), [1]*p1))
bspline1 = BSpline([p1], [knot1])
xi = np.linspace(0, 1, 1000)
N1 = bspline1.DN([xi])
A1 = (N1.T@N1).A
A1[ 0,  :] = 0
# A1[ :,  0] = 0
A1[ 0,  0] = 1
A1[-1,  :] = 0
# A1[ :, -1] = 0
A1[-1, -1] = 1
bx1, by1 = spline1(ctrl_pts1, [xi])@N1
bx1[ 0] = ctrl_pts1[0,  0]
bx1[-1] = ctrl_pts1[0, -1]
by1[ 0] = ctrl_pts1[1,  0]
by1[-1] = ctrl_pts1[1, -1]
ctrl_ptsb1 = np.array([np.linalg.solve(A1, bx1), np.linalg.solve(A1, by1)])

nb_pts2 = 5
p2 = 3
knot2 = np.concatenate(([0]*p2, np.linspace(0, 1, nb_pts2 - p2 + 1), [1]*p2))
bspline2 = BSpline([p2], [knot2])
xi = np.linspace(0, 1, 1000)
N2 = bspline2.DN([xi])
A2 = sps.block_diag([N2, N2])
b2 = spline2(ctrl_pts2, [xi]).ravel()
ctrl_ptsb2 = np.linalg.solve((A2.T@A2).A, A2.T@b2).reshape((2, -1))

xi = np.linspace(0, 1, 1000)
xb1, yb1 = bspline1(ctrl_ptsb1, [xi])
xb2, yb2 = bspline2(ctrl_ptsb2, [xi])
fig, ax = plt.subplots()
ls = ":"
ax.scatter(ctrl_ptsb1[0], -ctrl_ptsb1[1], c='#999999')
ax.plot(ctrl_ptsb1[0], -ctrl_ptsb1[1], '#999999', linestyle=ls)
ax.scatter(ctrl_ptsb2[0], -ctrl_ptsb2[1], c='#999999')
ax.plot(ctrl_ptsb2[0], -ctrl_ptsb2[1], '#999999', linestyle=ls)
ax.plot(xb1, -yb1, '#000000')
ax.plot(xb2, -yb2, '#000000')
ax.set_aspect(1)
ax.set_axis_off()

bspline1.plotMPL(ctrl_ptsb1*np.array([1, -1])[:, None], n_eval_per_elem=20)
ax = plt.gca()
bspline2.plotMPL(ctrl_ptsb2*np.array([1, -1])[:, None], n_eval_per_elem=20, ax=ax)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
ax.set_axis_off()
# %%