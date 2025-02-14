# %%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import matplotlib.colors as mcolors
from bsplyne import BSpline
from bsplyne.geometries_in_3D import _scale_rotate_translate

ctrl_pts = np.array([[[0   , 0.2 , 0.5 , 0.8 , 1   ], 
                      [0   , 0.15, 0.45, 0.6 , 0.6 ]], 
                     [[0   , 0.2 , 0.5 , 0.8 , 1   ], 
                      [0.4 , 0.4 , 0.55, 0.85, 1   ]], 
                     [[0   , 0   , 0   , 0   , 0   ], 
                      [0   , 0   , 0   , 0   , 0   ]]])
ctrl_pts = np.array([ctrl_pts[1], 1-ctrl_pts[0], ctrl_pts[2]])
degrees = [1, 2]
knots = [np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0.33, 0.66, 1, 1, 1])]
spline = BSpline(degrees, knots)

ctrl_pts1 = ctrl_pts
ctrl_pts2 = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [1/np.sqrt(2), -1/np.sqrt(2), 0], np.pi, [1, 1, 0])
ctrl_pts11 = ctrl_pts1
ctrl_pts12 = ctrl_pts2
ctrl_pts21 = _scale_rotate_translate(ctrl_pts1, [1, 1, 1], [0, 0, 1], np.pi/2, [0, 0, 0])
ctrl_pts22 = _scale_rotate_translate(ctrl_pts2, [1, 1, 1], [0, 0, 1], np.pi/2, [0, 0, 0])
ctrl_pts31 = _scale_rotate_translate(ctrl_pts1, [1, 1, 1], [0, 0, 1], np.pi, [0, 0, 0])
ctrl_pts32 = _scale_rotate_translate(ctrl_pts2, [1, 1, 1], [0, 0, 1], np.pi, [0, 0, 0])
ctrl_pts41 = _scale_rotate_translate(ctrl_pts1, [1, 1, 1], [0, 0, 1], 3*np.pi/2, [0, 0, 0])
ctrl_pts42 = _scale_rotate_translate(ctrl_pts2, [1, 1, 1], [0, 0, 1], 3*np.pi/2, [0, 0, 0])

colors = [['tab:blue'  , 'tab:orange'], 
          ['tab:purple', 'tab:brown' ], 
          ['tab:pink'  , 'tab:gray'  ], 
          ['tab:olive' , 'tab:cyan'  ]]
fig, ax = plt.subplots()
spline.plotMPL(ctrl_pts11[:2], ax=ax, interior_color=colors[0][0])
spline.plotMPL(ctrl_pts12[:2], ax=ax, interior_color=colors[0][1])
spline.plotMPL(ctrl_pts21[:2], ax=ax, interior_color=colors[1][0])
spline.plotMPL(ctrl_pts22[:2], ax=ax, interior_color=colors[1][1])
spline.plotMPL(ctrl_pts31[:2], ax=ax, interior_color=colors[2][0])
spline.plotMPL(ctrl_pts32[:2], ax=ax, interior_color=colors[2][1])
spline.plotMPL(ctrl_pts41[:2], ax=ax, interior_color=colors[3][0])
spline.plotMPL(ctrl_pts42[:2], ax=ax, interior_color=colors[3][1])
spline.plotMPL(ctrl_pts11[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[0][0])])
spline.plotMPL(ctrl_pts12[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[0][1])])
spline.plotMPL(ctrl_pts21[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[1][0])])
spline.plotMPL(ctrl_pts22[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[1][1])])
spline.plotMPL(ctrl_pts31[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[2][0])])
spline.plotMPL(ctrl_pts32[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[2][1])])
spline.plotMPL(ctrl_pts41[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[3][0])])
spline.plotMPL(ctrl_pts42[:2] + np.array([2, 0]).reshape((2, 1, 1)), ax=ax, interior_color=[x*0.5 for x in mcolors.to_rgb(colors[3][1])])

ax.set_aspect(1)
indices = [1, 2, 3]
legend = ax.get_legend()
ax.legend([h for i, h in enumerate(legend.legend_handles) if i in indices], 
          [t.get_text() if i!=indices[-1] else t.get_text() + r' (\$C^0\$ lines)' for i, t in enumerate(legend.get_texts()) if i in indices], 
          loc='center', bbox_to_anchor=(0.5, 1.25))
ax.axis('off')
fig = plt.gcf()
fig.set_size_inches(3, 2.25)
fig.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("./multipatch_issue.svg")
plt.show()

# %%
