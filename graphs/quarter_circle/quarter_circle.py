# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from bsplyne import new_quarter_circle
import tikzplotlib
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

c, pts = new_quarter_circle([0, 0, 0], [0, 0, 1], 1)
pts = pts[:2]
c.bases[0].plotN(show=False)
ax = plt.gca()
colors = [line.get_color() for line in ax.get_lines()]
el_borders = np.unique(c.bases[0].knot)
ax.set_xticks(el_borders)
ax.set_xticklabels(['']*len(el_borders))
ax.set_xticks(0.5*(el_borders[1:] + el_borders[:-1]), minor=True)
ax.set_xticklabels([f'$el_{i}$' for i in range(len(el_borders) - 1)], minor=True)
ax.tick_params(which='minor', length=0)
plt.legend(bbox_to_anchor=(1.3, 0.25))
tikzplotlib_fix_ncols(plt.gcf())
tikzplotlib.save("./quarter_circle_basis.tikz", extra_axis_parameters=[r'xtick={' + r', '.join(el_borders.astype('str')) + r'}', 
                                                                       r'xticklabels={' + r', '.join([r'\empty']*el_borders.size) + r'}', 
                                                                       r'extra x ticks={' + r', '.join((0.5*(el_borders[1:] + el_borders[:-1])).astype('str')) + r'}', 
                                                                       r'extra x tick labels={' + r', '.join([r'\(\displaystyle el_' + str(i) + r'\)' for i in range(el_borders.size - 1)]) + r'}', 
                                                                       r'extra x tick style={major tick length=0ex}', 
                                                                       r'minor xtick={}', 
                                                                       r'minor xticklabels={}'])
plt.show()

XI = c.linspace()
xi = XI[0]
fig, ax = plt.subplots()
ax.scatter(*pts, c=colors)
rgb = np.array([[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors], dtype='int').T
seg_rgb = c(rgb, XI)
seg_rgb = ((seg_rgb[:, 1:] + seg_rgb[:, :-1])//2).astype('int')
seg_colors = [f'#{r:02x}{g:02x}{b:02x}' for r, g, b in seg_rgb.T]
vals = c(pts, XI)
seg = np.concatenate((vals[:, None, :-1].T, vals[:, None, 1:].T), axis=1)
spline_line1 = ax.add_collection(LineCollection(seg, colors=seg_colors, linewidth=5))
ind_ann = 16
ax.annotate(
    'Original\nB-spline line',# The text to display
    xy=1.01*vals[:, ind_ann],       # The point to annotate (x, y)
    xytext=(1, 0.9),           # The position for the text (x, y)
    arrowprops=dict(facecolor='k', arrowstyle='->', connectionstyle="angle,angleA=0,angleB=90,rad=10"), # Arrow properties
    fontsize=12,                # Font size of the text
    color='k',                  # Color of the text
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor=seg_colors[ind_ann]),  # Text box properties
    ha='center',                # Horizontal alignment of the text
    va='center'                 # Vertical alignment of the text
)
ax.plot(np.cos(np.pi/2*xi), np.sin(np.pi/2*xi), c='wheat', linewidth=1)
a = 0.35*np.pi
ax.annotate(
    'Exact quarter circle',      # The text to display
    xy=(np.cos(a), np.sin(a)),   # The point to annotate (x, y)
    xytext=(0.75, 1.15),           # The position for the text (x, y)
    arrowprops=dict(facecolor='k', arrowstyle='->', connectionstyle="angle,angleA=0,angleB=90,rad=10"), # Arrow properties
    fontsize=12,                # Font size of the text
    color='k',                  # Color of the text
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor="wheat"),  # Text box properties
    ha='center',                # Horizontal alignment of the text
    va='center'                 # Vertical alignment of the text
)
vals = c(pts, [el_borders])
eps = 2e-3
el_borders[0] += eps
el_borders[-1] -= eps
seg = np.concatenate((c(pts, [el_borders - eps])[:, None, :].T, c(pts, [el_borders + eps])[:, None, :].T), axis=1)
ax.add_collection(LineCollection(seg, color='k', linewidth=20))
locs = 0.5*(vals[:, 1:] + vals[:, :-1])
for i in range(locs.shape[1]):
    ax.annotate(f"$el_{i}$", locs[:, i]*1.15, fontsize=20, ha='center', va="center")
i = 2
init = pts[:, i].copy()
pts[:, i] *= 0.75
ax.scatter(*pts, c=colors, alpha=0.25)
ax.arrow(*0.95*init, *(1.1*pts[:, i] - 0.95*init), color='k', width=0.01, zorder=100)
for i in range(c.bases[0].n + 1):
    ax.annotate(f"$P_{i}$", pts[:, i]*0.9, c=colors[i], fontsize=20, ha='center', va="center")
vals = c(pts, XI)
seg = np.concatenate((vals[:, None, :-1].T, vals[:, None, 1:].T), axis=1)
ax.add_collection(LineCollection(seg, colors=seg_colors, linewidth=5, alpha=0.25))
ind_ann = 9
ax.annotate(
    'Deformed\nB-spline line', # The text to display
    xy=0.99*vals[:, ind_ann],       # The point to annotate (x, y)
    xytext=(0.3, 0.2),           # The position for the text (x, y)
    arrowprops=dict(facecolor='k', arrowstyle='->', connectionstyle="angle,rad=10"), # Arrow properties
    fontsize=12,                # Font size of the text
    color='k',                  # Color of the text
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor=seg_colors[ind_ann]),  # Text box properties
    ha='center',                # Horizontal alignment of the text
    va='center'                 # Vertical alignment of the text
)
# ax.annotate(r"$\xi = 0$", pts[:, 0]*0.8, c=colors[0], fontsize=15, ha='right', va="center")
# ax.annotate(r"$\xi = 1$", pts[:, -1]*0.8, c=colors[-1], fontsize=15, ha='center', va="center")
ax.set_xlim(-0.1, 1.25)
ax.set_ylim(-0.1, 1.25)
ax.set_aspect(1)
ax.axis('off')
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("./quarter_circle_bspline.tikz", extra_axis_parameters=['axis equal'])
plt.show()


# %%
from bsplyne import new_quarter_pipe
from bsplyne.geometries_in_3D import _scale_rotate_translate

spline, ctrl_pts = new_quarter_pipe([0, 0, 0], [0, 0, 1], 1, 5)
ctrl_pts = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [0, 0, 1], -np.pi/4, [0, 0, 0])
k = 2/ctrl_pts[0,  :, -1]
ctrl_pts[0, :, -1] = 2*np.ones_like(k)
ctrl_pts[1, :, -1] *= k
ctrl_pts1 = ctrl_pts
ctrl_pts2 = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [0, 0, 1], np.pi/2, [0, 0, 0])
ctrl_pts3 = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [0, 0, 1], np.pi, [0, 0, 0])
ctrl_pts4 = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [0, 0, 1], 3*np.pi/2, [0, 0, 0])

colors = ['b', 'y', 'm', 'c']
fig, ax = plt.subplots()
spline.plot(ctrl_pts1[:2], ax=ax, interior_color=colors[0])
spline.plot(ctrl_pts2[:2], ax=ax, interior_color=colors[1])
spline.plot(ctrl_pts3[:2], ax=ax, interior_color=colors[2])
spline.plot(ctrl_pts4[:2], ax=ax, interior_color=colors[3])

for i, p in enumerate(ax.patches):
    p._label = f"Patch {i+1}"
for l in ax.lines[1:-1]:
    if l._label[:6]!="_child":
        l.remove()
ax.set_aspect(1)
# ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 6])
# ax.legend(loc="upper right")
ax.legend(bbox_to_anchor=(1.7, 0.25))
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("./multipatch_issue.tikz", extra_axis_parameters=['axis equal image', r'font=\footnotesize'])
plt.show()
# %%
from bsplyne import BSpline
from bsplyne.geometries_in_3D import _scale_rotate_translate

ctrl_pts = np.array([[[0   , 0.2 , 0.5 , 0.8 , 1   ], 
                      [0   , 0.15, 0.45, 0.6 , 0.6 ]], 
                     [[0   , 0.2 , 0.5 , 0.8 , 1   ], 
                      [0.4 , 0.4 , 0.55, 0.85, 1   ]], 
                     [[0   , 0   , 0   , 0   , 0   ], 
                      [0   , 0   , 0   , 0   , 0   ]]])
degrees = [1, 2]
knots = [np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0.33, 0.66, 1, 1, 1])]
spline = BSpline(degrees, knots)

ctrl_pts1 = ctrl_pts
ctrl_pts2 = _scale_rotate_translate(ctrl_pts, [1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), 0], np.pi, [0, 0, 0])
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
spline.plot(ctrl_pts11[:2], ax=ax, interior_color=colors[0][0])
spline.plot(ctrl_pts12[:2], ax=ax, interior_color=colors[0][1])
spline.plot(ctrl_pts21[:2], ax=ax, interior_color=colors[1][0])
spline.plot(ctrl_pts22[:2], ax=ax, interior_color=colors[1][1])
spline.plot(ctrl_pts31[:2], ax=ax, interior_color=colors[2][0])
spline.plot(ctrl_pts32[:2], ax=ax, interior_color=colors[2][1])
spline.plot(ctrl_pts41[:2], ax=ax, interior_color=colors[3][0])
spline.plot(ctrl_pts42[:2], ax=ax, interior_color=colors[3][1])

for i, p in enumerate(ax.patches):
    p._label = f"Patch {i+1}"
for l in ax.lines[1:-1]:
    if l._label[:6]!="_child":
        l.remove()
ax.set_aspect(1)
# ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 6])
# ax.legend(loc="upper right")
ax.legend(bbox_to_anchor=(1.7, 0.25))
ax.axis('off')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("./multipatch_issue.tikz", extra_axis_parameters=['axis equal image', r'font=\footnotesize'])
plt.show()
# %%
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
spline.plot(ctrl_pts11[:2], ax=ax, interior_color=colors[0][0])
spline.plot(ctrl_pts12[:2], ax=ax, interior_color=colors[0][1])
spline.plot(ctrl_pts21[:2], ax=ax, interior_color=colors[1][0])
spline.plot(ctrl_pts22[:2], ax=ax, interior_color=colors[1][1])
spline.plot(ctrl_pts31[:2], ax=ax, interior_color=colors[2][0])
spline.plot(ctrl_pts32[:2], ax=ax, interior_color=colors[2][1])
spline.plot(ctrl_pts41[:2], ax=ax, interior_color=colors[3][0])
spline.plot(ctrl_pts42[:2], ax=ax, interior_color=colors[3][1])

for i, p in enumerate(ax.patches):
    p._label = f"Patch {i+1}"
for l in ax.lines[1:-1]:
    if l._label[:6]!="_child":
        l.remove()
ax.set_aspect(1)
# ax.set_xlim([ax.get_xlim()[0], ax.get_xlim()[1] + 6])
# ax.legend(loc="upper right")
ax.legend(bbox_to_anchor=(1.7, 0.25))

ax.legend(*list(zip(*[(h, l) for h, l in zip(*ax.get_legend_handles_labels()) if l[:5] != 'Patch'])))

ax.axis('off')
tikzplotlib_fix_ncols(fig)
tikzplotlib.save("./multipatch_issue.tikz", extra_axis_parameters=['axis equal image', r'font=\footnotesize'])
plt.show()
# %%
