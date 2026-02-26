# %%
import numpy as np
np.random.seed(2)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from bsplyne import new_cube, MultiPatchBSplineConnectivity

# Generates three consistent shades of a base color
def make_triplet(hex_color, sats=(1.0, 0.8, 0.6, 0.4), vals=(1.0, 0.9, 0.8, 0.7)):
    from colorsys import rgb_to_hsv, hsv_to_rgb
    r, g, b = [int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5)]
    h, s, v = rgb_to_hsv(r, g, b)
    to_hex = lambda r, g, b: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return tuple(to_hex(*hsv_to_rgb(h, min(1, s*m), min(1, v*n))) for m, n in zip(sats, vals))


# Create 2 B-spline cubic patchs
splines_vol, pts_vol = [], []
for center in ([0.5, 0.5, 0.5], [1.5, 0.5, 0.5]):
    spline, pts = new_cube(center, [0, 0, 1], 1)
    pts_vol.append(spline.orderElevation(pts, [0, 1, 1]))
    splines_vol.append(spline)
# Make the connectivity based on the position of the control points
conn_vol = MultiPatchBSplineConnectivity.from_separated_ctrlPts(pts_vol)
# add some noise to the control points to deform the geometry
unique_pts_vol = conn_vol.pack(conn_vol.agglomerate(pts_vol))
unique_pts_vol += 0.05*np.random.randn(*unique_pts_vol.shape)
pts_vol = conn_vol.separate(conn_vol.unpack(unique_pts_vol))

# Plot both deformed patchs on the same graph
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
ax = fig.add_subplot(projection='3d')
colors = ['#1b9e77', '#d95f02']
handles = []
for i, (spline, pts, color) in enumerate(zip(splines_vol, pts_vol, colors)):
    ctrl_col, int_col, elem_col, bord_col = make_triplet(color)
    spline.plotMPL(pts, ax=ax, ctrl_color=ctrl_col, interior_color=int_col, elem_color=elem_col, border_color=bord_col)
    handles.append(Patch(facecolor=int_col, edgecolor=bord_col, label=f"Patch " + f"$\\Omega_{i+1}^{{vol}}$"))
for i, pts in enumerate(pts_vol):
    other_vecs = np.array([pts[:, 1, 0, 0] - pts[:, 0, 0, 0], pts[:, 0, 1, 0] - pts[:, 0, 0, 0], pts[:, 0, 0, 1] - pts[:, 0, 0, 0]])
    other_vecs /= np.linalg.norm(other_vecs, axis=1)[:, None]
    origin = pts[:, 0, 0, 0] - 0.1*other_vecs.mean(axis=0)
    for vec, label in zip(other_vecs, (r"$\xi$", r"$\eta$", r"$\zeta$")):
        ax.quiver(*origin, *vec, color='k', length=0.2, normalize=True)
        ax.text(*(origin + 0.3*vec), label, color='k', ha='center', va='center', zorder=100)
ax.set(title="Volume multipatch B-splines de deux cubes déformés", xlabel="x", ylabel="y", zlabel="z", aspect='equal')
ax.legend(handles=handles, loc="upper right")
ax.view_init(elev=-20, azim=220)
fig.savefig("two_vol_patchs.pdf")

# %%
# Extract exterior borders'connectivity and B-splines
conn_ext, splines_ext, vol_to_ext = conn_vol.extract_exterior_borders(splines_vol)
# Select corresponding control points
unique_pts_ext = unique_pts_vol[:, vol_to_ext]
pts_ext = conn_ext.separate(conn_ext.unpack(unique_pts_ext))

# Plot every surface patch on the same graph
fig = plt.figure(figsize=(6, 6), constrained_layout=True)
ax = fig.add_subplot(projection='3d')
colors = ['#1b9e77', '#d95f02']
handles = []
for grp in range(2):
    ctrl_col, int_col, elem_col, bord_col = make_triplet(colors[grp])
    for i in range(grp*5, (grp+1)*5):
        spline, pts = splines_ext[i], pts_ext[i]
        spline.plotMPL(pts, ax=ax, ctrl_color=ctrl_col, interior_color=int_col, elem_color=elem_col, border_color=bord_col)
    handles.append(Patch(facecolor=int_col, edgecolor=bord_col, label='surfaces extérieures du patch ' + f"$\\Omega_{grp+1}^{{vol}}$"))
for i, pts in enumerate(pts_ext):
    other_vecs = np.array([pts[:, 1, 0] - pts[:, 0, 0], pts[:, 0, 1] - pts[:, 0, 0]])
    other_vecs /= np.linalg.norm(other_vecs, axis=1)[:, None]
    origin = pts[:, 0, 0] + 0.1*other_vecs.mean(axis=0)
    for vec, label in zip(other_vecs, (r"$\xi$", r"$\eta$")):
        ax.quiver(*origin, *vec, color='k', length=0.2, normalize=True)
        end = origin + 0.3*vec
        ax.text(*end, label, color='k', ha='center', va='center', zorder=100)
ax.set(title="Extraction des surfaces B-spline extérieures du volume multipatch", xlabel="x", ylabel="y", zlabel="z", aspect='equal')
ax.legend(handles=handles)
ax.view_init(elev=-20, azim=220)
fig.savefig("surf_ext.pdf")

# %%