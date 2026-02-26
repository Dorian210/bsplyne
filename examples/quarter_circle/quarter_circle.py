# %%
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
line_styles = [
    "-",  # style simple plein
    (0, (6, 2)),  # motif 2 éléments
    (0, (3, 1)),  # motif 2 éléments court
    (0, (1, 1)),  # motif 2 éléments très court
    (0, (6, 2, 1, 2)),  # motif 4 éléments
    (0, (3, 1, 1, 1)),  # motif 4 éléments court
    (0, (6, 2, 3, 2)),  # motif 4 éléments mixte
    (0, (6, 2, 1, 2, 1, 2)),  # motif 6 éléments long
    (0, (3, 1, 1, 1, 1, 1)),  # motif 6 éléments court
    (0, (6, 2, 3, 2, 1, 2)),  # motif 6 éléments mixte
]
combined_cycler = cycler(color=default_colors) + cycler(linestyle=line_styles)
plt.rc("axes", prop_cycle=combined_cycler)
plt.rcParams.update({"font.size": 10})
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from bsplyne import new_quarter_circle


# %%
c, pts = new_quarter_circle([0, 0, 0], [0, 0, 1], 1)
pts = pts[:2]
c.bases[0].plotN(show=False)
fig = plt.gcf()
fig.set_size_inches(4, 2.25)
ax = plt.gca()
el_borders = np.unique(c.bases[0].knot)
ax.set_xticks(el_borders)
ax.set_xticklabels(["0"] + [""] * (len(el_borders) - 2) + ["1"])
ax.set_xticks(0.5 * (el_borders[1:] + el_borders[:-1]), minor=True)
ax.set_xticklabels(
    [r"\$el_" + str(i) + r"\$" for i in range(len(el_borders) - 1)], minor=True
)
ax.tick_params(which="minor", length=0)
ax.set_xlabel(r"\$\xi\$")
ax.set_yticks([0, 1])
plt.legend(
    [r"\$N_" + str(i) + r"(\xi)\$" for i in range(c.bases[0].n + 1)],
    bbox_to_anchor=(1.0, 1.0),
)
fig.tight_layout()
plt.rcParams["svg.fonttype"] = "none"
fig.savefig("./quarter_circle_basis.svg")
plt.show()


# %%
XI = c.linspace()
xi = XI[0]
fig, ax = plt.subplots()
fig.set_size_inches(4, 2.25)
colors = list(mcolors.TABLEAU_COLORS.values())
colors = [colors[i % 10] for i in range(c.getNbFunc())]
ax.scatter(*pts, c=colors)
rgb = np.array(
    [[int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in colors], dtype="int"
).T
seg_rgb = c(rgb, XI)
seg_rgb = ((seg_rgb[:, 1:] + seg_rgb[:, :-1]) // 2).astype("int")
seg_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in seg_rgb.T]
vals = c(pts, XI)
seg = np.concatenate((vals[:, None, :-1].T, vals[:, None, 1:].T), axis=1)
spline_line1 = ax.add_collection(LineCollection(seg, colors=seg_colors, linewidth=5))
ind_ann = 16
ax.annotate(
    "Original\nB-spline line",  # The text to display
    xy=1.01 * vals[:, ind_ann],  # The point to annotate (x, y)
    xytext=(1.3, 0.9),  # The position for the text (x, y)
    arrowprops=dict(
        facecolor="k",
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10",
    ),  # Arrow properties
    fontsize=10,  # Font size of the text
    color="k",  # Color of the text
    bbox=dict(
        boxstyle="round,pad=0.3", edgecolor="k", facecolor=seg_colors[ind_ann]
    ),  # Text box properties
    ha="center",  # Horizontal alignment of the text
    va="center",  # Vertical alignment of the text
)
ax.plot(
    np.cos(np.pi / 2 * xi),
    np.sin(np.pi / 2 * xi),
    c="k",
    linewidth=1,
    linestyle="--",
    label=r"Exact quarter circle",
)
a = 0.35 * np.pi
vals = c(pts, [el_borders])
eps = 2e-3
el_borders[0] += eps
el_borders[-1] -= eps
seg = np.concatenate(
    (
        c(pts, [el_borders - eps])[:, None, :].T,
        c(pts, [el_borders + eps])[:, None, :].T,
    ),
    axis=1,
)
ax.add_collection(LineCollection(seg, color="k", linewidth=20))
locs = 0.5 * (vals[:, 1:] + vals[:, :-1])
for i in range(locs.shape[1]):
    ax.annotate(
        r"\$el_" + str(i) + r"\$",
        locs[:, i] * 1.15,
        fontsize=10,
        ha="center",
        va="center",
    )
i = 2
init = pts[:, i].copy()
pts[:, i] *= 0.75
ax.scatter(*pts, c=colors, alpha=0.25)
ax.arrow(
    *0.95 * init, *(1.1 * pts[:, i] - 0.95 * init), color="k", width=0.01, zorder=100
)
for i in range(c.bases[0].n + 1):
    ax.annotate(
        r"\$P_" + str(i) + r"\$",
        pts[:, i] * 0.85,
        c=colors[i],
        fontsize=10,
        ha="center",
        va="center",
    )
vals = c(pts, XI)
seg = np.concatenate((vals[:, None, :-1].T, vals[:, None, 1:].T), axis=1)
ax.add_collection(LineCollection(seg, colors=seg_colors, linewidth=5, alpha=0.25))
ind_ann = 8
ax.annotate(
    "Deformed\nB-spline line",  # The text to display
    xy=0.99 * vals[:, ind_ann],  # The point to annotate (x, y)
    xytext=(0.2, 0.0),  # The position for the text (x, y)
    arrowprops=dict(
        facecolor="k", arrowstyle="->", connectionstyle="angle,rad=10"
    ),  # Arrow properties
    fontsize=10,  # Font size of the text
    color="k",  # Color of the text
    bbox=dict(
        boxstyle="round,pad=0.3", edgecolor="k", facecolor=seg_colors[ind_ann]
    ),  # Text box properties
    ha="center",  # Horizontal alignment of the text
    va="center",  # Vertical alignment of the text
)
ax.set_xlim(right=locs[0].max() * 1.15)
ax.set_ylim(top=locs[1].max() * 1.15)
ax.set_aspect(1)
ax.axis("off")
plt.legend(bbox_to_anchor=(0.5, 0.65))
fig.tight_layout()
plt.rcParams["svg.fonttype"] = "none"
fig.savefig("./quarter_circle_bspline.svg")
plt.show()


# %%
