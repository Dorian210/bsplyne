# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import new_quarter_circle
import tikzplotlib
c, pts = new_quarter_circle([0, 0, 0], [0, 0, 1], 1)
c.bases[0].plotN(show=False)
colors = [line.get_color() for line in plt.gca().get_lines()]
tikzplotlib.save("./quarter_circle_basis.tikz")
plt.show()
def save_1D_to_2D(spline, ctrl_pts, file_name):
    xi = np.linspace(spline.bases[0].span[0], spline.bases[0].span[1], 100)
    vals = spline(ctrl_pts, [xi])
    fig, ax = plt.subplots()
    ax.scatter(*ctrl_pts[:2], c=colors)
    for i in range(c.bases[0].n + 1):
        ax.annotate(f"$\\lambda_{i}$", ctrl_pts[:2, i]*0.9, c=colors[i], fontsize=20, ha='center', va="center")
    ax.plot(vals[0], vals[1], label="$x(\\xi)$")
    ax.set_aspect(1)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.legend(loc='lower left')
    tikzplotlib.save(file_name, extra_axis_parameters=['axis equal'])
    plt.show()
save_1D_to_2D(c, pts, "./quarter_circle_bspline.tikz")
pts[:, 2] *= 0.75
save_1D_to_2D(c, pts, "./quarter_circle_bspline_moved.tikz")
