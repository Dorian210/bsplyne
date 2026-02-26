# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

from bsplyne import new_quarter_pipe


# %%
spl, ctrl = new_quarter_pipe([0, 0, 0], [0, 0, 1], 1, 1)
ctrl = spl.orderElevation(ctrl, [0, 1])
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(projection='3d')
spl.plotMPL(ctrl, ax=ax, elem_color='#d95f02', border_color='#d95f02')
r = 1.2
A = np.array([r*np.cos(np.pi*5/16), r*np.sin(np.pi*5/16), 0])
B = np.array([r*np.cos(np.pi*3/16), r*np.sin(np.pi*3/16), 0])
x, y, z = A
dx, dy, dz = B - A
ax.add_artist(Arrow3D(x, y, z, dx, dy, dz, mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k", connectionstyle="arc3,rad=-0.12"))
ax.text(*(A + B)*0.5*1.1, r"\$\xi\$", ha="center", va="top")
r = 1.1
A = np.array([0, r, 0])
B = np.array([0, r, 0.5])
x, y, z = A
dx, dy, dz = B - A
ax.add_artist(Arrow3D(x, y, z, dx, dy, dz, mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k", connectionstyle="arc3,rad=-0."))
ax.text(*(A + B)*0.5*1.1, r"\$\eta\$", ha="left", va="center")
ax.set_xlabel(r'\$x\$', labelpad=-10)
ax.xaxis.set_rotate_label(False)
ax.set_ylabel(r'\$y\$', labelpad=-10)
ax.yaxis.set_rotate_label(False)
ax.set_zlabel(r'\$z\$', labelpad=-10)
ax.zaxis.set_rotate_label(False)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    for tick in axis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
legend = ax.get_legend()
ax.legend(legend.legend_handles[:-1], [text.get_text() for text in legend.get_texts()[:-1]])
fig.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("./quarter_pipe_bspline.svg")


# %%
xi, eta = spl.linspace(n_eval_per_elem=100)
Xi, Eta = np.meshgrid(xi, eta, indexing='ij')
N = spl.DN([xi, eta]).T.A.reshape((*ctrl.shape[1:], xi.size, eta.size))
Nxi = spl.bases[0].N(xi).T.A
xi_min, xi_max = spl.bases[0].span
eta_min, eta_max = spl.bases[1].span
Neta = spl.bases[1].N(eta).T.A
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(projection='3d')
fonction_dinteret = (3, 0)
far = 0.3
for i in range(ctrl.shape[1]):
    ax.plot(xi, eta_min*np.ones_like(xi) - far*(eta_max - eta_min), Nxi[i], alpha=0.2 + 0.8*int(i==fonction_dinteret[0]), color='#a6761d')
for i in range(ctrl.shape[2]):
    ax.plot(xi_min*np.ones_like(eta) - far*(xi_max - xi_min), eta, Neta[i], alpha=0.2 + 0.8*int(i==fonction_dinteret[1]), color='#a6761d')
ax.plot_surface(Xi, Eta, N[fonction_dinteret[0], fonction_dinteret[1]], rstride=100, cstride=100, edgecolor='#d95f02', facecolor='#7570b3', alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([0, 0.25, 0.5, 0.75, 1])
ax.set_zticklabels(['0', '', '', '', '1'])
ax.set_xlabel(r"\$\xi\$", labelpad=-10)
ax.xaxis.set_rotate_label(False)
ax.set_ylabel(r"\$\eta\$", labelpad=-10)
ax.yaxis.set_rotate_label(False)
ax.view_init(elev=30, azim=45)
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("./quarter_pipe_basis.svg")


# %%
