# %% Imports
from bsplyne import MultiPatchBSplineConnectivity, new_cube
import numpy as np

# %% Create a base B-spline cube : one patch
# Initialize a cubic spline and its control points
spline, ctrl_pts = new_cube([0, 0, 0], [0, 0, 1], 1)

# Elevate the order of the spline for more flexibility (originally of degree (1, 1, 1))
ctrl_pts = spline.orderElevation(ctrl_pts, [1, 1, 1])

# Insert knots to refine the mesh (originally 1 element per parametric direction)
ctrl_pts = spline.knotInsertion(ctrl_pts, [3, 3, 3])

# %% Create a set of 8 identical splines : the multipatch connected mesh
splines = [spline] * 8

# Shift control points to create 8 offset cubes for a 2x2x2 multipatch cube
# Each cube is translated according to the 8 possible combinations of ±0.5 in each direction
separated_ctrl_pts = [
    ctrl_pts + np.array([+0.5, +0.5, +0.5])[:, None, None, None],  # Cube 1: +x, +y, +z
    ctrl_pts + np.array([+0.5, +0.5, -0.5])[:, None, None, None],  # Cube 2: +x, +y, -z
    ctrl_pts + np.array([+0.5, -0.5, +0.5])[:, None, None, None],  # Cube 3: +x, -y, +z
    ctrl_pts + np.array([+0.5, -0.5, -0.5])[:, None, None, None],  # Cube 4: +x, -y, -z
    ctrl_pts + np.array([-0.5, +0.5, +0.5])[:, None, None, None],  # Cube 5: -x, +y, +z
    ctrl_pts + np.array([-0.5, +0.5, -0.5])[:, None, None, None],  # Cube 6: -x, +y, -z
    ctrl_pts + np.array([-0.5, -0.5, +0.5])[:, None, None, None],  # Cube 7: -x, -y, +z
    ctrl_pts + np.array([-0.5, -0.5, -0.5])[:, None, None, None],  # Cube 8: -x, -y, -z
]

# Construct the connectivity between patches from the control points positions
connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrl_pts)

print(f"Multipatch connectivity created for a {connectivity.nb_unique_nodes} nodes mesh.")

# %% Save the original cube in ParaView format
connectivity.save_paraview(splines, separated_ctrl_pts, 
                           "out_paraview", "original_cube", 
                           n_eval_per_elem=5)

# %% Apply a random deformation field
# Generate a random deformation field (3 components, one per unique node)
u_field = 1e-2 * np.random.randn(3, connectivity.nb_unique_nodes)

# Agglomerate control points to apply the deformation field
# The unique format allows for C0 fields and, here, shape modifications
unique_ctrl_pts = connectivity.pack(connectivity.agglomerate(separated_ctrl_pts))

# Apply the deformation field
new_unique_ctrl_pts = unique_ctrl_pts + u_field

# Separate control points to return to the initial structure
new_separated_ctrl_pts = connectivity.separate(connectivity.unpack(new_unique_ctrl_pts))

# Save the deformed cube in ParaView format
connectivity.save_paraview(splines, new_separated_ctrl_pts, 
                           "out_paraview", "deformed_cube", 
                           n_eval_per_elem=5)

# %% Save the cube with the deformation field as a Paraview field
connectivity.save_paraview(
    splines, separated_ctrl_pts,
    "out_paraview", "cube_w_field",
    unique_fields={"u": u_field[None]},  # Add the 'u' field to the Paraview mesh
    n_eval_per_elem=5
)

# %%
