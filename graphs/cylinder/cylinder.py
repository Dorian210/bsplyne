# %%
import numpy as np
import matplotlib.pyplot as plt
from bsplyne import new_cylinder, new_degenerated_cylinder
c, pts = new_cylinder([0, 0, 0], [0, 0, 1], 1, 10)
c.saveParaview(pts, "./", "cylinder")