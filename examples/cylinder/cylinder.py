# %%
import os
from bsplyne import new_quarter_pipe
c, pts = new_quarter_pipe([0, 0, 0], [0, 0, 1], 1, 1)
c.saveParaview(pts, os.getcwd(), "cylinder", n_eval_per_elem=100)
# %%
