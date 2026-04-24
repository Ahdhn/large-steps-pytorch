import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.main import optimize_shape
from scripts.constants import *
from scripts.io_ply import write_ply
from largesteps.parameterize import enable_matrix_dump, flush_matrix_dump
from largesteps.optimize import AdamUniform

from igl import write_triangle_mesh

output_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.dirname(__file__)))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

scene = "nefertiti"
dump_dir = os.path.join(output_dir, "solver_dumps", scene)
enable_matrix_dump(dump_dir)

folder = SCENES_DIR
filename = os.path.join(folder, scene, f"{scene}.xml")
#remesh_steps = [500, 1500, 3000, 4500, 7000, 10000, 12000, 14000]
remesh_steps = [250, 500, 750, 1000]

params = {
    "steps": 1000,
    "step_size" : 2e-3,
    "loss": "l1",
    "optimizer": AdamUniform,
    "boost" : 3,
    #"lambda": 19,
    "alpha": 0.98,
    "remesh": remesh_steps.copy(),
    }

out = optimize_shape(filename, params)
flush_matrix_dump()

# Write one OBJ per remesh stage, using the exact (v_unique, f_unique) arrays
# fed to `compute_matrix`. These share row/column ordering with the dumped
# matrices mat_0XX.mtx (one matrix per stage, same k).
n_stages = len(out["v_unique"])
pd.DataFrame(data=list(range(n_stages))).to_csv(
    os.path.join(output_dir, "solver_dumps", scene, "remesh_steps.csv")
)
for k in range(n_stages):
    write_triangle_mesh(
        os.path.join(output_dir, "solver_dumps", scene, f"res_{k:02d}.obj"),
        out["v_unique"][k].astype(np.float64),
        out["f_unique"][k].astype(np.int32),
    )
