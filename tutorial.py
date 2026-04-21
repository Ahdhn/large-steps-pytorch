import torch 
from tqdm import trange
import os
from scripts.load_xml import load_scene
from scripts.render import NVDRenderer
import matplotlib.pyplot as plt

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

from scripts.geometry import compute_vertex_normals, compute_face_normals

import numpy as np
import polyscope as ps
import polyscope.imgui as psim


# Load the scene
filepath = os.path.join(os.getcwd(), "scenes", "planck", "planck.xml")
scene_params = load_scene(filepath)

# Load reference shape
v_ref = scene_params["mesh-target"]["vertices"]
n_ref = scene_params["mesh-target"]["normals"]
f_ref = scene_params["mesh-target"]["faces"]

# Load source shape
v = scene_params["mesh-source"]["vertices"]
f = scene_params["mesh-source"]["faces"]

# Initialize the renderer
renderer = NVDRenderer(scene_params, shading=True, boost=3)

# Render the reference images
ref_imgs = renderer.render(v_ref, n_ref, f_ref)
# plt.imshow((ref_imgs[5,...,:-1].clip(0,1).pow(1/2.2)).cpu().numpy(), origin='lower')
# plt.axis('off')
# plt.savefig('ref.png', bbox_inches='tight', dpi=150)
# plt.show()

steps = 1000
step_size = 3e-2
lambda_ = 19

M = compute_matrix(v, f, lambda_)
u = to_differential(M, v)
u.requires_grad = True
opt = AdamUniform([u], step_size)
v_steps = torch.zeros((steps+1, *v.shape), device='cuda')
losses = torch.zeros(steps+1, device='cuda')

for it in trange(steps):
    v = from_differential(M, u, 'Cholesky')
    face_normals = compute_face_normals(v, f)
    n = compute_vertex_normals(v, f, face_normals)
    opt_imgs = renderer.render(v, n, f)
    loss = (opt_imgs - ref_imgs).abs().mean()
    with torch.no_grad():
        losses[it] = loss
        v_steps[it] = v
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    opt_imgs = renderer.render(v, n, f)
    loss = (opt_imgs - ref_imgs).abs().mean()
    losses[-1] = loss
    v_steps[-1] = v
   
plt.figure()
plt.plot(losses.cpu().numpy())
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('loss.png', bbox_inches='tight', dpi=150)

STRIDE = 10
frames_v = v_steps[::STRIDE].detach().cpu().numpy()
frames_f = f.detach().cpu().numpy()
v_ref_np = v_ref.detach().cpu().numpy()
f_ref_np = f_ref.detach().cpu().numpy()
losses_np = losses.detach().cpu().numpy()
num_frames = frames_v.shape[0]

ps.init()
ps.set_ground_plane_mode("shadow_only")
ref_mesh = ps.register_surface_mesh(
    "reference", v_ref_np, f_ref_np,
    color=(0.2, 0.8, 0.3), transparency=0.35, smooth_shade=True,
)
opt_mesh = ps.register_surface_mesh(
    "optimization", frames_v[0], frames_f, smooth_shade=True,
)

state = {"frame": 0, "playing": True, "speed": 1}

def callback():
    _, state["playing"] = psim.Checkbox("play", state["playing"])
    psim.SameLine()
    if psim.Button("reset"):
        state["frame"] = 0
    _, state["frame"] = psim.SliderInt("frame", state["frame"], 0, num_frames - 1)
    _, state["speed"] = psim.SliderInt("speed (frames/tick)", state["speed"], 1, 10)
    if state["playing"]:
        state["frame"] = (state["frame"] + state["speed"]) % num_frames
    opt_mesh.update_vertex_positions(frames_v[state["frame"]])
    iter_idx = min(state["frame"] * STRIDE, len(losses_np) - 1)
    psim.Text(f"iter: {iter_idx}/{len(losses_np) - 1}   loss: {losses_np[iter_idx]:.4e}")

ps.set_user_callback(callback)
ps.show()