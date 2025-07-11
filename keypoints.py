import torch
import pandas as pd
import numpy as np
import os
from smplx import SMPL

# === Input files ===
pt_path = "results/hardik.pt"          # CoMotion output
csv_path = "segments/hardik.csv"       # Phase segments
output_dir = "phases"              # Output CSV folder
os.makedirs(output_dir, exist_ok=True)

# === SMPL Model for extracting joints ===
smpl_model_path = "src/comotion_demo/data/smpl"  # Folder containing SMPL_NEUTRAL.pkl
smpl = SMPL(model_path=smpl_model_path, gender='neutral', batch_size=1)

# === Load pose data ===
data = torch.load(pt_path, map_location="cpu")
frame_indices = data["frame_idx"].numpy()

# === Load frame-wise labels ===
label_df = pd.read_csv(csv_path)
label_map = dict(zip(label_df["frame"], label_df["label"]))

# === Initialize containers per phase
phase_containers = {
    "jump": [],
    "bfc": [],
    "ffc": [],
    "release": [],
    "followthrough": []
}

# === Group data per frame (in case multiple persons)
frame_to_indices = {}
for i, frame_id in enumerate(frame_indices):
    frame_to_indices.setdefault(frame_id, []).append(i)

# === Process each frame
for frame_id, indices in frame_to_indices.items():
    label = label_map.get(frame_id)
    if label not in phase_containers:
        continue

    max_area = -1
    best_row = None

    for idx in indices:
        pose = torch.tensor(data["pose"][idx]).unsqueeze(0).float()
        betas = torch.tensor(data["betas"][idx]).unsqueeze(0).float()
        trans = torch.tensor(data["trans"][idx]).unsqueeze(0).float()

        smpl_output = smpl(
            betas=betas,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3],
            transl=trans
        )
        joints = smpl_output.joints[0, :24].detach().numpy()  # [24, 3]

        # Compute bounding box area in x-y plane
        x_min, y_min = joints[:, 0].min(), joints[:, 1].min()
        x_max, y_max = joints[:, 0].max(), joints[:, 1].max()
        area = (x_max - x_min) * (y_max - y_min)

        if area > max_area:
            max_area = area
            row = {"frame": int(frame_id)}
            for j in range(24):
                row[f"joint{j}_x"] = joints[j, 0]
                row[f"joint{j}_y"] = joints[j, 1]
                row[f"joint{j}_z"] = joints[j, 2]
            best_row = row

    if best_row:
        phase_containers[label].append(best_row)

# === Save each phase to CSV
for phase, rows in phase_containers.items():
    df = pd.DataFrame(rows)
    save_path = os.path.join(output_dir, f"{phase}.csv")
    df.to_csv(save_path, index=False)
    print(f"[SAVED] {phase}: {len(rows)} frames → {save_path}")
