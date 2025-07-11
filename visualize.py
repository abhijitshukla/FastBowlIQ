import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm

from src.comotion_demo.models import comotion
from src.comotion_demo.utils import dataloading, helper

try:
    from aitviewer.configuration import CONFIG
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
    from aitviewer.scene.camera import OpenCVCamera

    comotion_model_dir = Path(comotion.__file__).parent
    CONFIG.smplx_models = os.path.join(comotion_model_dir, "../data")
    CONFIG.window_type = "pyqt6"
    aitviewer_available = True
except ModuleNotFoundError:
    print("WARNING: Skipped aitviewer import, ensure it is installed to run visualization.")
    aitviewer_available = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def prepare_scene_black_background(viewer, width, height, K, fps=30):
    viewer.reset()
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.scene.bg_color = (0.0, 0.0, 0.0, 1.0)

    extrinsics = np.eye(4)[:3]
    cam = OpenCVCamera(K, extrinsics, cols=width, rows=height, viewer=viewer)
    viewer.scene.add(cam)

    # Farther back and angled downward so legs/toes fit
    viewer.scene.camera.position = [0, 0, -25]     # <- Zoomed out
    viewer.scene.camera.target = [0, -1.5, 0]      # <- Focus lower

    viewer.auto_set_camera_target = False
    viewer.set_temp_camera(cam)
    viewer.playback_fps = fps



def add_pose_to_scene(viewer, smpl_layer, betas, pose, trans, color=(0.6, 0.6, 0.6), alpha=1, color_ref=None):
    if betas.ndim == 2:
        betas = betas[None]
        pose = pose[None]
        trans = trans[None]

    poses_root = pose[..., :3]
    poses_body = pose[..., 3:]
    max_people = pose.shape[1]

    if (betas != 0).any():
        for person_idx in range(max_people):
            person_color = (
                color if color_ref is None
                else (color_ref[person_idx % len(color_ref)] * 0.4 + 0.3)
            )
            person_color = [float(c) for c in person_color] + [alpha]

            valid_vals = (betas[:, person_idx] != 0).any(-1)
            if valid_vals.any():
                trans[~valid_vals, person_idx, 2] = -10000
                viewer.scene.add(
                    SMPLSequence(
                        smpl_layer=smpl_layer,
                        betas=betas[:, person_idx],
                        poses_root=poses_root[:, person_idx],
                        poses_body=poses_body[:, person_idx],
                        trans=trans[:, person_idx],
                        color=person_color,
                    )
                )


@click.command()
@click.option("--width", default=1920, type=int, help="Width of output video.")
@click.option("--height", default=1080, type=int, help="Height of output video.")
@click.option("--fps", default=30, type=int, help="Frames per second.")
@click.option("--color-r", default=0.6, type=float)
@click.option("--color-g", default=0.6, type=float)
@click.option("--color-b", default=0.6, type=float)
@click.option("--alpha", default=1.0, type=float)
def visualize_pt_on_black_hardcoded(width, height, fps, color_r, color_g, color_b, alpha):
    input_pt_path = Path("results/hardik.pt")
    output_video_dir = Path("rendered")
    output_video_path = output_video_dir / f"{input_pt_path.stem}"

    if not aitviewer_available:
        logging.error("AITViewer is not available. Please install it to run visualization.")
        return

    logging.info(f"Visualizing SMPL poses from {input_pt_path} to {output_video_path}")

    K = dataloading.get_default_K(torch.zeros(1, 3, height, width)).cpu().numpy()
    viewer = HeadlessRenderer(size=(width, height))

    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
    if not input_pt_path.exists():
        logging.error(f"Error: .pt file not found at {input_pt_path}")
        return

    try:
        preds = torch.load(input_pt_path, map_location="cpu")
    except Exception as e:
        logging.error(f"Error loading .pt file: {e}")
        return

    if "frame_idx" not in preds:
        logging.error("'.pt' file must contain 'frame_idx' to visualize sequence.")
        return

    max_frame_idx = preds["frame_idx"].max().item()
    num_total_frames = max_frame_idx + 1

    if "id" in preds and len(preds["id"]) > 0:
        unique_person_ids = torch.unique(preds["id"])
        max_people_to_render = len(unique_person_ids)
        person_id_to_render_idx = {id.item(): i for i, id in enumerate(unique_person_ids)}
    else:
        max_people_to_render = 1
        person_id_to_render_idx = {0: 0}
        logging.warning("No 'id' field found in .pt file or no detections. Assuming a single person.")

    all_betas = torch.zeros(num_total_frames, max_people_to_render, preds['betas'].shape[-1], dtype=preds['betas'].dtype)
    all_pose = torch.zeros(num_total_frames, max_people_to_render, preds['pose'].shape[-1], dtype=preds['pose'].dtype)
    all_trans = torch.zeros(num_total_frames, max_people_to_render, preds['trans'].shape[-1], dtype=preds['trans'].dtype)

    for i in tqdm(range(len(preds["frame_idx"])), desc="Preparing poses for visualization"):
        frame_idx = preds["frame_idx"][i].item()
        person_id = preds["id"][i].item() if "id" in preds else 0
        if person_id in person_id_to_render_idx:
            render_idx = person_id_to_render_idx[person_id]
            all_betas[frame_idx, render_idx] = preds["betas"][i]
            all_pose[frame_idx, render_idx] = preds["pose"][i]
            all_trans[frame_idx, render_idx] = preds["trans"][i]

    prepare_scene_black_background(viewer, width, height, K, fps)

    rendering_colors = [np.array([color_r, color_g, color_b])] * max_people_to_render
    if 'id' in preds and max_people_to_render > 0:
        rendering_colors = [helper.color_ref[idx % len(helper.color_ref)] for idx in range(max_people_to_render)]
        rendering_colors = np.array(rendering_colors)

    add_pose_to_scene(
        viewer,
        smpl_layer,
        all_betas,
        all_pose,
        all_trans,
        color=(color_r, color_g, color_b),
        alpha=alpha,
        color_ref=rendering_colors,
    )

    output_video_dir.mkdir(parents=True, exist_ok=True)
    viewer.save_video(
        video_dir=str(output_video_path),
        output_fps=fps,
        ensure_no_overwrite=False,
    )
    logging.info(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    visualize_pt_on_black_hardcoded()
