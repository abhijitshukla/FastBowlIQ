import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.comotion_demo.models import comotion
from src.comotion_demo.utils import dataloading, helper
from src.comotion_demo.utils import track as track_utils

# ====== HARDCODED CONFIG ======
input_path = Path("sample/hardik.mp4")             # path to input video
output_dir = Path("results/")                 # output directory
start_frame = 0
num_frames = 1_000_000_000
frameskip = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_mps = torch.mps.is_available()

output_dir.mkdir(parents=True, exist_ok=True)
cache_path = output_dir / f"{input_path.stem}.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def track_poses(input_path, cache_path):
    model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()

    detections = []
    tracks = []
    initialized = False

    for image, K in tqdm(
        dataloading.yield_image_and_K(input_path, start_frame, num_frames, frameskip),
        desc="Running CoMotion",
    ):
        if not initialized:
            image_res = image.shape[-2:]
            model.init_tracks(image_res)
            initialized = True

        detection, track = model(image, K, use_mps=use_mps)
        detection = {k: v.cpu() for k, v in detection.items()}
        track = track.cpu()
        detections.append(detection)
        tracks.append(track)

    detections = {k: [d[k] for d in detections] for k in detections[0].keys()}
    tracks = torch.stack(tracks, 1)
    tracks = {k: getattr(tracks, k) for k in ["id", "pose", "trans", "betas"]}

    track_ref = track_utils.cleanup_tracks(
        {"detections": detections, "tracks": tracks},
        K,
        model.smpl_decoder.cpu(),
        min_matched_frames=1,
    )

    if track_ref:
        frame_idxs, track_idxs = track_utils.convert_to_idxs(
            track_ref, tracks["id"][0].squeeze(-1).long()
        )
        preds = {k: v[0, frame_idxs, track_idxs] for k, v in tracks.items()}
        preds["id"] = preds["id"].squeeze(-1).long()
        preds["frame_idx"] = frame_idxs
        torch.save(preds, cache_path)
        print(f"✅ Saved .pt file to: {cache_path}")


if __name__ == "__main__":
    track_poses(input_path, cache_path)
