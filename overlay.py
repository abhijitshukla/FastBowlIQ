import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# ===== CONFIG =====
input_video_path = "rendered/hardik.mp4"
output_video_path = "final_output/hardik_overlayed.mp4"
csv_path = "final_output/biomech_results.csv"

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = (0, 0, 255)  # Red text
line_thickness = 2
line_spacing = 30  # Vertical spacing between text lines

# Read CSV
df = pd.read_csv(csv_path)
row = df.iloc[0]

# Extract biomechanical annotations
biomech_analysis = {
    "bfc": {
        "frame": int(row["Frame_BFC"]),
        "lines": [
            f"Back Knee Flexion: {row['Back_Knee_Angle_BFC']:.1f} deg"
        ]
    },
    "ffc": {
        "frame": int(row["Frame_FFC"]),
        "lines": [
            f"Stride Length: {row['Stride_Length_m']:.2f} m ({row['Stride_Length_in']:.1f} in)",
            f"Alignment: {row['Alignment']}"
        ]
    },
    "release": {
        "frame": int(row["Frame_Release"]),
        "lines": [
            f"Elbow Angle: {row['Max_Elbow_Angle']:.1f} deg",
            f"Hip-Shoulder Separation: {row['Hip_Shoulder_Separation']:.1f} deg",
            f"Front Knee Angle: {row['Front_Knee_Angle_Release']:.1f} deg",
            f"Delivery Reach: {row['Delivery_Reach_m']:.2f} m",
            f"Lateral Flexion: {row['Lateral_Flexion']:.1f} deg"
        ]
    }
}

# Map frame index to lines
overlay_frames = {v["frame"]: v["lines"] for v in biomech_analysis.values()}

# Open video
cap = cv2.VideoCapture(str(input_video_path))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {input_video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"ℹ️ FPS: {fps}, Resolution: {frame_width}x{frame_height}")

# Prepare writer
out = cv2.VideoWriter(
    str(output_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

frame_idx = 0
all_lines_so_far = []

def safe_text_position(x, y, w, h):
    x = max(0, min(x, frame_width - w - 10))
    y = max(h + 10, min(y, frame_height - 10))
    return x, y

# Process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in overlay_frames:
        print(f"📌 Annotating phase at frame {frame_idx}")
        new_lines = overlay_frames[frame_idx]
        all_lines_so_far.extend(new_lines)

        frame_copy = frame.copy()
        for j, text in enumerate(all_lines_so_far):
            text_size = cv2.getTextSize(text, font, font_scale, line_thickness)[0]
            x, y = 50, 50 + j * line_spacing
            x, y = safe_text_position(x, y, *text_size)

            cv2.putText(
                frame_copy,
                text,
                (x, y),
                font,
                font_scale,
                font_color,
                line_thickness,
                cv2.LINE_AA
            )

        for _ in range(int(fps)):
            out.write(frame_copy)

    else:
        # Draw ongoing text overlays (if any)
        if all_lines_so_far:
            frame_copy = frame.copy()
            for j, text in enumerate(all_lines_so_far):
                text_size = cv2.getTextSize(text, font, font_scale, line_thickness)[0]
                x, y = 50, 50 + j * line_spacing
                x, y = safe_text_position(x, y, *text_size)

                cv2.putText(
                    frame_copy,
                    text,
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                    cv2.LINE_AA
                )
            out.write(frame_copy)
        else:
            out.write(frame)

    frame_idx += 1

cap.release()
out.release()
print(f"✅ Done! Overlay video saved to: {output_video_path}")
