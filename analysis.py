import os
import numpy as np
import pandas as pd

# === Utility Functions ===
def get_joint(row, joint_id):
    return np.array([
        row[f"joint{joint_id}_x"],
        row[f"joint{joint_id}_y"],
        row[f"joint{joint_id}_z"]
    ], dtype=float)

def angle_between(j1, j2, j3):
    a = j1 - j2
    b = j3 - j2
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def hip_shoulder_separation(row):
    left_shoulder = get_joint(row, 13)
    right_shoulder = get_joint(row, 14)
    left_hip = get_joint(row, 1)
    right_hip = get_joint(row, 2)
    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip
    unit_shoulder = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)
    unit_hip = hip_vec / (np.linalg.norm(hip_vec) + 1e-8)
    dot = np.dot(unit_shoulder, unit_hip)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return np.degrees(angle)

def stride_length(row):
    right_ankle = get_joint(row, 8)
    left_ankle = get_joint(row, 7)
    return np.linalg.norm(right_ankle - left_ankle)

def estimate_model_height(row):
    head = get_joint(row, 15)
    ankle = get_joint(row, 7)
    return np.linalg.norm(head - ankle) + 0.1 

def scale_distance(model_distance, model_height, actual_height):
    scale_factor = actual_height / (model_height + 1e-8)
    meters = model_distance * scale_factor
    inches = meters * 39.3701
    return meters, inches

def lateral_flexion(row):
    spine = get_joint(row, 9)
    pelvis = get_joint(row, 0)
    vec = spine - pelvis
    angle = np.arctan2(vec[0], vec[1])
    return np.degrees(angle)

def classify_alignment(row):
    right_hip = get_joint(row, 2)
    left_hip = get_joint(row, 1)
    right_shoulder = get_joint(row, 17)
    left_shoulder = get_joint(row, 16)
    delivery_vec = np.array([0, 0, 1])  # Z-axis is bowling direction
    hip_vec = left_hip - right_hip
    shoulder_vec = left_shoulder - right_shoulder

    hip_angle = np.degrees(np.arccos(np.clip(np.dot(hip_vec, delivery_vec) / (np.linalg.norm(hip_vec) + 1e-8), -1.0, 1.0)))
    shoulder_angle = np.degrees(np.arccos(np.clip(np.dot(shoulder_vec, delivery_vec) / (np.linalg.norm(shoulder_vec) + 1e-8), -1.0, 1.0)))

    def classify(angle):
        if 70 <= angle <= 110:
            return "side-on"
        elif angle <= 30 or angle >= 150:
            return "front-on"
        else:
            return "semi side-on"

    hip_label = classify(hip_angle)
    shoulder_label = classify(shoulder_angle)
    return hip_label if hip_label == shoulder_label else f"mixed ({hip_label}/{shoulder_label})"


# === Main Analysis ===
def analyze():
    print("\n===== BIOMECHANICAL ANALYSIS (CSV Based) =====\n")

    bfc_df = pd.read_csv("phases/bfc.csv")
    ffc_df = pd.read_csv("phases/ffc.csv")
    release_df = pd.read_csv("phases/release.csv")

    actual_height = 1.83  # meters

    # --- BFC
    bfc_frame = int(bfc_df['frame'].iloc[0]) if 'frame' in bfc_df else 0
    bfc_angles = []
    for _, row in bfc_df.iterrows():
        hip = get_joint(row, 1)
        knee = get_joint(row, 4)
        ankle = get_joint(row, 7)
        bfc_angles.append(angle_between(hip, knee, ankle))
    bfc_knee = np.mean(bfc_angles)
    print(f"📍 BFC — Back Knee Angle Avg: {bfc_knee:.2f}°")

    # --- FFC
    ffc_row = ffc_df.iloc[0]
    ffc_frame = int(ffc_row['frame']) if 'frame' in ffc_df.columns else 0
    stride_raw = stride_length(ffc_row)
    model_height = estimate_model_height(ffc_row)
    stride_m, stride_in = scale_distance(stride_raw, model_height, actual_height)
    alignment = classify_alignment(ffc_row)
    print(f"📍 FFC — Stride Length: {stride_m:.2f} m / {stride_in:.2f} in")
    print(f"📍 FFC — Alignment: {alignment}")

    # --- Elbow + HS Separation
    combined_rows = pd.concat([ffc_df, release_df])
    chuck_angles = []
    hs_angles = []
    for _, row in combined_rows.iterrows():
        shoulder = get_joint(row, 17)
        elbow = get_joint(row, 19)
        wrist = get_joint(row, 21)
        chuck_angles.append(abs(angle_between(shoulder, elbow, wrist) - 180))
        hs_angles.append(hip_shoulder_separation(row))
    elbow_max = np.max(chuck_angles)
    hs_avg = np.mean(hs_angles)
    print(f"📍 Max Elbow Angle: {elbow_max:.2f}° (Chucking check)")
    print(f"📍 Hip-Shoulder Separation Avg: {hs_avg:.2f}°")

    # --- Release
    release_row = release_df.iloc[0]
    release_frame = int(release_row['frame']) if 'frame' in release_df.columns else 0
    hip = get_joint(release_row, 1)
    knee = get_joint(release_row, 4)
    ankle = get_joint(release_row, 7)
    front_knee_angle = angle_between(hip, knee, ankle)

    palm = get_joint(release_row, 23)
    toe = get_joint(release_row, 10)
    raw_z_distance = abs(palm[2] - toe[2])
    reach_m, reach_in = scale_distance(raw_z_distance, model_height, actual_height)

    flexion = lateral_flexion(release_row)
    print(f"📍 Release — Front Knee Angle: {front_knee_angle:.2f}°")
    print(f"📍 Release — Delivery Reach: {reach_m:.2f} m / {reach_in:.2f} in")
    print(f"📍 Release — Lateral Flexion: {flexion:.2f}°")

    # === Save to CSV with frame numbers ===
    results = {
        "Frame_BFC": [bfc_frame],
        "Back_Knee_Angle_BFC": [bfc_knee],
        "Frame_FFC": [ffc_frame],
        "Stride_Length_m": [stride_m],
        "Stride_Length_in": [stride_in],
        "Alignment": [alignment],
        "Max_Elbow_Angle": [elbow_max],
        "Hip_Shoulder_Separation": [hs_avg],
        "Frame_Release": [release_frame],
        "Front_Knee_Angle_Release": [front_knee_angle],
        "Delivery_Reach_m": [reach_m],
        "Delivery_Reach_in": [reach_in],
        "Lateral_Flexion": [flexion]
    }

    df_out = pd.DataFrame(results)
    df_out.to_csv("final_output/biomech_results.csv", index=False)
    print("\n✅ Saved biomechanical results with frame numbers to 'biomech_results.csv'")


if __name__ == "__main__":
    analyze()
