import cv2
import os
import csv
import tkinter as tk

# === CONFIG ===
video_path = "sample/hardik.mp4"
output_csv = "segments/hardik.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# === Get screen resolution ===
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# === Labels and key mapping ===
labels = ["jump", "bfc", "ffc", "release", "followthrough"]
label_keys = {ord(str(i + 1)): labels[i] for i in range(len(labels) - 1)}  # 1–4
single_frame_labels = {"bfc", "ffc", "release"}
recorded_once = set()

print("\n========== Labeling Instructions ==========")
for i, label in enumerate(labels[:-1]):
    print(f" Press {i + 1} → label '{label}'")
print(" After 'release', all remaining frames will be labeled as 'followthrough'")
print(" Press SPACE to move to next frame without labeling")
print(" Press 'q' to quit early\n")

# === Video processing ===
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0
frame_labels = []

cv2.namedWindow("Manual Labeler", cv2.WINDOW_NORMAL)

while frame_id < total_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame dynamically to fit within screen dimensions
    h, w = frame.shape[:2]
    scale = min((screen_width - 100) / w, (screen_height - 100) / h)  # leave margin
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Set the window size and show
    cv2.resizeWindow("Manual Labeler", new_w, new_h)
    frame_disp = resized.copy()
    cv2.putText(frame_disp, f"Frame: {frame_id}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Manual Labeler", frame_disp)

    key = cv2.waitKey(0)

    if key == ord('q'):
        print("❌ Quit manually.")
        break

    elif key in label_keys:
        current_label = label_keys[key]

        if current_label in single_frame_labels:
            if current_label not in recorded_once:
                frame_labels.append((frame_id, current_label))
                recorded_once.add(current_label)
                print(f"✅ Recorded: {current_label} at frame {frame_id}")
            else:
                print(f"⚠️ {current_label} already recorded")

            if current_label == "release":
                for fid in range(frame_id + 1, total_frames):
                    frame_labels.append((fid, "followthrough"))
                print(f"🟩 Auto-labeled followthrough from frame {frame_id + 1}")
                break

        else:
            frame_labels.append((frame_id, current_label))
            print(f"✅ Recorded: {current_label} at frame {frame_id}")

        frame_id += 1

    elif key == 32:  # Space bar
        frame_id += 1

    else:
        print("⏭️ Invalid key — use 1–4, SPACE to skip, or 'q' to quit")

cap.release()
cv2.destroyAllWindows()

# === Save CSV ===
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "label"])
    writer.writerows(frame_labels)

print(f"\n📁 Labels saved to: {output_csv}")
