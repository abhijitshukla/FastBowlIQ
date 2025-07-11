# 🏏 FastBowlIQ: AI-Powered Cricket Biomechanics Analysis

**FastBowlIQ** is a complete pipeline for analyzing the biomechanics of fast bowlers using 3D SMPL pose estimation, phase labeling, biomechanical metric extraction, and automatic feedback generation using AI.

It uses Apple’s CoMotion for pose tracking and Google Gemini to generate detailed performance reports.

---

## 🍏 Apple CoMotion Setup

This project relies on [Apple CoMotion](https://github.com/apple/ml-comotion) to extract 3D pose information from a bowling video using the SMPL model.

### ⚙️ Setup Instructions:
1. Clone the [Apple CoMotion](https://github.com/apple/ml-comotion) repository.
2. Copy or symlink the required folders (e.g., `src/comotion_demo/`) into your working directory as `comotion/` or `src/`.
3. Download the required SMPL model (`SMPL_NEUTRAL.pkl`) and place it in:
   ```
   src/comotion_demo/data/
   ```

> 🙏 **Huge thanks to Apple for releasing CoMotion as open source.** This project wouldn't be possible without their contribution.

---

## 🗂️ Folder Structure

```
.
├── main.py                # Run CoMotion tracking on input video
├── segment.py             # Manually label bowling phases (jump, BFC, FFC, etc.)
├── keypoints.py           # Extract SMPL keypoints for labeled frames
├── analysis.py            # Compute biomechanical metrics from joints
├── overlay.py             # Add biomechanical feedback text onto video
├── feedback.py            # Generate feedback report using Gemini API
├── requirements.txt
├── README.md
├── sample/                # Input videos
├── results/               # Output .pt file from CoMotion
├── segments/              # Manually labeled frames per phase
├── phases/                # CSVs of 3D joints per phase
├── final_output/          # Final results: metrics CSV, annotated video, feedback.md
├── rendered/              # 3D SMPL visualization videos
├── src/, comotion/        # SMPL and CoMotion model code (from Apple)
├── tempCodeRunnerFile.py  # (Ignore - VSCode temp file)
```

---

## 🚀 Execution Pipeline

| Step | Script         | Description                                                       | Output                                  |
|------|----------------|-------------------------------------------------------------------|------------------------------------------|
| 1    | `main.py`      | Run Apple CoMotion to extract 3D SMPL pose from input video       | `results/hardik.pt`                      |
| 2    | `segment.py`   | Label bowling phases frame-by-frame (manual GUI)                 | `segments/hardik.csv`                    |
| 3    | `keypoints.py` | Extract keypoints from `.pt` file for each phase                 | CSV files in `phases/`                   |
| 4    | `analysis.py`  | Compute angles/distances and biomechanical metrics               | `final_output/biomech_results.csv`       |
| 5    | `overlay.py`   | Add metric annotations onto original video                       | `final_output/hardik_overlayed.mp4`      |
| 6    | `feedback.py`  | Generate structured feedback using Gemini and save as Markdown   | `final_output/biomech_feedback.md`       |

---

## 🧠 Skills & Technologies Used

| Area                   | Tools / Techniques                         |
|------------------------|---------------------------------------------|
| **3D Pose Estimation** | Apple CoMotion, SMPL model                  |
| **Phase Labeling**     | OpenCV GUI for manual frame annotation      |
| **Numerical Analysis** | NumPy, pandas                               |
| **Biomechanics**       | Joint angle and stride calculations         |
| **AI Feedback**        | Google Gemini API                           |
| **Video Processing**   | OpenCV, overlay rendering                   |
| **Code Automation**    | Python scripting                            |

---

## 🔐 Gemini API Setup

In `feedback.py`, you must paste your **own Gemini API key** directly here:

```python
API_KEY = "YOUR_API_KEY_HERE"
```

---

## ✅ Requirements

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

> You must also install and configure Apple CoMotion and its SMPL dependencies as described above.

---

## 📚 References & Inspiration

- [Attractors in Fast Bowling – Steff Jones](https://www.linkedin.com/pulse/attractors-fast-bowling-steff-jones/) — Excellent guide that informed the biomechanical metric selection and interpretation.

---

## 🙌 Acknowledgements

- [Apple CoMotion](https://github.com/apple/ml-comotion) — for the SMPL-based tracking framework  
- [Google Gemini](https://ai.google.dev/) — for enabling intelligent performance feedback  
- [SMPL Model](http://smpl.is.tue.mpg.de/) — for realistic human body mesh reconstruction  

---

## 📎 License

This project is intended for educational and research use.  
Apple CoMotion and Gemini API are subject to their respective licenses and terms of use.

---

