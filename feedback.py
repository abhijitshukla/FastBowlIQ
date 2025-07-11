import pandas as pd
import google.generativeai as genai

# === CONFIG ===
CSV_PATH = "final_output/biomech_results.csv"
API_KEY = "****your api key****"  # <-- Replace with your actual Gemini API key

# === Setup Gemini ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# === Load biomechanical data ===
df = pd.read_csv(CSV_PATH)
row = df.iloc[0]

# === Construct Prompt ===
prompt = f"""
You are a professional cricket biomechanics analyst.

A bowler's performance has been measured and the following biomechanical metrics were recorded:

- Back Knee Flexion at BFC: {row['Back_Knee_Angle_BFC']:.1f}°
- Stride Length: {row['Stride_Length_m']:.2f} m ({row['Stride_Length_in']:.1f} in)
- Alignment: {row['Alignment']}
- Elbow Angle at Release: {row['Max_Elbow_Angle']:.1f}°
- Hip-Shoulder Separation: {row['Hip_Shoulder_Separation']:.1f}°
- Front Knee Angle at Release: {row['Front_Knee_Angle_Release']:.1f}°
- Delivery Reach: {row['Delivery_Reach_m']:.3f} m
- Lateral Flexion: {row['Lateral_Flexion']:.1f}°

Provide detailed analysis with:
1. Feedback on each metric (what's good, what needs improvement).
2. Injury risk level for each aspect (Low/Medium/High) and why.
3. Overall injury risk comment.
4. Overall biomechanics efficiency score out of 10, with justification.

Format it clearly using Markdown with tables and headings.

Do **not** include any bowler name, date, or assumptions about location, match, or context.
"""

# === Generate feedback ===
response = model.generate_content(prompt)

# === Output to terminal ===
print("\n🎯 Biomechanical Feedback:\n")
print(response.text)

# === Save as Markdown ===
markdown_path = "final_output/biomech_feedback.md"
with open(markdown_path, "w", encoding="utf-8") as f:
    f.write("# 🏏 Bowler Biomechanics Analysis\n\n")
    f.write(response.text)

print(f"\n✅ Feedback saved to Markdown file: {markdown_path}")
