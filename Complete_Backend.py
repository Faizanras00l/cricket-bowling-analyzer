import cv2
import mediapipe as mp
import numpy as np
import requests
import os

# === OpenRouter DeepSeek Setup ===
API_KEY = "sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  # ← Replace with your actual key
MODEL_ID = "deepseek/deepseek-chat"

def deepseek_feedback(features):
    if not API_KEY or "sk-or-" not in API_KEY:
        print("⚠️ API key not configured.")
        return

    prompt = f"""
You are a cricket biomechanics coach. Based on this data:
- Elbow: {features['elbow_angle']}°
- Shoulder: {features['shoulder_angle']}°
- Arm verticality: {features['arm_verticality']}°
- Stride length: {features['stride_length']}×

Give 2 short tips (under 20 words each) to improve. Be direct and cricket-specific.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            reply = res.json()["choices"][0]["message"]["content"]
            print("\n🤖 AI Feedback:\n" + reply.strip() + "\n")
        else:
            print(f"⚠️ API error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ API request failed: {str(e)}")

# === Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def extract_3d(part, lm):
    p = lm[mp_pose.PoseLandmark[part].value]
    return [p.x, p.y, p.z], p.visibility

def extract_features(landmarks):
    ls, vs1 = extract_3d('LEFT_SHOULDER', landmarks)
    le, vs2 = extract_3d('LEFT_ELBOW', landmarks)
    lw, vs3 = extract_3d('LEFT_WRIST', landmarks)
    lh, vs4 = extract_3d('LEFT_HIP', landmarks)
    la, vs5 = extract_3d('LEFT_ANKLE', landmarks)
    ra, vs6 = extract_3d('RIGHT_ANKLE', landmarks)
    rs, _   = extract_3d('RIGHT_SHOULDER', landmarks)

    if min(vs1, vs2, vs3, vs4, vs5, vs6) < 0.6:
        return None

    elbow_angle = calculate_3d_angle(ls, le, lw)
    shoulder_angle = calculate_3d_angle(lh, ls, le)
    arm_verticality = calculate_3d_angle(lh, ls, lw)
    shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
    stride = np.linalg.norm(np.array(la) - np.array(ra)) / shoulder_width if shoulder_width else 0

    return {
        "elbow_angle": round(elbow_angle, 1),
        "shoulder_angle": round(shoulder_angle, 1),
        "arm_verticality": round(arm_verticality, 1),
        "stride_length": round(stride, 2)
    }

def rule_based_feedback(f):
    tips = []
    if f['elbow_angle'] < 165:
        tips.append(f"Elbow: {f['elbow_angle']}° → Extend >165°")
    if f['stride_length'] > 1.6:
        tips.append(f"Stride: {f['stride_length']}× → Reduce to <1.5×")
    if f['arm_verticality'] < 80:
        tips.append(f"Arm: {f['arm_verticality']}° → Keep >85°")
    if f['shoulder_angle'] > 40:
        tips.append(f"Shoulder: {f['shoulder_angle']}° → Tighten <35°")
    return tips if tips else ["✅ Form looks legal."]

def draw_info_box(frame, features, feedback):
    h, w, _ = frame.shape
    x, y = 10, h - 150
    cv2.rectangle(frame, (x, y), (x + 460, y + 140), (20, 20, 20), -1)
    cv2.putText(frame, "🏏 Pose Feedback", (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    angles = [
        f"Elbow Angle     : {features['elbow_angle']}°",
        f"Shoulder Angle  : {features['shoulder_angle']}°",
        f"Arm Verticality : {features['arm_verticality']}°",
        f"Stride (norm)   : {features['stride_length']}×"
    ]
    for i, a in enumerate(angles):
        cv2.putText(frame, a, (x+10, y+50 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,255,200), 1)
    for i, tip in enumerate(feedback[:3]):
        cv2.putText(frame, tip[:40], (x+250, y+30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

def analyze_bowling_pose(video_path, output_path="pose_feedback_output.mp4"):
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    ai_prompt_printed = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            features = extract_features(result.pose_landmarks.landmark)
            if features:
                feedback = rule_based_feedback(features)
                draw_info_box(frame, features, feedback)

                if not ai_prompt_printed:
                    deepseek_feedback(features)
                    ai_prompt_printed = True

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✅ Done! Output saved as: {output_path}")

# === Replace this with your local video path ===
video_path = "your_video.mp4"  # e.g. r"C:\Users\Faizan\Videos\bowling_clip.mp4"
analyze_bowling_pose(video_path)
