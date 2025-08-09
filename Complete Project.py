import cv2
import mediapipe as mp
import numpy as np
import requests
import os
import uuid
from flask import Flask, request, jsonify, render_template_string, url_for

# ==============================================================================
# 1. FLASK APPLICATION SETUP
# ==============================================================================

app = Flask(__name__, static_folder='static')

# --- Configuration ---
# Replace with your actual key
API_KEY = "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"
MODEL_ID = "deepseek/deepseek-chat"
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER


# ==============================================================================
# 2. BACKEND: POSE ANALYSIS & AI FEEDBACK LOGIC
# ==============================================================================

# === AI Feedback Setup (Modified to return data, not print) ===
def deepseek_feedback(features):
    """
    Sends pose features to the DeepSeek model via OpenRouter for feedback.
    Returns the AI's feedback as a string.
    """
    if not API_KEY or "sk-or-" not in API_KEY:
        print("API key not configured.")
        return "API key not configured. Please check the server."

    # --- FIX: Changed prompt to request 4 tips instead of 2 ---
    prompt = f"""
You are a professional cricket biomechanics coach. Based on this data from a bowling action's release point:
- Elbow Angle: {features['elbow_angle']}°
- Shoulder Angle: {features['shoulder_angle']}°
- Arm Verticality: {features['arm_verticality']}°
- Stride Length (normalized to shoulder width): {features['stride_length']}×

Give 4 very short, direct, and actionable tips (under 20 words each) to improve the bowling form, focusing on different aspects like balance, power, injury prevention, and efficiency.
Format the output as a simple list. Example:
- Keep your elbow straighter for more power.
- Shorten your stride to improve balance.
- Engage your core for better stability.
- Follow through fully towards the target.
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
            return reply.strip()
        else:
            error_message = f"API error {res.status_code}: {res.text}"
            print(f" {error_message}")
            return f"Could not get AI feedback. {error_message}"
    except Exception as e:
        error_message = f"API request failed: {str(e)}"
        print(f" {error_message}")
        return f"Could not get AI feedback. {error_message}"


# === MediaPipe Pose Setup ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_angle(a, b, c):
    """Calculates the angle between three 3D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def extract_3d(part, lm):
    """Extracts 3D coordinates and visibility for a given body part."""
    p = lm[mp_pose.PoseLandmark[part].value]
    return [p.x, p.y, p.z], p.visibility

def extract_features(landmarks):
    """Extracts key biomechanical features from pose landmarks."""
    try:
        ls, vs1 = extract_3d('LEFT_SHOULDER', landmarks)
        le, vs2 = extract_3d('LEFT_ELBOW', landmarks)
        lw, vs3 = extract_3d('LEFT_WRIST', landmarks)
        lh, vs4 = extract_3d('LEFT_HIP', landmarks)
        lk, _   = extract_3d('LEFT_KNEE', landmarks)
        la, vs5 = extract_3d('LEFT_ANKLE', landmarks)
        ra, vs6 = extract_3d('RIGHT_ANKLE', landmarks)
        rs, _   = extract_3d('RIGHT_SHOULDER', landmarks)

        if min(vs1, vs2, vs3, vs4, vs5, vs6) < 0.6:
            return None

        shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
        
        return {
            "elbow_angle": round(calculate_3d_angle(ls, le, lw), 1),
            "shoulder_angle": round(calculate_3d_angle(lh, ls, le), 1),
            "arm_verticality": round(calculate_3d_angle(lh, ls, lw), 1),
            "front_knee_angle": round(calculate_3d_angle(lh, lk, la), 1),
            "stride_length": round(np.linalg.norm(np.array(la) - np.array(ra)) / shoulder_width if shoulder_width > 0 else 0, 2),
        }
    except Exception:
        return None

def draw_info_box(frame, features):
    """Draws the feedback box with release metrics on the video frame."""
    h, w, _ = frame.shape
    box_x, box_y, box_h = 10, h - 130, 120
    cv2.rectangle(frame, (box_x, box_y), (box_x + 280, box_y + box_h), (20, 20, 20), -1)
    
    cv2.putText(frame, "Metrics (at Release)", (box_x + 10, box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    metrics = [
        f"Elbow Angle: {features.get('elbow_angle', 'N/A')}°",
        f"Arm Verticality: {features.get('arm_verticality', 'N/A')}°",
        f"Knee Angle: {features.get('front_knee_angle', 'N/A')}°",
        f"Stride (norm): {features.get('stride_length', 'N/A')}x"
    ]
    for i, metric in enumerate(metrics):
        cv2.putText(frame, metric, (box_x + 10, box_y + 45 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

def analyze_bowling_pose(video_path):
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    # Pass 1: Extract data from all frames and find release point
    all_frame_features = []
    release_frame_features = None
    max_elbow_angle = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        frame_features = None
        if result.pose_landmarks:
            frame_features = extract_features(result.pose_landmarks.landmark)
            if frame_features and frame_features.get('elbow_angle', 0) > max_elbow_angle:
                max_elbow_angle = frame_features['elbow_angle']
                release_frame_features = frame_features
        all_frame_features.append(frame_features)
    
    ai_feedback = "No suitable pose detected for analysis."
    if release_frame_features:
        ai_feedback = deepseek_feedback(release_frame_features)

    # Pass 2: Generate both Full and Skeleton videos
    uid = uuid.uuid4().hex
    output_full_filename = f"output_full_{uid}.mp4"
    output_skeleton_filename = f"output_skeleton_{uid}.mp4"
    output_full_path = os.path.join(app.config['STATIC_FOLDER'], output_full_filename)
    output_skeleton_path = os.path.join(app.config['STATIC_FOLDER'], output_skeleton_filename)
    
    writer_full = cv2.VideoWriter(output_full_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
    writer_skeleton = cv2.VideoWriter(output_skeleton_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        
        black_frame = np.zeros_like(frame)

        if result.pose_landmarks:
            landmark_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            connection_spec = mp_drawing.DrawingSpec(color=(25, 250, 170), thickness=2, circle_radius=2)
            
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_spec, connection_spec)
            mp_drawing.draw_landmarks(black_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_spec, connection_spec)
            
            if release_frame_features:
                draw_info_box(frame, release_frame_features)
                draw_info_box(black_frame, release_frame_features)

        writer_full.write(frame)
        writer_skeleton.write(black_frame)

    cap.release()
    writer_full.release()
    writer_skeleton.release()
    
    print(f"Analysis complete! Outputs generated.")

    return {
        "output_video_full_url": url_for('static', filename=output_full_filename, _external=True),
        "output_video_skeleton_url": url_for('static', filename=output_skeleton_filename, _external=True),
        "release_features": release_frame_features if release_frame_features else {},
        "frame_data": all_frame_features,
        "ai_feedback": ai_feedback,
        "video_dimensions": {"width": w, "height": h},
        "fps": fps
    }

# ==============================================================================
# 3. FRONTEND: HTML, CSS, JAVASCRIPT
# ==============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>BowlForm AI</title>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com" rel="preconnect"/>
<link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@800&family=Montserrat:wght@300;400;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet"/>
<style type="text/tailwindcss">
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(25,250,170,0.4);}
        70% { box-shadow: 0 0 0 20px rgba(25,250,170,0);}
        100% { box-shadow: 0 0 0 0 rgba(25,250,170,0);}
    }
    .animate-float { animation: float 6s ease-in-out infinite; }
    .animate-pulse-glow { animation: pulse 2s infinite; }
    .font-orbitron { font-family: 'Orbitron', sans-serif; }
    .font-montserrat { font-family: 'Montserrat', sans-serif; }
    .fade-out {
        animation: fadeOut 1s ease-in-out forwards;
        animation-delay: 2.5s;
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; visibility: hidden; }
    }
    .particle {
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        animation: rise 10s infinite linear;
    }
    @keyframes rise {
        from { transform: translateY(100vh) scale(0); opacity: 1; }
        to { transform: translateY(-10vh) scale(1); opacity: 0; }
    }
    :root {
        --brand-glow: #19faaa;
        --neon-blue: #0ea8f0;
    }
    .glassmorphic {
        backdrop-filter: blur(12px);
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        border: 1px solid rgba(25,250,170,0.2);
        box-shadow: 0 0 20px rgba(25,250,170,0.1);
    }
    .upload-button-glow {
        box-shadow: 0 0 15px 5px rgba(25,250,170,0.3), 0 0 30px 10px rgba(25,250,170,0.1);
    }
    .upload-button-glow:hover {
        box-shadow: 0 0 20px 7px rgba(25,250,170,0.5), 0 0 40px 15px rgba(25,250,170,0.2);
    }
    .loader {
        border: 4px solid rgba(255,255,255,0.2);
        border-left-color: var(--brand-glow);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    body {
        background-color: #0d1117;
        font-family: 'Montserrat', 'Space Grotesk', sans-serif;
    }
    .orbitron { font-family: 'Orbitron', sans-serif; }
    .glassmorphic-analysis {
        background: rgba(22, 29, 39, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .pill-button {
        @apply px-4 py-2 rounded-full text-white font-bold transition-all duration-300 whitespace-nowrap;
        background: rgba(14,168,240,0.2);
        border: 1px solid var(--neon-blue);
        box-shadow: 0 0 10px rgba(14,168,240,0.4);
    }
    .pill-button:hover {
        background: rgba(14,168,240,0.5);
        box-shadow: 0 0 15px rgba(14,168,240,0.6);
    }
    .pill-button.active {
        background: var(--neon-blue);
        box-shadow: 0 0 15px rgba(14,168,240,0.6);
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: var(--neon-blue); border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover { background: #19faaa; }
</style>
</head>
<body class="bg-gradient-to-br from-[#0d0d26] to-[#103c3c] text-white">

<div id="intro-section" class="relative w-screen h-screen overflow-hidden z-50">
    <div class="absolute top-0 left-0 w-full h-full pointer-events-none">
        <div class="particle" style="width: 2px; height: 2px; left: 10%; animation-duration: 12s; animation-delay: 0s;"></div>
        <div class="particle" style="width: 3px; height: 3px; left: 20%; animation-duration: 7s; animation-delay: 1s;"></div>
    </div>
    <div class="fade-out flex flex-col items-center justify-center h-full text-center">
        <div class="animate-float">
            <div class="relative w-48 h-48 md:w-56 md:h-56 animate-pulse-glow rounded-full flex items-center justify-center bg-black/20">
                <svg class="h-24 w-24 text-[var(--brand-glow)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
        </div>
        <h1 class="font-orbitron text-white text-[42px] md:text-[56px] font-extrabold mt-8 tracking-wider">BowlForm AI</h1>
        <p class="font-montserrat text-gray-300 text-[18px] md:text-[22px] font-light mt-2 tracking-wider">Refine. Review. Rise.</p>
    </div>
</div>

<div id="upload-section" class="relative min-h-screen w-full overflow-hidden hidden">
    <div class="absolute inset-0 bg-cover bg-center" style="background-image: url('https://images.unsplash.com/photo-1595392490148-d75073809623?q=80&w=2070&auto=format&fit=crop');"></div>
    <div class="absolute inset-0 bg-gray-900/70 backdrop-blur-sm"></div>
    <div class="relative z-10 flex min-h-screen w-full items-center justify-center p-4" id="upload-container">
        <div class="w-full max-w-lg">
            <div class="glassmorphic p-6 md:p-8">
                <div class="text-center mb-8">
                    <h1 class="text-3xl md:text-4xl orbitron font-bold text-white tracking-wide">Bowling Analysis</h1>
                    <p class="text-gray-300 mt-2">Get instant AI feedback on your action.</p>
                </div>
                <form class="space-y-6" id="analysis-form">
                    <div class="text-center pt-4">
                        <label class="cursor-pointer group" for="video-upload">
                            <div class="w-40 h-40 mx-auto border-2 border-dashed border-gray-500 rounded-full flex flex-col items-center justify-center transition-all duration-300 hover:border-[var(--brand-glow)] hover:bg-white/5 upload-button-glow">
                                <span class="material-icons text-5xl text-gray-400 group-hover:text-[var(--brand-glow)] transition-colors">upload</span>
                                <span class="mt-2 text-sm font-semibold text-white">Upload Video</span>
                            </div>
                        </label>
                        <input accept=".mp4,.mov,.avi" class="hidden" id="video-upload" type="file"/>
                        <p id="file-name-display" class="text-sm text-[var(--brand-glow)] mt-4 h-5"></p>
                        <p class="text-xs text-gray-400 mt-2">Use a clear side-view video for best results.</p>
                    </div>
                </form>
            </div>
        </div>
    </div>
    <div class="hidden fixed inset-0 z-50 bg-gray-900/80 backdrop-blur-md flex items-center justify-center" id="processing-modal">
        <div class="text-center">
            <div class="loader mx-auto"></div>
            <h2 class="text-2xl font-bold mt-6 text-white orbitron">Analyzing your action...</h2>
            <p class="text-gray-300 mt-2">This may take a minute. Please wait.</p>
        </div>
    </div>
</div>

<div id="analysis-section" class="min-h-screen flex-col items-center justify-center p-4 lg:p-8 bg-gradient-to-br from-[#0d1117] to-[#101d23] hidden">
    <header class="w-full max-w-7xl mx-auto mb-6">
        <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
                <svg class="h-8 w-8 text-[var(--neon-blue)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                <h1 class="text-2xl orbitron font-bold tracking-wider">BowlForm AI</h1>
            </div>
            <button class="pill-button" id="analyze-another-btn">Analyze Another</button>
        </div>
    </header>
    <main class="w-full flex flex-col lg:flex-row items-start justify-center gap-8 max-w-7xl mx-auto">
        <aside class="w-full lg:w-1/4 glassmorphic-analysis rounded-2xl p-6 order-2 lg:order-1">
            <h2 class="orbitron text-xl font-bold mb-4 text-[var(--neon-blue)]">AI COACHING TIPS</h2>
            <div id="ai-feedback-container" class="space-y-3"></div>
        </aside>
        
        <div class="w-full lg:w-1/2 flex flex-col items-center order-1 lg:order-2">
            <div id="analysis-video-container" class="w-full bg-black rounded-2xl overflow-hidden shadow-2xl shadow-[var(--neon-blue)]/20">
                <video class="w-full h-full object-contain" controls loop id="analysis-video">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div class="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 mt-8 w-full">
                <button id="full-mode-btn" class="pill-button active w-full sm:w-auto">Full Mode</button>
                <button id="skeleton-mode-btn" class="pill-button w-full sm:w-auto">Skeleton Mode</button>
                <a id="download-btn" class="pill-button w-full sm:w-auto flex items-center justify-center gap-2 cursor-pointer">
                    <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path clip-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" fill-rule="evenodd"></path></svg>
                    Download
                </a>
            </div>
        </div>
        
        <aside class="w-full lg:w-1/4 glassmorphic-analysis rounded-2xl p-6 order-3">
            <h2 class="orbitron text-xl font-bold mb-4 text-[var(--neon-blue)]">REAL-TIME METRICS</h2>
            <div id="metrics-container" class="space-y-4">
                <div class="flex justify-between items-center"><span class="font-semibold text-gray-300">Elbow Angle</span><span class="orbitron font-bold text-2xl text-white" id="elbow-angle">...</span></div>
                <div class="flex justify-between items-center pt-2"><span class="font-semibold text-gray-300">Front Knee Angle</span><span class="orbitron font-bold text-2xl text-white" id="knee-angle">...</span></div>
                <div class="flex justify-between items-center pt-2"><span class="font-semibold text-gray-300">Arm Verticality</span><span class="orbitron font-bold text-2xl text-white" id="arm-verticality">...</span></div>
                <div class="flex justify-between items-center pt-2"><span class="font-semibold text-gray-300">Stride Length</span><span class="orbitron font-bold text-2xl text-white" id="stride-length">...</span></div>
            </div>
            <p class="text-xs text-gray-500 mt-6 text-center italic">Metrics update as video plays or is paused.</p>
        </aside>
    </main>
</div>

<script>
    // --- PAGE TRANSITIONS ---
    setTimeout(() => {
        document.getElementById('intro-section')?.style.setProperty('display', 'none', 'important');
        document.getElementById('upload-section')?.classList.remove('hidden');
    }, 3500);

    // --- DOM ELEMENTS & STATE ---
    const videoUploadInput = document.getElementById('video-upload');
    const processingModal = document.getElementById('processing-modal');
    const analysisSection = document.getElementById('analysis-section');
    const videoPlayer = document.getElementById('analysis-video');
    const fullModeBtn = document.getElementById('full-mode-btn');
    const skeletonModeBtn = document.getElementById('skeleton-mode-btn');
    const downloadBtn = document.getElementById('download-btn');

    let frameData = [];
    let videoFPS = 30;
    let videoUrlFull = '';
    let videoUrlSkeleton = '';
    let currentMode = 'full';

    // --- EVENT LISTENER FOR VIDEO UPLOAD ---
    videoUploadInput.addEventListener('change', function() {
        if (this.files && this.files.length > 0) {
            document.getElementById('file-name-display').textContent = this.files[0].name;
            document.getElementById('upload-container').classList.add('hidden');
            processingModal.classList.remove('hidden');

            const formData = new FormData();
            formData.append('video', this.files[0]);

            fetch('/analyze', { method: 'POST', body: formData })
            .then(response => {
                if (!response.ok) throw new Error(`Server error: ${response.statusText}`);
                return response.json();
            })
            .then(data => {
                if (data.error) throw new Error(data.error);
                populateAnalysisPage(data);
                
                document.getElementById('upload-section').classList.add('hidden');
                processingModal.classList.add('hidden');
                analysisSection.style.display = 'flex';
            })
            .catch(error => {
                console.error('Error during analysis:', error);
                alert(`An error occurred: ${error.message}. Please try again.`);
                document.getElementById('upload-container').classList.remove('hidden');
                processingModal.classList.add('hidden');
            });
        }
    });
    
    // --- METRICS & VIDEO HANDLING ---
    function updateMetricsDisplay(metrics) {
        document.getElementById('elbow-angle').textContent = metrics?.elbow_angle ? `${metrics.elbow_angle}°` : 'N/A';
        document.getElementById('knee-angle').textContent = metrics?.front_knee_angle ? `${metrics.front_knee_angle}°` : 'N/A';
        document.getElementById('arm-verticality').textContent = metrics?.arm_verticality ? `${metrics.arm_verticality}°` : 'N/A';
        document.getElementById('stride-length').textContent = metrics?.stride_length ? `${metrics.stride_length}x` : 'N/A';
    }
    
    function updateMetricsOnTimeUpdate() {
        const frameNumber = Math.floor(this.currentTime * videoFPS);
        if (frameData && frameNumber >= 0 && frameNumber < frameData.length) {
            updateMetricsDisplay(frameData[frameNumber]);
        } else {
            updateMetricsDisplay(null);
        }
    }

    function switchVideoMode(mode) {
        if (currentMode === mode && videoPlayer.src) return;

        const currentTime = videoPlayer.currentTime;
        const isPlaying = !videoPlayer.paused;
        currentMode = mode;
        
        const newSrc = (mode === 'full') ? videoUrlFull : videoUrlSkeleton;
        videoPlayer.src = newSrc;
        videoPlayer.load();

        const onVideoLoaded = () => {
            videoPlayer.currentTime = currentTime;
            if (isPlaying) {
                videoPlayer.play().catch(e => console.error("Autoplay prevented on switch:", e));
            }
        };
        videoPlayer.addEventListener('loadeddata', onVideoLoaded, { once: true });
        
        fullModeBtn.classList.toggle('active', mode === 'full');
        skeletonModeBtn.classList.toggle('active', mode === 'skeleton');
        downloadBtn.href = newSrc;
        downloadBtn.setAttribute('download', `bowling_analysis_${mode}_${new Date().toISOString().split('T')[0]}.mp4`);
    }

    // --- PAGE POPULATION & SETUP ---
    function populateAnalysisPage(data) {
        frameData = data.frame_data;
        videoFPS = data.fps;
        videoUrlFull = data.output_video_full_url;
        videoUrlSkeleton = data.output_video_skeleton_url;

        const videoContainer = document.getElementById('analysis-video-container');
        if (data.video_dimensions && data.video_dimensions.width > 0) {
            videoContainer.style.aspectRatio = data.video_dimensions.width / data.video_dimensions.height;
        } else {
            videoContainer.style.aspectRatio = '16 / 9';
        }

        updateMetricsDisplay(data.release_features);
        switchVideoMode('full');
        videoPlayer.src = videoUrlFull;

        const feedbackContainer = document.getElementById('ai-feedback-container');
        feedbackContainer.innerHTML = '';
        const tips = data.ai_feedback.split('\\n').filter(tip => tip.trim().startsWith('-'));
        if (tips.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'list-disc list-inside space-y-2 text-gray-200';
            tips.forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip.replace(/^-/, '').trim();
                ul.appendChild(li);
            });
            feedbackContainer.appendChild(ul);
        } else {
            feedbackContainer.innerHTML = `<p class="text-gray-300">${data.ai_feedback}</p>`;
        }
    }
    
    // --- EVENT LISTENERS ---
    // --- FIX: Added 'pause' and 'seeked' listeners for robust metric updates ---
    videoPlayer.addEventListener('timeupdate', updateMetricsOnTimeUpdate);
    videoPlayer.addEventListener('pause', updateMetricsOnTimeUpdate);
    videoPlayer.addEventListener('seeked', updateMetricsOnTimeUpdate);
    
    fullModeBtn.addEventListener('click', () => switchVideoMode('full'));
    skeletonModeBtn.addEventListener('click', () => switchVideoMode('skeleton'));
    document.getElementById('analyze-another-btn').addEventListener('click', function() {
        analysisSection.style.display = 'none';
        document.getElementById('upload-section').classList.remove('hidden');
        document.getElementById('upload-container').classList.remove('hidden');
        videoUploadInput.value = '';
        document.getElementById('file-name-display').textContent = '';
        videoPlayer.src = '';
        videoPlayer.load();
    });
</script>
</body>
</html>
"""


# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles video upload, processing, and returns JSON results."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file part"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = f"input_{uuid.uuid4().hex}.mp4"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        try:
            results = analyze_bowling_pose(input_path)
            return jsonify(results)
        except Exception as e:
            print(f"Error during analysis route: {e}")
            return jsonify({"error": "An internal error occurred during analysis."}), 500
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

# ==============================================================================
# 5. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    app.run(debug=True)