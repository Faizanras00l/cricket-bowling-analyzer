# BowlForm AI – Cricket Bowling Action Analyzer

BowlForm AI is an AI-powered web application that analyzes cricket bowling actions from video. It uses advanced pose estimation (MediaPipe) and integrates with the DeepSeek model (via OpenRouter) to provide instant, actionable feedback for bowlers.

---

## 🚀 Capabilities

- **Video Upload:** Accepts `.mp4`, `.mov`, `.avi` bowling action videos.
- **Pose Estimation:** Extracts key biomechanical metrics (elbow angle, knee angle, arm verticality, stride length) using MediaPipe.
- **AI Coaching Feedback:** Sends pose data to DeepSeek (via OpenRouter) for cricket-specific improvement tips.
- **Real-Time Metrics:** Displays joint angles and stride metrics as the video plays.
- **Dual Video Output:** Generates both a full annotated video and a skeleton-only video.
- **Download Results:** Lets users download the analyzed videos.
- **Modern UI:** Responsive, glassmorphic interface with smooth transitions.

---

## 📂 Project Structure

- [`Complete Project.py`](Complete%20Project.py): Main Flask backend and frontend (single-file web app).
- [`Complete front End .html`](Complete%20front%20End%20.html): Standalone HTML frontend (for design reference).
- [`Complete_Backend.py`](Complete_Backend.py): Standalone backend script (for CLI/testing).


## 🛠️ Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- [Git](https://git-scm.com/)
- Free [OpenRouter](https://openrouter.ai/) account and API key (for DeepSeek model access)

---

## ⚡ Installation

1. **Clone the Repository**

   ```sh
   !git clone https://github.com/Faizanras00l/cricket-bowling-analyzer.git
   %cd cricket-bowling-analyzer
   ```

2. **Install Python Dependencies**

   ```sh
   pip install flask opencv-python mediapipe numpy requests
   ```

---

## 🔑 Getting a Free OpenRouter API Key

1. Go to [OpenRouter.ai](https://openrouter.ai/).
2. Sign up for a free account.
3. Visit the [API Keys page](https://openrouter.ai/keys).
4. Click **Create Key** and copy your key (starts with `sk-or-...`).
5. Free trial credits are available for DeepSeek and other models.

---

## 🛡️ Setting Up the API Key

- Open [`Complete Project.py`](Complete%20Project.py).
- Find the line:

  ```python
  API_KEY = "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"
  ```

- Replace the value with your own API key from OpenRouter:

  ```python
  API_KEY = "sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  ```

---

## ▶️ Running the Application

1. **Start the Flask Server**

   ```sh
   python "Complete Project.py"
   ```

2. **Open in Browser**

   Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

3. **Upload a Bowling Video**

   - Click "Upload Video" and select your bowling action video.
   - Wait for processing (may take up to a minute).
   - View AI feedback, real-time metrics, and download results.

---

## 📝 Notes

- **Privacy:** Uploaded videos are processed locally and deleted after analysis.
- **API Usage:** Free OpenRouter credits may be limited; upgrade if needed for more usage.
- **Performance:** For best results, use clear, side-view, slow-motion bowling videos.
- **No Other APIs Needed:** You only need your own OpenRouter API key—no other API keys or services are required.

---

## ❓ FAQ

**Q: Should I add a demo video to GitHub?**  
A: Yes! Adding a demo video or GIF helps users understand your project quickly. Place it in your repo and reference it in the README as shown above.

**Q: What if I get API errors?**  
A: Make sure your OpenRouter API key is correct and has credits.

**Q: What video formats are supported?**  
A: `.mp4`, `.mov`, `.avi`

---

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

## 🙏 Credits

- [MediaPipe](https://mediapipe.dev/) for pose estimation.
- [OpenRouter](https://openrouter.ai/) and [DeepSeek](https://deepseek.com/) for AI feedback.

---

## 📬 Contact


For questions or contributions, open an issue or contact [faizanrasool20004@gmail.com](mailto:faizanrasool20004@gmail.com).

