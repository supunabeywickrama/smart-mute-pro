# 🔮 VoCleanse — AI-Powered Voice Purification in One Click

## 🧩 Overview
VoCleanse is an intelligent AI-based audio–video processing tool that automatically detects and censors unwanted words (profanity, filler words, or sensitive terms) from any uploaded audio or video file.  
Using speech-to-text models, fuzzy matching, and intelligent audio synthesis, VoCleanse lets users preview and export clean, professional-sounding media with just one click.

Demo video: ([link](https://drive.google.com/drive/folders/1zdso1EAJzdktsV8CwIMX0BK0kwiz8s4D))
---

## ⚙️ AI / Machine Learning Features

| AI Component | Technology Used | Description |
|-------------|----------------|-------------|
| 🎙️ Automatic Speech Recognition (ASR) | Faster-Whisper (OpenAI Whisper) | Converts speech into text with precise word timestamps. High accuracy for real/noisy speech. |
| 🧩 Word-Level Segmentation | Whisper `word_timestamps=True` | Extracts start/end time of each spoken word for precise beeps/muting. |
| 💡 Fuzzy Word Matching (NLP) | RapidFuzz (Levenshtein similarity) | Detects variations of banned words for robust censoring. |
| 🔊 Smart Audio Censorship | PyDub + FFmpeg | Synthesizes beep or volume-ducking with smooth transitions. |
| 🧠 Adaptive Audio Timing | Dynamic segmentation logic | Adds margins (±60 ms default) to avoid partial phoneme leakage. |
| 📊 Confidence-Based Filtering | ASR probability score | Mute only high-certainty detections. |
| 🗣️ Human-in-the-loop Review | Streamlit UI | Edit flagged words, preview changes, export results. |
| 🔁 Continuous Learning Ready | Optional model fine-tuning | Can be extended for regional language or accent support. |

---

## 🚀 Key Functional Features

| Feature | Description |
|--------|-------------|
| 🎧 Multiformat Input | Supports .mp4, .mov, .wav, .m4a, etc. |
| 🧠 AI Word Recognition | Precise timestamps for all spoken words |
| 🕵️ Selective Mute Logic | Exact + fuzzy matching options |
| 🔇 Beep or Duck Modes | Replace voice with beep or soft attenuation |
| 📊 Interactive Dashboard | Editable word table and mute flags |
| 📉 Timeline Visualization | Shows muted vs active segments (Altair) |
| 🔁 Before/After Preview | A/B check without exporting |
| 📦 Output | Export .csv, .json, .srt, .mp4, .wav |
| ⚡ Fast Processing | GPU acceleration supported |
| 🌈 Modern UI | Fully dark-mode optimized |

---

## 🧠 Tech Stack Summary

| Category | Tool / Library | Purpose |
|---------|----------------|---------|
| Core ML | faster-whisper | Speech recognition |
| NLP Matching | rapidfuzz | Fuzzy banned-word detection |
| Audio Processing | pydub + ffmpeg | Beeps, slicing, mixing & crossfades |
| Frontend UI | streamlit | Interactive browser app |
| Visualization | altair + pandas | Timeline, tables |
| Deployment | Hugging Face Spaces | Cloud hosting |
| GPU Runtime | onnxruntime/cuda | Speedup inference |

---

## 📂 Project Structure

<img width="696" height="450" alt="image" src="https://github.com/user-attachments/assets/e7b91465-8efd-43e9-a531-d6fdfa34282b" />


---

## ⚙️ How It Works (Pipeline Flow)
<img width="477" height="736" alt="image" src="https://github.com/user-attachments/assets/e0d39033-6e39-48d8-a638-983dc3e181e5" />

---

## 🧾 System Requirements

| Component | Recommended |
|----------|-------------|
| OS | Windows 10 / Ubuntu 22+ |
| Python | 3.9 – 3.11 |
| RAM | ≥ 4 GB |
| GPU | CUDA-capable (optional speedup) |
| FFmpeg | Installed or declared in `packages.txt` |

---

## 💻 Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
---

## ☁️ Deploy on Hugging Face Spaces

Fully deployable using Streamlit SDK.
Add this to top of README.md when using Spaces:
---
title: VoCleanse
emoji: 🔮
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.37.0"
app_file: app.py
pinned: false
license: mit
---

Push repo:
git add .
git commit -m "Deploy VoCleanse to Hugging Face"
git push origin main

---

## 🧪 Future Enhancements


🎯 Fine-tuned profanity/keyword detection


🗣️ Speaker diarization (mute selected voices only)


🔁 Batch uploads for large workflows


📊 Analytics dashboard (usage, confidence)


🌍 Multilingual UI (Sinhala, Tamil, Hindi, etc.)

---

## 👨‍💻 Developer

Supun Tharaka (ALSupun)
🎓 B.Sc. (Hons) Electronics & Computer Science
🏫 University of Kelaniya
💡 Embedded Systems, AI & Smart Automation
📫 Contact: supuun2001@gmail.com
🔗 GitHub: ([link](https://github.com/supunabeywickrama))
🔗 LinkedIn: ([link](https://www.linkedin.com/in/supun-tharaka-6bb8b5278/))


---


