---
title: DEBI Face Recognition
emoji: 👤
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 👤 DEBI Face Recognition App

Real-time face detection and recognition platform built for the **DEBI Hackathon 2026**.

## Features

- **Snapshot Mode** — Take a photo or upload an image for instant face recognition (works on any device/network)
- **Live Mode** — Real-time webcam recognition via WebRTC
- **Database Management** — Add new faces on the fly via the sidebar
- **Deep Learning** — Powered by DeepFace (Facenet) for high-accuracy embeddings
- **CI/CD** — Automated deployment to Hugging Face Spaces via GitHub Actions

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection & Embeddings | DeepFace (Facenet model) |
| Bounding Boxes | OpenCV |
| Web Framework | Streamlit |
| Real-time Video | streamlit-webrtc |
| Database | JSON file |
| Deployment | Hugging Face Spaces |
| CI/CD | GitHub Actions |

## How It Works

1. A face is detected in the camera frame using DeepFace
2. A 128-dimensional embedding vector is extracted via the Facenet model
3. The embedding is compared against stored embeddings in `database.json` using Euclidean distance
4. If the distance is below the threshold (0.8), the person is recognized
5. A bounding box is drawn with the person's name inside (or "Unknown")

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```