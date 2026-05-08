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

## 🏆 Hackathon Compliance (DEBI 2026)

This project is built according to the official Debi Hackathon guidelines:

- [x] **Device Camera Access**: Implemented via `st.camera_input` (Snapshot) and `streamlit-webrtc` (Live).
- [x] **Face Recognition**: Uses `face_recognition` (dlib-based) for high-accuracy embeddings.
- [x] **Comparison Engine**: Compares detected embeddings against a pre-stored `database.json` file.
- [x] **Visual Feedback**:
    - ✅ **Recognized**: Green boundary box with name and confidence percentage.
    - ❓ **Unknown**: Red boundary box labeled "Unknown".
- [x] **Database**: Structured JSON storage associating 128D embeddings with names.
- [x] **Deployment**: Successfully deployed to Hugging Face Spaces.
- [x] **CI/CD**: Fully automated pipeline via GitHub Actions (Lint $\rightarrow$ Verify $\rightarrow$ Deploy).
