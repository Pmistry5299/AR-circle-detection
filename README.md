# AR-circle-detection
Augmented Reality project using OpenCV and WebSocket to overlay Earth image on detected circles.
# 🔍 Augmented Reality Circle Detection with Earth Overlay 🌍

This is a beginner AR project where I combined **OpenCV** and **WebSocket** to create an interactive visual that detects circles in a webcam feed and overlays a rotating Earth image on top of them. All rendered in real time!

## 🎯 What it Does

- Uses Hough Circle Detection to detect round markers in a video feed.
- Overlays a transparent Earth image on the detected circles.
- Continuously rotates the Earth overlay.
- Broadcasts detected circle data to any WebSocket-connected frontend (can be used for web visualizations or Unity integration).

https://user-images.githubusercontent.com/your-profile/video-preview.mp4 *(Replace with actual GitHub-hosted link or YouTube)*

## 🛠️ Tech Used

- Python
- OpenCV
- Websocket Server
- NumPy
- Threading

## 🧠 What I Learned

- Real-time object detection with OpenCV.
- Using alpha channels for image transparency.
- Thread-safe WebSocket communication.
- Circle consolidation to avoid false positives.

## 📸 Video Demo

> 🎬 
## 🚀 Run it Locally

```bash
pip install opencv-python websocket-server numpy
python main.py
