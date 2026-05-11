# Smart Retail AI Analytics System

## Overview

An AI-powered Smart Retail Analytics System built using YOLOv8, DeepSORT, Flask, OpenCV, and SQLite.

This project transforms CCTV/video feeds into intelligent retail analytics dashboards capable of real-time people detection, tracking, heatmap generation, crowd monitoring, and customer behavior analysis.

---

## Features

### AI Detection & Tracking
- Real-time person detection using YOLOv8
- Multi-object tracking using DeepSORT
- Re-Identification (ReID) support
- Unique ID assignment for visitors

### Retail Analytics
- Live people counting
- Entry/Exit counting
- Inside store analytics
- Dwell-time tracking
- Crowd alert system

### Heatmap Analytics
- Customer movement visualization
- High-traffic zone detection
- Heatmap overlay generation

### Privacy Features
- Automatic person blur system
- Privacy-preserving surveillance

### Dashboard
- Live AI camera feed
- Real-time analytics cards
- Visitor analytics charts
- Flask-based dashboard UI

### Database & Reports
- SQLite database integration
- CSV analytics export

---

## Technologies Used

- Python
- YOLOv8
- DeepSORT
- OpenCV
- Flask
- SQLite
- NumPy
- Pandas
- HTML/CSS
- Chart.js

---

## System Architecture

1. Video/Webcam Input
2. YOLOv8 Person Detection
3. DeepSORT Tracking & ReID
4. Analytics Processing
5. Heatmap Generation
6. Database Storage
7. Flask Dashboard Visualization

---

## Project Structure

```bash
RetailAI/
│
├── main.py
├── tracker.py
├── heatmap.py
├── export_csv.py
├── database.py
├── retail_ai.db
├── yolov8n.pt
│
├── templates/
│   └── index.html
│
├── README.md
```

---

## Installation

### Clone Repository

```bash
git clone YOUR_GITHUB_LINK
cd RetailAI
```

### Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install flask
pip install numpy
pip install pandas
pip install deep-sort-realtime
```

---

## Run Project

```bash
python main.py
```

Open browser:

```bash
http://127.0.0.1:5000
```

---

## Key Features Demonstrated

- Real-time AI surveillance
- Multi-object tracking
- Retail customer analytics
- Crowd monitoring
- Heatmap analytics
- Dwell-time analysis
- Privacy-preserving AI
- Dashboard development

---

## Future Improvements

- Multi-camera support
- Cloud deployment
- Face recognition
- Telegram/Email alerts
- Power BI integration
- Suspicious activity detection
- React frontend

---

## Author

Deependra Kumar
