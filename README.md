README
CalHacks AI 2025
# Total Emergency Detection AI

This project leverages AI and computer vision to **detect real-world emergencies from streamed video data**, including:

- Fires  
- Car crashes  
- People collapsing (e.g., fainting, seizures, medical emergencies)

When an emergency is detected, the system can **automatically notify the appropriate first responders** via configurable alert mechanisms (e.g., webhooks, email, SMS, or 911 dispatch).

---

## Project Goal

To develop an AI-powered pipeline that monitors live or remote video feeds, **analyzes frames in real time**, and triggers emergency response workflows with high confidence — reducing response time and improving public safety.

---

## Features

- **Live camera or video file input**
- **Remote image fetching via HTTP server**
- **Image extraction at fixed intervals (e.g., every 0.5 seconds)**
- **AI-based emergency classification using open-source vision models**
- **Supports fire, crash, and collapse detection**
- **Auto-saving of emergency frames**
- **(Coming soon)**: Auto-alert system for responders

---

## Setup

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/emergency-ai.git
cd emergency-ai
bash'''

### 2. Install Dependencies
'''bash
pip install -r requirements.txt '''

BLIP is downloaded automatically using transformers. No API key required.

---

## How it Works

### A. Local Video Stream
'''bash
python extract_from_video.py --source path/to/video.mp4 '''

This script:
Extracts frames every 0.5 seconds
Saves them to extracted_images/
Classifies each image and prints result

### B. Remote Image Server (Optional)
To fetch images from a remote endpoint (e.g., /screenshot_full on an ngrok server):
'''bash
python classify_remote.py '''

This script:
Sends a GET request
Decodes base64 image response
Classifies the image (e.g., "fire" or "none")
Saves it if it contains an emergency
