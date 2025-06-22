import openai
from PIL import Image
import base64
import os
import cv2
import time
from classify import classify_image_with_blip

def alert_emergency(label, image_path):
    print(f"[ALERT] Detected {label.upper()} in frame: {image_path}")

def extract_and_classify(source=0, output_dir='extracted_images', interval=0.5):
    output = []

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30
    frame_interval = int(fps * interval)

    frame_count = 0
    saved_count = 0

    print(f"[INFO] Starting stream at {fps:.1f} FPS, analyzing every {interval}s...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video finished or failed to read frame.")
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")

            label = classify_image_with_blip(filename)
            print(f"[CLASSIFY] Frame {saved_count:04d}: {label}")


            if label in ['fire', 'crash']:
                alert_emergency(label, filename)

            output.append(label)

            saved_count += 1
            frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print(f"[INFO] Frame {frame_id} / {total_frames}")

            time.sleep(0.5)

        frame_count += 1

    cap.release()
    print("[INFO] Stream ended.")
    return output


