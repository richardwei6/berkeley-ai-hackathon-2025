import requests
import base64
import json
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from classify import classify_pil_image

import requests
import msgpack
import base64
import os

def decode_screenshot_response(response_bytes):
    unpacked = msgpack.unpackb(response_bytes, raw=False)
    img_b64 = unpacked.get("image")
    img_bytes = base64.b64decode(img_b64)

    image_path = "decoded_screenshot.jpg"
    with open(image_path, "wb") as f:
        f.write(img_bytes)

    loc = unpacked.get("loc")
    print("Location:", loc)
    print(f"Image saved to {image_path}")

    return image_path, loc

def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Converts a PIL Image to a base64 string.

    Args:
        image (Image.Image): The image to convert.
        format (str): The format to save the image in (e.g., 'JPEG', 'PNG').

    Returns:
        str: Base64-encoded string of the image.
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    image_bytes = buffer.read()
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    return base64_str
    

def fetch_image_from_server(url):
    try:
        print("[SERVER] Sending GET request...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        if response.status_code == 200:
            imgpath, loc = decode_screenshot_response(response.content)
        else:
            print(f"Failed to get screenshot: {response.status_code}")

        cv_image = cv2.imread(imgpath)
        cv2.imshow("Live Emergency Feed", cv_image)
        cv2.waitKey(1)

        img = Image.open(imgpath)
        
        label = classify_pil_image(img)
        if label in ["fire", "crash"]:
            send_url = "https://e852-2607-f140-400-76-b52a-4e04-73ef-52d6.ngrok-free.app/api/emergency-detection-base64"
            try:
                response = requests.post(send_url, data=image_to_base64(img), timeout=10)
                if response.status_code == 200:
                    print(f"[SERVER] Emergency '{label}' data sent successfully.")
                else:
                    print(f"[SERVER] POST failed with status {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[ERROR] Error sending emergency POST: {e}")
        else:
            print(f"[INFO] No emergency detected (label: {label}) — not sending.")

        return img

    except requests.RequestException as req_err:
        print(f"[SERVER] Request failed: {req_err}")
        return None
    except Exception as e:
        print(f"[SERVER] Error decoding or saving image: {e}")
        return None


