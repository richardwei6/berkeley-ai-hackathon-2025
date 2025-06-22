import requests
import base64
import json
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from classify import classify_pil_image


def fetch_image_from_server(url):
    try:
        print("[SERVER] Sending GET request...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Attempt to decode raw byte content as UTF-8, ignoring errors
        raw_text = response.content.decode("utf-8", errors="ignore")
        start_index = raw_text.find("iVBOR")
        if start_index == -1:
            raise ValueError("No valid base64 image data found in response")

        base64_data = raw_text[start_index:].strip()
        # Decode and open image
        print("[SERVER] Decoding base64 image data...")
        missing_padding = len(base64_data) % 4
        if missing_padding != 0:
            base64_data += '=' * (4 - missing_padding)
        image_bytes = base64.b64decode(base64_data)

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Emergency Feed", cv_image)
        cv2.waitKey(1)


        image.save("server_image.jpg")
        print("[SERVER] Image saved as server_image.jpg")
        
        label = classify_pil_image(image)
        if label in ["fire", "crash"]:
            # payload = {
            #     "label": label,
            #     "image": base64_data
            # }
            send_url = "https://e852-2607-f140-400-76-b52a-4e04-73ef-52d6.ngrok-free.app/api/emergency-detection-base64"
            try:
                response = requests.post(send_url, data=base64_data, timeout=10)
                if response.status_code == 200 or response.status_code == 500:
                    print(f"[SERVER] Emergency '{label}' data sent successfully.")
                else:
                    print(f"[SERVER] POST failed with status {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[ERROR] Error sending emergency POST: {e}")
        else:
            print(f"[INFO] No emergency detected (label: {label}) — not sending.")

        return image

    except requests.RequestException as req_err:
        print(f"[SERVER] Request failed: {req_err}")
        return None
    except Exception as e:
        print(f"[SERVER] Error decoding or saving image: {e}")
        return None