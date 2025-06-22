import requests
import base64
from io import BytesIO
from PIL import Image

url = "https://fb3d-2607-f140-400-68-3006-f365-d41f-e5a6.ngrok-free.app/screenshot_full"

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
    image_bytes = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image.save("test_server_image.jpg")
    print("[SERVER] Image saved as test_server_image.jpg")

except requests.RequestException as req_err:
    print(f"[SERVER] Request failed: {req_err}")
except Exception as e:
    print(f"[SERVER] Error decoding or saving image: {e}")
