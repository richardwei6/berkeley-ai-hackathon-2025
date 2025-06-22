from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_pil_image(image: Image.Image):
    try:
        inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).lower()

        print(f"[BLIP] Caption: {caption}")

        if "fire" in caption:
            return "fire"
        elif "crash" in caption or "accident" in caption or "wreck" in caption:
            return "crash"
        else:
            return "none"

    except Exception as e:
        print(f"[ERROR] BLIP classification failed: {e}")
        return "error"


from server_connection import fetch_image_from_server
def classify_remote_image(url):
    print('------------------------------------------------------------------------------')
    image = fetch_image_from_server(url)
    
    if image is None:
        print("[ERROR] Could not fetch image.")
        return "error"

    label = classify_pil_image(image)
    print(f"[CLASSIFY] Remote image classified as: {label}")
    return label

