from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model once globally
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def classify_image_with_blip(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

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
        print(f"[!] BLIP classification failed: {e}")
        return "error"
