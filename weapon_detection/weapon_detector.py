import torch
import cv2
import os
from PIL import Image
import numpy as np

default_yolov5_model = 'weapon_detection/yolov5s.pt'
default_output_dir = "weapon_detection/weapons"

class_names = ["knife", "pistol"]

class WeaponDetector:
    def __init__(self, model_path=default_yolov5_model, output_dir=default_output_dir):
        self.output_dir = output_dir

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load YOLOv5 model using torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = 0.25  # confidence threshold

        # Get class indices for weapons
        self.weapon_class_ids = []
        for class_id, name in self.model.names.items():
            if name.lower() in ['knife', 'pistol']:
                self.weapon_class_ids.append(class_id)
        if not self.weapon_class_ids:
            raise ValueError("The model does not contain weapon classes (knife, pistol).")

    def _detect(self, imgfile):
        orig_img = cv2.imread(imgfile)
        if orig_img is None:
            raise ValueError(f"Could not read image: {imgfile}")
        results = self.model(orig_img)
        detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

        weapon_crops = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) in self.weapon_class_ids and conf > 0.4:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cropped = orig_img[y1:y2, x1:x2]
                weapon_name = self.model.names[int(cls)]
                weapon_crops.append((cropped, float(conf), weapon_name))
        return weapon_crops

    def detect_dir(self, directory):
        for f in os.listdir(directory):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.detect(os.path.join(directory, f))

    def detect(self, filepath):
        weapon_crops = self._detect(filepath)

        if weapon_crops:
            filename = os.path.basename(filepath)
            image_output_dir = os.path.join(self.output_dir, filename)
            os.makedirs(image_output_dir, exist_ok=True)

            output_filenames = []
            for i, (cropped, confidence, weapon_name) in enumerate(weapon_crops):
                if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    print(f"Skipping weapon {i} - too small")
                    continue
                output_filename = f"{image_output_dir}/{weapon_name}_{i}_conf_{confidence:.2f}.jpg"
                cv2.imwrite(output_filename, cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
                print(f"Saved: {output_filename}")
                output_filenames.append(output_filename)
            return output_filenames
        else:
            print(f"No weapons detected in {filepath}")
