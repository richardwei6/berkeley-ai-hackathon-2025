import os
from camera_input.camera_input import CameraInput
from person_detection.people_cropper import PeopleCropper
from flask import Flask, send_file
import base64
import msgpack
from datetime import datetime, timedelta
from weapon_detection.weapon_detector import WeaponDetector

test_loc = "37.8688956,-122.2600617"

shared_screenshots_dir = "./shared/screenshots"
shared_people_dir = "./shared/people"
shared_weapon_dir = "./shared/weapon"

if not os.path.exists(shared_screenshots_dir):
    os.makedirs(shared_screenshots_dir)
if not os.path.exists(shared_people_dir):
    os.makedirs(shared_people_dir)
if not os.path.exists(shared_weapon_dir):
    os.makedirs(shared_weapon_dir)

# Test camera input
camera = CameraInput(output_dir=shared_screenshots_dir)

# Detection models
people_cropper = PeopleCropper(output_dir=shared_people_dir)
weapon_detector = WeaponDetector(output_dir=shared_weapon_dir)

app = Flask(__name__)

@app.route('/screenshot_people', methods=['GET'])
def screenshot_people():
    print("Taking screenshot and detecting people")
    # Take screenshot using camera
    filename = camera.take_screenshot()
    
    if filename and os.path.exists(filename):
        remove_old_screenshots()
        # Process the image to detect people
        people_filenames = people_cropper.detect(filename)
        # Convert each cropped image to base64
        if (people_filenames is None or len(people_filenames) == 0):
            return "No people detected", 400

        encoded_images = []
        for person_file in people_filenames:
            with open(person_file, 'rb') as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                encoded_images.append(img_b64)

        # Pack into messagepack format
        response = {
            'people_images': encoded_images,
            "loc": test_loc,
        }
        packed_response = msgpack.packb(response)

        return packed_response, 200, {'Content-Type': 'application/x-msgpack'}
    else:
        return "Failed to take screenshot", 500

@app.route('/screenshot_weapons', methods=['GET'])
def screenshot_weapons():
    print("Taking screenshot and detecting weapons")
    # Take screenshot using camera
    filename = camera.take_screenshot()
    
    if filename and os.path.exists(filename):
        remove_old_screenshots()
        # Process the image to detect weapons
        weapon_filenames = weapon_detector.detect(filename)
        # Convert each cropped image to base64
        if (weapon_filenames is None or len(weapon_filenames) == 0):
            return "No weapons detected", 400

        encoded_images = []
        for weapon_file in weapon_filenames:
            with open(weapon_file, 'rb') as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                encoded_images.append(img_b64)

        # Pack into messagepack format
        response = {
            'weapon_images': encoded_images,
            "loc": test_loc,
        }
        packed_response = msgpack.packb(response)

        return packed_response, 200, {'Content-Type': 'application/x-msgpack'}
    else:
        return "Failed to take screenshot", 500

@app.route('/screenshot_full', methods=['GET'])
def screenshot_full():
    print("Taking screenshot")
    # Take screenshot using camera
    filename = camera.take_screenshot()
    
    if filename and os.path.exists(filename):
        remove_old_screenshots()
        # Convert the image to base64
        with open(filename, 'rb') as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

        # Pack into messagepack format
        response = {
            'image': img_b64,
            "loc": test_loc,
        }
        packed_response = msgpack.packb(response)

        return packed_response, 200, {'Content-Type': 'application/x-msgpack'}
    else:
        return "Failed to take screenshot", 500

def remove_old_screenshots():
    # Get all png files in screenshots dir sorted by creation time
    screenshot_files = [f for f in os.listdir(shared_screenshots_dir) if f.endswith('.png')]
    screenshot_files.sort(key=lambda x: os.path.getctime(os.path.join(shared_screenshots_dir, x)))
    
    # Remove all but the 5 most recent png files
    if len(screenshot_files) > 5:
        for file in screenshot_files[:-5]:
            os.remove(os.path.join(shared_screenshots_dir, file))
            print(f"Removed {file}")

    # Get all folders in people dir sorted by creation time
    people_folders = [f for f in os.listdir(shared_people_dir) if os.path.isdir(os.path.join(shared_people_dir, f))]
    people_folders.sort(key=lambda x: os.path.getctime(os.path.join(shared_people_dir, x)))

    # Remove all but the 5 most recent folders
    if len(people_folders) > 5:
        for folder in people_folders[:-5]:
            folder_path = os.path.join(shared_people_dir, folder)
            # Remove all files in the folder first
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            # Remove the empty folder
            os.rmdir(folder_path)
            print(f"Removed folder {folder}")

    # Get all folders in weapon dir sorted by creation time
    weapon_folders = [f for f in os.listdir(shared_weapon_dir) if os.path.isdir(os.path.join(shared_weapon_dir, f))]
    weapon_folders.sort(key=lambda x: os.path.getctime(os.path.join(shared_weapon_dir, x)))

    # Remove all but the 5 most recent folders
    if len(weapon_folders) > 5:
        for folder in weapon_folders[:-5]:
            folder_path = os.path.join(shared_weapon_dir, folder)
            # Remove all files in the folder first
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            # Remove the empty folder
            os.rmdir(folder_path)
            print(f"Removed folder {folder}")

if __name__ == '__main__':
    app.run(host='localhost', port=8100)
