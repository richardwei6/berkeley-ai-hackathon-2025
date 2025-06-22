import os
from camera_input.camera_input import CameraInput
from person_detection.people_cropper import PeopleCropper
from flask import Flask, send_file
import base64
import msgpack

shared_screenshots_dir = "./shared/screenshots"
shared_people_dir = "./shared/people"

if not os.path.exists(shared_screenshots_dir):
    os.makedirs(shared_screenshots_dir)

# Test camera input
camera = CameraInput(output_dir=shared_screenshots_dir)

# People detection
people_cropper = PeopleCropper(output_dir=shared_people_dir)

app = Flask(__name__)

@app.route('/screenshot_people', methods=['GET'])
def take_screenshot():
    print("Taking screenshot")
    # Take screenshot using camera
    filename = camera.take_screenshot()
    
    if filename and os.path.exists(filename):
        # Process the image to detect people
        people_filenames = people_cropper.detect(filename)
        # Convert each cropped image to base64

        encoded_images = []
        for person_file in people_filenames:
            with open(person_file, 'rb') as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                encoded_images.append(img_b64)

        # Pack into messagepack format
        response = {
            'images': encoded_images
        }
        packed_response = msgpack.packb(response)

        return packed_response, 200, {'Content-Type': 'application/x-msgpack'}
    else:
        return "Failed to take screenshot", 500

if __name__ == '__main__':
    app.run(host='localhost', port=8100)
