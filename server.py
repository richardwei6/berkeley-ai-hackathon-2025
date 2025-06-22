import os
from camera_input.camera_input import CameraInput
from person_detection.people_cropper import PeopleCropper
from flask import Flask, send_file
import threading

shared_screenshots_dir = "../shared/screenshots"

# Test camera input
camera = CameraInput(output_dir=shared_screenshots_dir)

# People detection
people_cropper = PeopleCropper()

app = Flask(__name__)

@app.route('/screenshot', methods=['GET'])
def take_screenshot():
    # Take screenshot using camera
    filename = camera.take_screenshot()
    
    if filename and os.path.exists(filename):
        # Process the image to detect people
        people_cropper.detect(shared_screenshots_dir)
        
        # Return the screenshot file
        return send_file(filename, mimetype='image/jpeg')
    else:
        return "Failed to take screenshot", 500

if __name__ == '__main__':
    app.run(host='localhost', port=8100)
    camera.start_feed()
