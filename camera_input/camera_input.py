import cv2
import os
from datetime import datetime, timedelta

class CameraInput:
    def __init__(self, camera_index=0, output_dir='screenshots'):
        self.cap = cv2.VideoCapture(camera_index)
        self.output_dir = output_dir
        self.running = False

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def start_feed(self):
        self.running = True
        print("Starting webcam feed. Press 'q' to quit.")
        while self.running:
            tries = 0
            while tries < 3:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    tries += 1
                    continue
                break
            
            if not ret:
                print("Failed to grab frame after 3 tries.")
                exit()

            cv2.imshow('Webcam Feed', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                self.running = False
            elif key & 0xFF == ord('k'):
                self.take_screenshot()

        self.cap.release()
        cv2.destroyAllWindows()


    def take_screenshot(self):
        if not self.cap.isOpened():
            print("Webcam is not open.")
            return

        tries = 0
        while tries < 3:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame - %i", tries)
                tries += 1
                continue
            break

        if ret:
            timestamp = datetime.now().strftime("%m:%d~%H:%M:%S:%f")
            filename = os.path.join(self.output_dir, f"screenshot {timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
            return filename
        else:
            print("!! Failed to take screenshot after 3 tries !!")    