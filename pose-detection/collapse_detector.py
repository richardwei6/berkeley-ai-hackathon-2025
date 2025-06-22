import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import time

class CollapseDetector:
    """Class for real-time pose-based collapse detection"""
    
    def __init__(self, model_path='pose-detection/collapseModel.h5', 
                 training_data_path='pose-detection/pose_labels.csv'):
        """
        Initialize the collapse detector
        
        Args:
            model_path: Path to the trained collapse model
            training_data_path: Path to training data for scaler fitting
        """
        print("Initializing CollapseDetector...")
        
        # Load MoveNet model
        print("Loading MoveNet model...")
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        self.movenet = self.model.signatures['serving_default']
        
        # Load the trained collapse model
        print("Loading collapse model...")
        self.collapse_model = tf.keras.models.load_model(model_path)
        
        # Load training data and fit scaler
        print("Loading training data for scaler...")
        df = pd.read_csv(training_data_path)
        X = df.drop("label", axis=1).values
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        # Define skeleton connections
        self.SKELETON = [
            (0, 1), (1, 3), (0, 2), (2, 4),       # Head -> Shoulders -> Arms
            (5, 7), (7, 9), (6, 8), (8, 10),      # Arms
            (5, 6), (5, 11), (6, 12),             # Torso
            (11, 12), (11, 13), (13, 15),         # Legs
            (12, 14), (14, 16)
        ]
        
        print("CollapseDetector initialized successfully!")
    
    def detect_pose(self, frame):
        """
        Detect pose keypoints using MoveNet
        
        Args:
            frame: RGB image frame
            
        Returns:
            keypoints: Array of shape (17, 2) with normalized coordinates
        """
        input_image = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 256, 256)
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = self.movenet(input_image)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :2]  # shape: (17, 2)
        return keypoints
    
    def preprocess_keypoints(self, keypoints):
        """
        Flatten keypoints for model input
        
        Args:
            keypoints: Array of shape (17, 2)
            
        Returns:
            flattened: Array of shape (34,)
        """
        return keypoints.flatten()  # shape: (34,)
    
    def predict_collapse(self, keypoints):
        """
        Predict collapse score from pose keypoints
        
        Args:
            keypoints: Array of shape (17, 2) with pose keypoints
            
        Returns:
            collapse_score: Float between 0 and 1 indicating collapse risk
        """
        # Preprocess keypoints
        input_vec = self.preprocess_keypoints(keypoints).reshape(1, -1)
        input_vec = self.scaler.transform(input_vec)
        
        # Predict collapse score
        collapse_score = self.collapse_model.predict(input_vec, verbose=0)[0][0]
        return collapse_score
    
    def draw_pose(self, frame, keypoints):
        """
        Draw pose skeleton on frame
        
        Args:
            frame: BGR image frame to draw on
            keypoints: Array of shape (17, 2) with normalized coordinates
        """
        h, w, _ = frame.shape
        # Convert normalized coordinates to pixel
        keypoints_px = [(int(x * w), int(y * h)) for y, x in keypoints]

        # Draw bones
        for start, end in self.SKELETON:
            x1, y1 = keypoints_px[start]
            x2, y2 = keypoints_px[end]
            cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

        # Draw joints
        for (x, y) in keypoints_px:
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    def get_risk_status(self, collapse_score):
        """
        Get risk status and color based on collapse score
        
        Args:
            collapse_score: Float between 0 and 1
            
        Returns:
            tuple: (color, status_text)
        """
        if collapse_score > 0.7:
            color = (0, 0, 255)  # Red for high collapse risk
            status = "HIGH RISK"
        else:
            color = (0, 255, 0)  # Green for low risk
            status = "LOW RISK"
        return color, status
    
    def process_frame(self, frame):
        """
        Process a single frame for pose detection and collapse prediction
        
        Args:
            frame: BGR image frame from camera
            
        Returns:
            tuple: (processed_frame, collapse_score, success)
        """
        # Convert BGR to RGB for MoveNet
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect pose
            keypoints = self.detect_pose(rgb_frame)
            
            # Predict collapse score
            collapse_score = self.predict_collapse(keypoints)
            
            # Draw pose skeleton
            self.draw_pose(frame, keypoints)
            
            # Get risk status
            color, status = self.get_risk_status(collapse_score)
            
            # Display collapse score and status
            cv2.putText(frame, f'Collapse Score: {collapse_score:.3f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Status: {status}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame, collapse_score, True
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            cv2.putText(frame, 'Error detecting pose', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, None, False
    
    def run_realtime_detection(self, camera_index=0):
        """
        Run real-time collapse detection using webcam
        
        Args:
            camera_index: Camera device index (default: 0)
        """
        print("Starting real-time collapse detection...")
        print("Press 'q' to quit.")
        
        cap = cv2.VideoCapture(camera_index)
        
        # Allow camera to warm up
        time.sleep(2)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            processed_frame, collapse_score, success = self.process_frame(frame)
            
            if success:
                print(f"Collapse Score: {collapse_score:.3f}")
            
            cv2.imshow('Collapse Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


def main():
    """Main function to demonstrate the CollapseDetector class"""
    detector = CollapseDetector()
    detector.run_realtime_detection()


if __name__ == "__main__":
    main() 