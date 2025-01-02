import cv2
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from collections import deque
import uuid

# Class to track vehicles and count them when they cross a specified line
class VehicleTracker:

    def __init__(self, line_position):
        self.vehicles = {}  # Dictionary to store vehicle data
        self.line_position = line_position  # Vertical line position to count vehicles
        self.counted_vehicles = set()  # Set of vehicles that have already been counted
        self.count = 0  # Total vehicle count

    def update(self, detections):
        current_vehicles = {}  # Dictionary to store current frame vehicles

        for bbox in detections:
            matched = False  # Flag to check if a detection matches an existing vehicle

            for vehicle_id, vehicle_data in self.vehicles.items():
                # Check overlap with existing vehicles
                if self._calculate_overlap(bbox, vehicle_data["bbox"]) > 0.3:
                    center_y = bbox[1] + bbox[3] // 2  # Calculate center of the bounding box
                    if vehicle_id not in self.counted_vehicles and center_y > self.line_position:
                        self.counted_vehicles.add(vehicle_id)  # Mark vehicle as counted
                    current_vehicles[vehicle_id] = {"bbox": bbox}  # Update vehicle data
                    matched = True
                    break

            if not matched:
                # Assign a unique ID to a new vehicle
                vehicle_id = str(uuid.uuid4())
                current_vehicles[vehicle_id] = {"bbox": bbox}
                self.count += 1  # Increment total vehicle count

        self.vehicles = current_vehicles  # Update tracked vehicles
        return self.count

    # Calculate overlap ratio between two bounding boxes
    def _calculate_overlap(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        return intersection / min(area1, area2)

# Function to detect cars and update the tracker
def detect_cars(frame, cascade_classifier, tracker, min_size=(50, 50)):
    # Apply Gaussian blur to reduce noise
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)  # Remove small noise
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # Fill gaps

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect cars using Haar Cascade Classifier
    cars = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)

    # Filter detections based on minimum size
    valid_detections = [(x, y, w, h) for (x, y, w, h) in cars if w >= min_size[0] and h >= min_size[1]]
    # Update tracker with valid detections
    current_count = tracker.update(valid_detections)

    # Draw bounding boxes and labels on the frame
    for vehicle_id, vehicle_data in tracker.vehicles.items():
        x, y, w, h = vehicle_data["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        label = "Counted" if vehicle_id in tracker.counted_vehicles else "Tracking"
        cv2.putText(frame, f"{label} {vehicle_id[:4]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame, current_count

# Streamlit application starts here
st.title('Side Profile Car Detection Using Haar Cascade')

st.markdown("""
This app allows you to upload a video and will count the number of cars passing a specified line in the video using **Haar Cascade Classifier** for side-profile car detection. The processed video with bounding boxes will be exported.
""")

# File uploader for video input
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

# Load Haar Cascade Classifier for side-profile car detection
cascade_path = 'sideview_cascade_classifier.xml'
car_cascade = cv2.CascadeClassifier(cascade_path)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Open video file
    cap = cv2.VideoCapture(temp_file.name)
    if not cap.isOpened():
        st.error("Error: Couldn't open video file.")
    else:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize tracker and set parameters
        line_position = 350  # Position of counting line
        min_size = (100, 100)  # Minimum detection size
        tracker = VehicleTracker(line_position=line_position)

        # Prepare output video
        out_video = NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
        out = cv2.VideoWriter(out_video.name, fourcc, fps, (frame_width, frame_height))

        # Process video frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect cars and update frame
            frame, current_count = detect_cars(frame, car_cascade, tracker, min_size)
            
            # Draw counting line and total car count
            cv2.line(frame, (0, line_position), (frame_width, line_position), (0, 0, 255), 2)
            cv2.putText(frame, f"Total Cars: {current_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Write processed frame to output video
            out.write(frame)

        # Release resources
        cap.release()
        out.release()

        # Display total car count
        st.success(f"Total Cars Counted: {current_count}")
        st.markdown("### Download the Processed Video")

        # Provide download link for processed video
        with open(out_video.name, 'rb') as video_file:
            video_bytes = video_file.read()
            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )