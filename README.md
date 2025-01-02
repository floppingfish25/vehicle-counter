# Side Profile Car Detection Using Haar Cascade

This project provides an application that detects and counts the number of cars passing a specified line in a video using **Haar Cascade Classifier** for side-profile car detection. The processed video includes bounding boxes around detected vehicles and the total car count is displayed.

## Approach

1. **Vehicle Detection**: The app uses Haar Cascade Classifier trained for side-profile car detection. It detects cars in each frame of the uploaded video.

2. **Vehicle Tracking and Counting**: Once vehicles are detected, the app tracks them across frames. When a vehicle crosses a predefined line, it is counted. The vehicle IDs and their bounding boxes are maintained to prevent double-counting.

3. **Post-Processing**: The processed video includes bounding boxes around detected vehicles with a label showing whether the vehicle has been counted or is still being tracked.

4. **Exporting the Video**: The final processed video is available for download with the total car count displayed on it.

## Tools Used

- **OpenCV**: Used for video processing, car detection, and drawing bounding boxes.
- **Streamlit**: Used for building the interactive web interface for video upload and processing.
- **Haar Cascade Classifier**: Utilized for side-profile vehicle detection.
- **Python**: The project is implemented in Python with libraries like OpenCV and Streamlit.

## Steps to Reproduce

### 1. Install Streamlit

```bash
pip install streamlit

### 1. Install Streamlit
