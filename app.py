import streamlit as st
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# Paths
DATA_DIR = "E:\\Intern\\NUT and Bolts\\UPLOAD"
MODEL_PATH = "E:\\Intern\\NUT and Bolts\\MODEL\\yolov8m_nut_bolt.pt"
DATA_YAML = "E:\\Intern\\NUT and Bolts\\data.yaml"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load trained model
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    st.warning("No trained model found. Please train the model first.")
    model = None

# Function to detect multiple objects properly
def detect_objects(image, conf_threshold=0.4, iou_threshold=0.5):
    if model is None:
        st.error("Model is not loaded. Please train YOLOv8m first.")
        return [], np.array(image)

    image_cv = np.array(image.convert("RGB"))  # Convert PIL image to OpenCV format
    results = model(image_cv, conf=conf_threshold, iou=iou_threshold)  # Adjust detection thresholds

    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()  # Get confidence score
            width = x2 - x1
            height = y2 - y1
            area = width * height  # Calculate area of detected object

            detected_objects.append((label, x1, y1, x2, y2, width, height, area, confidence))

            # Draw bounding box and label each object separately
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return detected_objects, image_cv

# Streamlit UI
st.title("🔩 Nut & Bolt Detector with Measurement")

st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.5)

uploaded_file = st.file_uploader("Upload an image for detection")

if uploaded_file:
    image = Image.open(uploaded_file)
    detected_objects, output_image = detect_objects(image, conf_threshold, iou_threshold)

    st.image(output_image, caption="Detected Objects", use_container_width=True)

    for obj in detected_objects:
        label, x1, y1, x2, y2, width, height, area, confidence = obj
        st.success(f"Detected: {label} - Width: {width}px, Height: {height}px, Area: {area} px² - Confidence: {confidence:.2f}")
