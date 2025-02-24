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

def train_model():
    st.write("Training YOLOv8m on Nut & Bolt dataset...")
    model = YOLO("yolov8m.pt")
    model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=8, device='cuda' if torch.cuda.is_available() else 'cpu')
    os.rename("runs/detect/train/weights/best.pt", MODEL_PATH)
    st.success("Model trained and saved!")

def detect_objects(image):
    if model is None:
        st.error("Model is not loaded. Please train YOLOv8m first.")
        return [], np.array(image)
    
    image_cv = np.array(image)
    results = model(image_cv)[0]
    
    detected_objects = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        label = results.names[int(box.cls[0])]
        confidence = box.conf[0].item()  # Confidence score
        width = x2 - x1
        height = y2 - y1
        area = width * height  # Calculate area of detected object
        
        if confidence > 0.5:  # Filter out low-confidence detections
            detected_objects.append((label, x1, y1, x2, y2, width, height, area, confidence))
            
            # Draw bounding box and label each object separately
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return detected_objects, image_cv

# Streamlit UI
st.title("🔩 Nut & Bolt Detector with Measurement")

st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ["Train Model", "Detect Objects"])

if mode == "Train Model":
    uploaded_files = st.file_uploader("Upload training images", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(DATA_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success("Images uploaded successfully!")
    
    if st.button("Train YOLOv8m Model"):
        train_model()

elif mode == "Detect Objects":
    uploaded_file = st.file_uploader("Upload an image for detection")
    if uploaded_file:
        image = Image.open(uploaded_file)
        detected_objects, output_image = detect_objects(image)
        
        st.image(output_image, caption="Detected Objects", use_container_width=True)
        
        for obj in detected_objects:
            label, x1, y1, x2, y2, width, height, area, confidence = obj
            st.success(f"Detected: {label} - Confidence: {confidence:.2f}, Width: {width}px, Height: {height}px, Area: {area} pixels²")
