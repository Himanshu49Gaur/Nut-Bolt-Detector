import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import shutil
import random

# Define paths
dataset_path = "datasets"
image_train_path = os.path.join(dataset_path, "images", "train")
image_val_path = os.path.join(dataset_path, "images", "val")
label_train_path = os.path.join(dataset_path, "labels", "train")
label_val_path = os.path.join(dataset_path, "labels", "val")
data_yaml_path = os.path.join(dataset_path, "data.yaml")
model_path = "trained_model.pt"

# Ensure dataset structure exists
os.makedirs(image_train_path, exist_ok=True)
os.makedirs(image_val_path, exist_ok=True)
os.makedirs(label_train_path, exist_ok=True)
os.makedirs(label_val_path, exist_ok=True)

# Load YOLOv8m model
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = YOLO("yolov8m.pt")  # Default model

# Function to detect nuts and bolts
def detect_objects(image):
    img = np.array(image)
    results = model(img)
    annotated_img = img.copy()
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = "Nut" if result.boxes.cls[0] == 0 else "Bolt"
            cv2.putText(annotated_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_img, results

# Function to create data.yaml dynamically
def create_data_yaml():
    data_yaml_content = f"""
path: {os.path.abspath(dataset_path)}
train: {os.path.abspath(image_train_path)}
val: {os.path.abspath(image_val_path)}

nc: 2
names: ["Nut", "Bolt"]
"""
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    st.success("✅ data.yaml created successfully!")

# Function to train the model
def train_model():
    if not os.path.exists(data_yaml_path):
        create_data_yaml()
    st.write("🔄 Training the model on uploaded images...")
    model.train(data=data_yaml_path, epochs=5, imgsz=640, save=True, project="runs/train")
    st.success("✅ Training complete! Model saved.")
    os.rename("runs/train/exp/weights/best.pt", model_path)

# Streamlit UI
st.title("🔩 Nut & Bolt Detector 🛠️")
st.sidebar.header("Options")
option = st.sidebar.radio("Choose Action:", ["Detect", "Train New Model"])

if option == "Detect":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Detect Objects"):
            detected_image, results = detect_objects(image)
            st.image(detected_image, caption="Detection Result", use_container_width=True)
            
            st.subheader("🔽 Detection Output:")
            for result in results:
                for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                    label = "Nut" if cls == 0 else "Bolt"
                    x1, y1, x2, y2 = map(int, box[:4])
                    st.write(f"**{label}:** Coordinates ({x1}, {y1}) to ({x2}, {y2})")

elif option == "Train New Model":
    st.write("📤 Upload images and labels for training.")
    image_files = st.file_uploader("Upload images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    label_files = st.file_uploader("Upload corresponding YOLO label files...", type=["txt"], accept_multiple_files=True)
    
    if image_files and label_files:
        for img in image_files:
            img_folder = image_train_path if random.random() > 0.2 else image_val_path
            img_path = os.path.join(img_folder, img.name)
            with open(img_path, "wb") as f:
                f.write(img.read())
        
        for label in label_files:
            label_folder = label_train_path if random.random() > 0.2 else label_val_path
            label_path_file = os.path.join(label_folder, label.name)
            with open(label_path_file, "wb") as f:
                f.write(label.read())
        
        st.success("✅ Images and labels uploaded successfully.")
    
    if st.button("📈 Train Model"):
        create_data_yaml()
        train_model()
