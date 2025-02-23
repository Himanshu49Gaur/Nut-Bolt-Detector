# Nut-Bolt-Detector

Nut & Bolt Detector 🛠️

This project is a nut and bolt detection system using YOLOv8 and Streamlit. It allows users to:

Detect nuts and bolts in uploaded images

Train a custom YOLO model with new images and labels

🚀 Features

Real-time detection of nuts and bolts

Custom model training with user-uploaded datasets

Interactive UI built with Streamlit

📌 Installation

1️⃣ Clone the repository:
```
git clone https://github.com/Himanshu49Gaur/nut-bolt-detector.git
cd nut-bolt-detector
```

2️⃣ Install dependencies:

``pip install -r requirements.txt``

3️⃣ Run the application:

`streamlit run app.py`

📸 Usage

Detect Objects: Upload an image to detect nuts and bolts.

Train New Model: Upload labeled datasets to train a new YOLO model.

📂 Dataset Structure

 ```datasets/
 ├── images/
 │   ├── train/
 │   ├── val/
 ├── labels/
 │   ├── train/
 │   ├── val/
 ├── data.yaml
```
🔧 Dependencies

`Python 3.x`

`Streamlit`

`OpenCV`

`NumPy`

`Pillow`

Ultralytics (YOLOv8)

📝 License

`MIT License`
