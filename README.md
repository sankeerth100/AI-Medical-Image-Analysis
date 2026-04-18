# 🏥 AI Medical Image Analysis System (Pneumonia Detection)

## 🚀 Overview
This project is a deep learning–based medical imaging system that detects pneumonia from chest X-ray images and provides explainable AI visualizations using Grad-CAM.

The system is built as an interactive Streamlit web application to simulate a real-world AI diagnostic assistant.

---

## 🎯 Problem Statement
Diagnosing pneumonia from chest X-rays manually is:
- Time-consuming
- Subject to human error
- Dependent on expert availability

This project aims to assist medical professionals using AI for faster and more interpretable predictions.

---

## 🧠 Key Features

### 🔬 Disease Detection
- Classifies chest X-ray images as:
  - Pneumonia
  - Normal

### 🔥 Explainable AI (Grad-CAM)
- Highlights regions in the image influencing model prediction
- Improves transparency and trust in AI decisions

### 📊 Visual Analytics
- Confidence score display
- Bar chart (Normal vs Pneumonia probability)
- Pie chart breakdown

### 📄 Report Generation
- Downloadable diagnosis summary report

### 🖥 Interactive UI
- Streamlit-based web dashboard
- Supports multiple image uploads

---

## 🏗 System Pipeline
Input Image (Chest X-ray)
↓
Preprocessing (Resize + Normalization)
↓
Deep Learning Model (MobileNetV2 / CNN)
↓
Prediction (Binary Classification)
↓
Grad-CAM Heatmap Generation
↓
Visualization on Streamlit Dashboard

## 🧪 Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit
- Grad-CAM (Explainable AI)

---

## 🧠 Model Details

- Architecture: MobileNetV2 (Transfer Learning)
- Input Size: 224 × 224
- Output: Binary Classification (Sigmoid)
- Task: Pneumonia Detection
- Loss Function: Binary Crossentropy

---

## 🔥 Explainable AI (Grad-CAM)

Grad-CAM is used to visualize **which parts of the X-ray image influenced the model’s prediction**, such as:
- Lung opacity regions
- Infection areas
- Abnormal texture patterns

This improves interpretability for medical use cases.

---

## 📁 Project Structure
AI-Medical-Image-Analysis/
│
├── app/
│ └── app.py
│
├── models/ (not uploaded to GitHub)
├── requirements.txt
├── README.md
## ⚠️ Deployment Note

- Model files are not stored in GitHub due to size limitations
Hugging Face Model Link:
https://huggingface.co/sankeerth100/pneumonia-detector-model/tree/main
- Hosted externally (Hugging Face / Drive)
- Loaded dynamically during runtime

---

## 📦 Installation

```bash
git clone https://github.com/sankeerth100/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis

pip install -r requirements.txt

streamlit run app/app.py
