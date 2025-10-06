# app.py
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import numpy as np

# ===== Class names (fix the order later if needed) =====
class_names = ["idle", "gerakan kepala-bawah", "gerakan kepala-leher", "gerakan tangan"]

# ===== Load Ultralytics YOLO model =====

model = YOLO("models/best.pt")  # path to your trained model

# ===== Draw boxes =====
def draw_boxes(frame, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

# ===== Streamlit UI =====
st.title("PACES (Paper-based Anti Cheating Examination System)")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict using YOLOv8
        results = model(frame)  # frame is HxWxC BGR numpy array
        for r in results:
            frame = draw_boxes(frame, r.boxes, class_names)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
