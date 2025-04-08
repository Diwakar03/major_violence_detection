import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
from collections import Counter
import os

# Load your trained model
model_path = r'C:\Users\diwak\Downloads\yolo\best.pt'  # update if needed
model = YOLO(model_path)
class_names = ['Normal', 'Violence', 'Weaponized']

st.title("🎥 Video Violence Classifier")
st.write("Upload a video (.avi) and get a classification result (Normal, Violence, Weaponized).")

uploaded_file = st.file_uploader("Upload an .avi video", type=["avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_vid:
        temp_vid.write(uploaded_file.read())
        temp_vid_path = temp_vid.name

    st.write("⏳ Processing video...")

    cap = cv2.VideoCapture(temp_vid_path)
    frame_interval = 10
    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            results = model.predict(frame, verbose=False)
            predicted_class_index = int(results[0].probs.top1)
            predictions.append(class_names[predicted_class_index])

        frame_count += 1

    cap.release()
    os.remove(temp_vid_path)

    if predictions:
        final_prediction = Counter(predictions).most_common(1)[0][0]
        st.success(f"🧠 **Final Prediction:** {final_prediction}")
    else:
        st.error("No frames processed. Try a different video.")

