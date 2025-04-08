import os
import tempfile
import streamlit as st
import gdown
import cv2
from ultralytics import YOLO
from collections import Counter

# ── CONFIG ──────────────────────────────────────────────────────────────────────

MODEL_NAME = "best.pt"
# Google Drive direct-download URL for your .pt file:
# e.g. if your share link is
#   https://drive.google.com/file/d/1AbCDeFGhiJKlmnopQrst/view?usp=sharing
# then FILE_ID = "1AbCDeFGhiJKlmnopQrst"
DRIVE_FILE_ID = "https://drive.google.com/file/d/1VMvqoEv84Blm8VFw8SyeyFntRylt-gqA/view?usp=sharing"
DRIVE_URL = f"https://drive.google.com/drive/folders/1URcMF_5JKIt3rqowc29V5O4Ia8IMWZ_-?usp=drive_link{DRIVE_FILE_ID}"

CLASS_NAMES = ['Normal', 'Violence', 'Weaponized']
FRAME_INTERVAL = 10  # sample every Nth frame

# ── MODEL DOWNLOAD ──────────────────────────────────────────────────────────────

if not os.path.isfile(MODEL_NAME):
    st.info("⏬ Downloading model weights…")
    # download into working dir
    gdown.download(DRIVE_URL, MODEL_NAME, quiet=False)
    st.success("✅ Model downloaded.")

# ── LOAD MODEL ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_NAME)

# ── STREAMLIT UI ────────────────────────────────────────────────────────────────

st.title("🎥 Video Violence Classifier")
st.write("Upload an `.avi` video and get a classification result (Normal, Violence, Weaponized).")

uploaded_file = st.file_uploader("Choose an .avi video file", type=["avi"])
if not uploaded_file:
    st.warning("Please upload an .avi video to begin.")
    st.stop()

# preview
st.video(uploaded_file)

# save to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

# process
st.info("⏳ Processing video frames…")
cap = cv2.VideoCapture(tmp_path)
preds = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_INTERVAL == 0:
        # YOLO predict returns a Results object
        results = model.predict(frame, verbose=False)
        # get top‑1 class index
        idx = int(results[0].probs.top1)
        preds.append(CLASS_NAMES[idx])

    frame_idx += 1

cap.release()
os.remove(tmp_path)

# aggregate and display
if preds:
    final = Counter(preds).most_common(1)[0][0]
    st.success(f"🧠 **Final Prediction:** {final}")
else:
    st.error("❌ No frames were processed. Try a different video.")
