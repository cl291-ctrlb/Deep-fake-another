import streamlit as st
import cv2
import numpy as np
from utils import predict

st.title("Deepfake Image Detection Demo")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    face, fake_prob = predict(image)

    if face is None:
        st.error("No face detected in the image.")
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        st.image(face, caption="Extracted Face", width=250)
        st.success(f"Fake Probability: {fake_prob}%")
