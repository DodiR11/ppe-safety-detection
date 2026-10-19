import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

from src.pipeline import extract_detections, detect_violations, annotate_image
from src.video import process_video

st.set_page_config(page_title="PPE Safety Detection", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("model/best.pt")


model = load_model()
print(model.names)

st.title("PPE   Safety Detection System")
st.write("Upload an image or video to detect safety equipment and violations.")

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)

if uploaded_file:

    if "image" in uploaded_file.type:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        results = model.predict(source=image_np, conf=conf, verbose=False)

        for r in results:
            detections =extract_detections(r, model)

            violations = detect_violations(detections)
            annotated_img  = annotate_image(r, violations)
            st.write(detections)

            col1, col2 =  st.columns(2)

            with col1:

                st.subheader("Original Image")
                st.image(image_np, use_container_width=True)

            with col2:
                st.subheader("Detection Result")
                st.image(annotated_img, use_container_width=True)

            st.subheader("Sumary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Detections", len(detections))
            c2.metric("Violations", len(violations))
            c3.metric("Status ", "Violation" if violations else "Safe")

            if violations:
                st.error(" violations detected!")
                for v in violations:
                    st.warning(v)
            else:
                st.success(" All workers are compliant")

    elif "video" in uploaded_file.type:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_file.read())

            input_path = temp_input.name

        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "output.mp4")

        st.info("Processing video... please wait")

        stats = process_video(
            model,
            input_path,
            output_path,
            conf=conf
        )

        st.subheader("Processed Video")

        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()

        st.video(video_bytes)

        st.subheader("Video Summary")

        st.write(f"Total frames: {stats['total_frames']}")
        st.write(f"Frames with violations: {stats['violation_frames']}")

        with open(output_path, "rb") as video_file:
            st.download_button(
                "Download Processed Video",
                data=video_file,

                file_name=  "processed_video.mp4",
                mime="video/mp4"
            )

        os.remove(input_path)