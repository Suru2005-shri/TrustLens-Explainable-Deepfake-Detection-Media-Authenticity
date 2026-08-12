"""
TrustLens - Streamlit demo app
Run: streamlit run app.py
Needs: pip install -r requirements.txt
"""
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile

from forensics import error_level_analysis, frequency_domain_analysis, temporal_consistency_analysis
from classifier import classify_image
from fusion import fuse_signals

st.set_page_config(page_title="TrustLens", page_icon="🔍", layout="centered")
st.title("🔍 TrustLens — Deepfake & Media Authenticity Verification")
st.caption("Multi-signal forensic analysis: pretrained classifier + ELA + frequency-domain + temporal consistency")

tab_img, tab_vid = st.tabs(["Image", "Video"])

with tab_img:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running 3-signal forensic analysis..."):
            clf_result = classify_image(image)
            ela_result = error_level_analysis(image)
            freq_result = frequency_domain_analysis(image)
            ela_display = ela_result.pop("ela_image")

            fused = fuse_signals([clf_result, ela_result, freq_result])

        st.subheader(f"Verdict: {fused['final_verdict']}")
        if fused["confidence_authentic_pct"] is not None:
            st.metric("Authenticity confidence", f"{fused['confidence_authentic_pct']}%")

        with st.expander("Signal breakdown (explainability)"):
            st.json(fused["signal_breakdown"])
            st.image(ela_display, caption="ELA heatmap — bright patches = recompression mismatch")

with tab_vid:
    uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="vid")
    if uploaded_vid:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        st.video(uploaded_vid)

        with st.spinner("Extracting frames + running forensic analysis..."):
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_every = max(1, frame_count // 15)  # sample ~15 frames
            i = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if i % sample_every == 0:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                i += 1
            cap.release()

            if not frames:
                st.error("Could not extract frames from video.")
            else:
                mid_frame = frames[len(frames) // 2]
                clf_result = classify_image(mid_frame)
                ela_result = error_level_analysis(mid_frame)
                freq_result = frequency_domain_analysis(mid_frame)
                ela_result.pop("ela_image")
                temporal_result = temporal_consistency_analysis(frames)

                fused = fuse_signals([clf_result, ela_result, freq_result, temporal_result])

        st.subheader(f"Verdict: {fused['final_verdict']}")
        if fused["confidence_authentic_pct"] is not None:
            st.metric("Authenticity confidence", f"{fused['confidence_authentic_pct']}%")
        with st.expander("Signal breakdown"):
            st.json(fused["signal_breakdown"])

st.divider()
st.caption(
    "⚠️ Limitation: detection accuracy drops for generation architectures not "
    "represented in training/validation data (FaceForensics++, Celeb-DF, DFDC). "
    "No detector generalizes perfectly to novel generators — treat scores as "
    "decision support, not ground truth."
)
