import streamlit as st
st.set_page_config(page_title="Gridlock AI", layout="wide", page_icon="🚦")

import os
from datetime import datetime
import logging
import tempfile
import io

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_HAS_CV2 = False
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None

_HAS_PIL = False
try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    Image = None

_HAS_PANDAS = False
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None

_HAS_NUMPY = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None

pipeline = None
try:
    from app.pipeline_runner import get_pipeline as _get_pipeline
    pipeline = _get_pipeline()
except Exception as e:
    logger.warning("Pipeline not available: %s", e)

# Pre-load pipeline silently at startup
if pipeline is not None and not pipeline._loaded:
    pipeline.load()

def safe_call(method, *args, **kwargs):
    if pipeline is None:
        return None
    try:
        return getattr(pipeline, method)(*args, **kwargs)
    except Exception as e:
        logger.warning("Pipeline.%s failed: %s", method, e)
        return None

st.sidebar.title("Gridlock AI")
selected_cam = st.sidebar.selectbox("Camera", ["All", "CAM_001", "CAM_002"])
auto_refresh = st.sidebar.checkbox("Auto-Refresh", value=False)

st.title("Traffic Enforcement & Risk Intelligence Platform")
st.markdown("Real-time monitoring and analytics.")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Live Feed", "Risk Map", "Hotspot Analysis", "Predicted Violations",
    "Repeat Offenders", "Enforcement Plan", "Search & Reports"
])

with t1:
    st.subheader("Live Feed")
    mode = st.radio("Input Mode", ["Simulated Feed", "Upload Image", "Upload Video"], horizontal=True)

    if mode == "Upload Image":
        st.markdown("**Enter image URL to analyze**")
        img_url = st.text_input("Image URL", placeholder="https://example.com/traffic-image.jpg")
        analyze = st.button("Analyze")
        if analyze and img_url:
            try:
                import requests
                resp = requests.get(img_url, timeout=15)
                resp.raise_for_status()
                img_bytes = resp.content
                if Image:
                    st.image(Image.open(io.BytesIO(img_bytes)), caption="Input Image", use_container_width=True)
                with st.spinner("Running detection..."):
                    result = safe_call("process_image", img_bytes, camera_id="CAM_001")
                    if result is None:
                        st.warning("Pipeline not available.")
                    elif result.get("error"):
                        st.warning(f"Result: {result['error']}")
                    if result and result.get("events"):
                        st.success(f"Detected {len(result['events'])} violation(s)")
                        if _HAS_PANDAS:
                            st.dataframe(pd.DataFrame([
                                {"Type": e.get("violation_type", "N/A"), "Confidence": f"{e.get('confidence', 0):.0%}", "Plate": e.get("plate_text", "N/A")}
                                for e in result["events"]
                            ]), hide_index=True, use_container_width=True)
                    else:
                        st.info("No violations detected.")
            except requests.RequestException:
                st.error("Could not download image. Check URL.")
            except Exception as e:
                st.error(f"Error: {e}")

    elif mode == "Upload Video":
        if not _HAS_CV2:
            st.warning("OpenCV not available — video upload disabled.")
        else:
            uploaded_vid = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_vid is not None:
                try:
                    uploaded_vid.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(uploaded_vid.read())
                        tmp_path = tmp.name
                    cap = cv2.VideoCapture(tmp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    st.info(f"Video: {total_frames} frames, {fps:.1f} fps")
                    sample_every = max(1, total_frames // 10)
                    all_violations = []
                    frame_idx = 0
                    progress_bar = st.progress(0)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_idx % sample_every == 0:
                            _, buffer = cv2.imencode(".jpg", frame)
                            result = safe_call("process_image", buffer.tobytes(), camera_id="CAM_001")
                            if result:
                                for ev in result.get("events", []):
                                    all_violations.append(ev)
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                        frame_idx += 1
                    cap.release()
                    os.unlink(tmp_path)
                    st.success(f"Processed {frame_idx} frames — {len(all_violations)} violation(s)")
                    if all_violations and _HAS_PANDAS:
                        st.dataframe(pd.DataFrame(all_violations), hide_index=True, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    logger.exception("Video processing error")

    else:
        st.session_state.setdefault("frame_counter", 0)
        st.session_state.frame_counter += 1
        if _HAS_PIL and Image:
            img = Image.new("RGB", (640, 360), (25, 30, 35))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"LIVE - {datetime.now().strftime('%H:%M:%S')}", fill=(255, 255, 255))
            d.text((10, 50), f"Frame #{st.session_state.frame_counter}", fill=(200, 200, 200))
            st.image(img, use_container_width=True)
        st.caption("Live feed display (simulated)")

        violations = safe_call("query_violations", limit=5)
        if violations and _HAS_PANDAS:
            st.markdown("**Recent Violations**")
            st.dataframe(pd.DataFrame(violations), hide_index=True, use_container_width=True)

with t2:
    st.subheader("Risk Map")
    st.info("Map requires folium/streamlit-folium packages.")

with t3:
    st.subheader("Hotspot Analysis")
    st.info("Chart requires altair package.")

with t4:
    st.subheader("Predicted Violations")
    st.info("Forecast data available when pipeline is connected.")

with t5:
    st.subheader("Repeat Offenders")
    st.info("Data available when pipeline is connected.")

with t6:
    st.subheader("Enforcement Plan")
    st.info("Plan available when pipeline is connected.")

with t7:
    st.subheader("Search & Reports")
    with st.form("search_form"):
        search_plate = st.text_input("License Plate")
        search_type = st.selectbox("Violation Type", ["All", "helmet", "triple_riding", "wrong_side", "red_light", "illegal_parking", "seatbelt"])
        search_btn = st.form_submit_button("Search")
    if search_btn:
        violations = safe_call("query_violations", plate=search_plate or None, violation_type=search_type if search_type != "All" else None, limit=100)
        if violations:
            if _HAS_PANDAS:
                df = pd.DataFrame(violations)
                st.dataframe(df, hide_index=True, use_container_width=True)
                st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), f"violations_{datetime.now():%Y%m%d}.csv")
        else:
            st.info("No results.")
