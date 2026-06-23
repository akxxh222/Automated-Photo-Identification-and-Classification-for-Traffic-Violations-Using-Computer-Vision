import os
import io
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import altair as alt
from datetime import datetime, timedelta
import time
import cv2
from PIL import Image
import logging

from pipeline_runner import get_pipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Gridlock AI Command Center", layout="wide", page_icon="🚦")

pipeline = get_pipeline()

if "violations_cache" not in st.session_state:
    st.session_state.violations_cache = []
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0
if "last_violation_time" not in st.session_state:
    st.session_state.last_violation_time = time.time()

st.sidebar.title("Gridlock AI Filters")
selected_cam = st.sidebar.selectbox("Camera Selector", ["All", "CAM_001", "CAM_002"])
date_range = st.sidebar.date_input("Date Range", [datetime.today() - timedelta(days=1), datetime.today()])
st.title("Traffic Enforcement & Risk Intelligence Platform")
st.markdown("Real-time monitoring, AI predictive analytics, and automated ticketing engine.")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Live Feed", "Risk Map", "Hotspot Analysis", "Predicted Violations",
    "Repeat Offenders", "Enforcement Plan", "Search & Reports"
])

with t1:
    st.subheader("Live Processing Feed")

    mode = st.radio("Input Mode", ["Simulated Feed", "Upload Image", "Upload Video"], horizontal=True)

    if mode == "Upload Image":
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_img is not None:
            uploaded_img.seek(0)
            image_bytes = uploaded_img.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Running detection pipeline..."):
                try:
                    result = pipeline.process_image(image_bytes, camera_id="CAM_001")
                    st.success(f"Processed: {result['processed_violations']} violation(s) detected")

                    if result["events"]:
                        events_data = []
                        for ev in result["events"]:
                            events_data.append({
                                "Type": ev.get("violation_type", ev.get("summary", "N/A")),
                                "Confidence": f"{ev.get('confidence', 0):.2%}",
                                "Plate": ev.get("plate_text", "N/A"),
                                "Fine": f"₹{ev.get('fine_amount', 0)}",
                            })
                        st.dataframe(pd.DataFrame(events_data), hide_index=True, use_container_width=True)

                        if result["events"][0].get("evidence_path"):
                            ev_path = result["events"][0]["evidence_path"]
                            if os.path.exists(ev_path):
                                st.image(ev_path, caption="Annotated Detection", use_container_width=True)

                    jr = result.get("junction_risk", {})
                    if jr.get("score") is not None:
                        st.metric("Junction Risk Score", f"{jr['score']:.1f}", jr.get("tier", "N/A"))
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

    elif mode == "Upload Video":
        uploaded_vid = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_vid is not None:
            uploaded_vid.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            st.info(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")

            sample_every = max(1, total_frames // 10)
            all_violations = []
            frame_idx = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every == 0:
                    _, buffer = cv2.imencode(".jpg", frame)
                    img_bytes = buffer.tobytes()
                    try:
                        result = pipeline.process_image(img_bytes, camera_id="CAM_001")
                        for ev in result.get("events", []):
                            all_violations.append({
                                "frame": frame_idx,
                                "type": ev.get("violation_type", "N/A"),
                                "confidence": ev.get("confidence", 0),
                                "plate": ev.get("plate_text", "N/A"),
                            })
                    except Exception:
                        pass
                    status_text.text(f"Processing frame {frame_idx}/{total_frames}")
                frame_idx += 1
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(tmp_path)

            st.success(f"Processed {frame_idx} frames — {len(all_violations)} violation(s) found")
            if all_violations:
                df_vid = pd.DataFrame(all_violations)
                st.dataframe(df_vid, hide_index=True, use_container_width=True)

    else:
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown("**Simulated Live Video**")
            st.session_state.frame_counter += 1
            dummy_frame = np.random.randint(50, 200, (360, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"LIVE - {datetime.now().strftime('%H:%M:%S')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(dummy_frame, f"Frame #{st.session_state.frame_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            st.image(dummy_frame, channels="BGR", use_container_width=True)

        with c2:
            st.markdown("**Real-Time Violation Log**")
            violations = pipeline.query_violations(limit=5)
            now = time.time()
            if violations:
                st.session_state.violations_cache = violations
                st.session_state.last_violation_time = now
                df_log = pd.DataFrame(violations)[['timestamp', 'plate_text', 'violation_type', 'camera_id']]
                st.dataframe(df_log, hide_index=True, use_container_width=True)
            elif st.session_state.violations_cache and (now - st.session_state.last_violation_time) < 30:
                st.caption("Showing cached violations")
                df_log = pd.DataFrame(st.session_state.violations_cache)[['timestamp', 'plate_text', 'violation_type', 'camera_id']]
                st.dataframe(df_log, hide_index=True, use_container_width=True)
            else:
                st.info("No violations logged recently.")

with t2:
    st.subheader("City-Wide Junction Risk Intelligence")
    c_map, c_details = st.columns([3, 1])

    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB dark_matter")
    cams = {"CAM_001": [12.9716, 77.5946], "CAM_002": [12.9352, 77.6245]}

    for cam, coords in cams.items():
        risk_data = pipeline.query_risk(f"J{cam[-3:]}")
        tier = risk_data["risk_tier"]
        color = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}.get(tier, "gray")
        folium.CircleMarker(
            location=coords, radius=10, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=f"<b>Junction:</b> J{cam[-3:]}<br><b>Score:</b> {risk_data['risk_score']:.1f}<br><b>Tier:</b> {tier}"
        ).add_to(m)

    with c_map:
        st_folium(m, width=800, height=450)

    with c_details:
        st.markdown("### Risk Legend")
        st.markdown("🔴 **CRITICAL** ≥ 8.0 - Immediate deployment")
        st.markdown("🟠 **HIGH** 5.0 - 7.9 - Deploy within 30 min")
        st.markdown("🟡 **MEDIUM** 2.0 - 4.9 - Flag for review")
        st.markdown("🟢 **LOW** < 2.0 - Monitor only")

with t3:
    st.subheader("Violation Density Hotspots (DBSCAN Clusters)")
    hotspots = pipeline.query_hotspots()

    if hotspots:
        col1, col2 = st.columns([2, 1])
        with col1:
            m_heat = folium.Map(location=[12.9716, 77.5946], zoom_start=12)
            heat_data = [[h["centroid"]["lat"], h["centroid"]["lon"], h["violation_count"]] for h in hotspots]
            HeatMap(heat_data).add_to(m_heat)
            st_folium(m_heat, width=700, height=400)

        with col2:
            st.markdown("**Dominant Violations per Hotspot**")
            df_hs = pd.DataFrame(hotspots)
            if not df_hs.empty:
                chart = alt.Chart(df_hs).mark_bar().encode(
                    x="cluster_id:O",
                    y="violation_count:Q",
                    color="dominant_violation:N",
                    tooltip=["cluster_id", "violation_count", "dominant_violation"]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hotspot clusters found for the selected window.")

with t4:
    st.subheader("Predictive Analytics (Next 24 Hours)")
    jid = st.selectbox("Select Junction for Forecast", ["J001", "J002"])

    forecast_resp = pipeline.query_forecast(jid, hours=24)
    forecast_data = forecast_resp.get("forecast") if isinstance(forecast_resp, dict) else forecast_resp

    if forecast_data:
        df_pred = pd.DataFrame(forecast_data)
        if "timestamp" in df_pred.columns:
            df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])

        line = alt.Chart(df_pred).mark_line(color="cyan").encode(
            x='timestamp:T', y='predicted_violations:Q', tooltip=['timestamp', 'predicted_violations', 'event_flag']
        )
        band = alt.Chart(df_pred).mark_errorband(opacity=0.3, color="cyan").encode(
            x='timestamp:T', y='confidence_interval.lower:Q', y2='confidence_interval.upper:Q'
        )

        events = df_pred[df_pred["event_flag"].notnull()]
        rules = alt.Chart(events).mark_rule(color="red", strokeDash=[3, 3]).encode(x='timestamp:T')
        text = alt.Chart(events).mark_text(align='left', dx=5, dy=-150, color="white").encode(x='timestamp:T', text='event_flag')

        st.altair_chart(band + line + rules + text, use_container_width=True)
    else:
        st.info("No forecast data available.")

with t5:
    st.subheader("Repeat Offender Registry")
    offenders = pipeline.query_repeat_offenders()

    if offenders:
        df_off = pd.DataFrame(offenders)

        def highlight_tier(val):
            color = '#ff4b4b' if val == 'HIGH RISK' else '#ffa500'
            return f'background-color: {color}'

        st.dataframe(df_off.style.map(highlight_tier, subset=['risk_tier']), use_container_width=True)
    else:
        st.info("No repeat offenders found in the selected window.")

with t6:
    st.subheader("Daily AI Enforcement Recommendation")
    plan = pipeline.query_enforcement_plan()

    if plan and plan.get("message") != "Plan not generated yet":
        st.success(f"Plan Generated for: **{plan.get('date', datetime.today().strftime('%Y-%m-%d'))}**")
        st.metric("Total Officers Required", plan.get("total_officers_needed", "N/A"))

        if "recommended_allocations" in plan:
            st.table(pd.DataFrame(plan["recommended_allocations"]))

        st.button("Download Plan as PDF (ReportLab)", disabled=True)
    else:
        st.info("No enforcement plan generated yet.")

with t7:
    st.subheader("Historical Violation Database")

    with st.form("search_form"):
        c1, c2, c3 = st.columns(3)
        search_plate = c1.text_input("License Plate")
        search_type = c2.selectbox("Violation Type", ["All", "helmet", "triple_riding", "wrong_side", "red_light", "illegal_parking", "seatbelt"])
        search_btn = st.form_submit_button("Search Database")

    violations = pipeline.query_violations(
        plate=search_plate if search_plate else None,
        violation_type=search_type if search_type != "All" else None,
        limit=100
    )

    if violations:
        df_res = pd.DataFrame(violations)
        display_cols = ['timestamp', 'plate_text', 'violation_type', 'camera_id', 'fine_amount', 'is_valid_plate']
        existing_cols = [col for col in display_cols if col in df_res.columns]
        st.dataframe(df_res[existing_cols], use_container_width=True)

        st.download_button(
            label="Export to CSV",
            data=df_res.to_csv(index=False).encode("utf-8"),
            file_name=f"violation_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No violations match your search criteria.")


