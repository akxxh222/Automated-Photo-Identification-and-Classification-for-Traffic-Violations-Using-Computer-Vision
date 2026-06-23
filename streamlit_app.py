import streamlit as st
st.set_page_config(page_title="Gridlock AI Command Center", layout="wide", page_icon="🚦")

import os
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tempfile
from datetime import datetime
import logging

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

_HAS_PD = False
try:
    import pandas as pd
    _HAS_PD = True
except ImportError:
    pd = None

_HAS_NP = False
try:
    import numpy as np
    _HAS_NP = True
except ImportError:
    np = None

import subprocess
try:
    subprocess.run(["apt-get", "install", "-y", "libgl1", "libglib2.0-0"], capture_output=True, timeout=30)
except Exception:
    pass

try:
    from scripts.download_models import main as _download_models
    _download_models()
except Exception as e:
    logger.warning("Model download failed: %s", e)

pipeline = None
_load_error = None
try:
    from app.pipeline_runner import get_pipeline as _gp
    pipeline = _gp()
except Exception as e:
    _load_error = str(e)
    logger.warning("Pipeline not loaded: %s", e)

def safe_call(method, *a, **kw):
    if pipeline is None:
        return None
    try:
        return getattr(pipeline, method)(*a, **kw)
    except Exception as e:
        logger.warning("%s failed: %s", method, e)
        return None

st.sidebar.title("Gridlock AI Filters")
st.sidebar.selectbox("Camera", ["All", "CAM_001", "CAM_002"])
st.sidebar.checkbox("Auto-Refresh (Live Feed)", value=False)

_model_status = "NOT LOADED"
if pipeline is not None:
    _ps = getattr(pipeline, '_loaded', False)
    _model_status = f"LOADED={_ps}"
    if _ps:
        _ps2 = getattr(pipeline, 'pipeline', None)
        if _ps2:
            _ps3 = getattr(_ps2, 'loaded', False)
            _model_status += f" | MLPipeline.loaded={_ps3}"
_model_files = [f for f in ["models/vehicle_detector.pt","models/helmet_detector.pt","models/triple_riding_detector.pt","models/plate_detector.pt"] if os.path.exists(f)]
_model_status += f" | models_found={len(_model_files)}/4"
st.sidebar.caption(f"Status: {_model_status}")
if _load_error:
    st.sidebar.error(f"Init error: {_load_error[:200]}")
if pipeline is not None and getattr(pipeline, '_load_err', None):
    st.sidebar.error(f"load() error: {pipeline._load_err[:300]}")

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
        f = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if f:
            data = f.read()
            if _HAS_PIL and Image:
                st.image(Image.open(io.BytesIO(data)), caption="Uploaded", use_container_width=True)
            with st.spinner("Running detection pipeline..."):
                r = safe_call("process_image", data, camera_id="CAM_001")
                if r and r.get("events"):
                    st.success(f"Detected {len(r['events'])} violation(s)")
                    if _HAS_PD:
                        st.dataframe(pd.DataFrame([
                            {"Type": e.get("violation_type","N/A"), "Confidence": f"{e.get('confidence',0):.0%}", "Plate": e.get("plate_text","N/A"), "Fine": f"₹{e.get('fine_amount',0)}"}
                            for e in r["events"]]), hide_index=True, use_container_width=True)
                    for e in r["events"]:
                        p = e.get("evidence_path")
                        if p and os.path.exists(p) and _HAS_PIL and Image:
                            st.image(Image.open(p), caption=f"Evidence: {e.get('violation_type','')}", use_container_width=True)
                    jr = r.get("junction_risk", {})
                    if jr and jr.get("score") is not None:
                        st.metric("Junction Risk Score", f"{jr['score']:.1f}", jr.get("tier","N/A"))
                else:
                    st.info("No violations detected.")
    elif mode == "Upload Video":
        if not _HAS_CV2:
            st.warning("OpenCV not available on this server. Video upload disabled.")
        else:
            f = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
            if f:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(f.read())
                    p = tmp.name
                cap = cv2.VideoCapture(p)
                tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                st.info(f"Video: {tf} frames, {fps:.1f} fps, {tf/fps if fps>0 else 0:.1f}s")
                se = max(1, tf // 10)
                vs = []; fi = 0; pb = st.progress(0)
                while True:
                    r, frm = cap.read()
                    if not r: break
                    if fi % se == 0:
                        _, buf = cv2.imencode(".jpg", frm)
                        res = safe_call("process_image", buf.tobytes(), camera_id="CAM_001")
                        if res:
                            vs.extend(res.get("events",[]))
                        pb.progress(min(fi/tf,1))
                    fi += 1
                cap.release(); os.unlink(p)
                st.success(f"Processed {fi} frames — {len(vs)} violation(s)")
                if vs and _HAS_PD:
                    st.dataframe(pd.DataFrame(vs), hide_index=True, use_container_width=True)
    else:
        st.session_state.setdefault("frame_counter",0)
        st.session_state.frame_counter += 1
        if _HAS_PIL and Image:
            img = Image.new("RGB", (640,360), (25,30,35))
            d = ImageDraw.Draw(img)
            d.text((10,10), f"LIVE - {datetime.now():%H:%M:%S}", fill=(255,255,255))
            d.text((10,50), f"Frame #{st.session_state.frame_counter}", fill=(200,200,200))
            st.image(img, use_container_width=True)
        vs = safe_call("query_violations", limit=5)
        if vs and _HAS_PD:
            st.markdown("**Real-Time Violation Log**")
            st.dataframe(pd.DataFrame(vs), hide_index=True, use_container_width=True)
        elif not vs:
            st.info("No violations logged recently.")

with t2:
    st.subheader("City-Wide Junction Risk Intelligence")
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[12.9716,77.5946], zoom_start=12, tiles="CartoDB dark_matter")
        for cam, coords in {"CAM_001":[12.9716,77.5946],"CAM_002":[12.9352,77.6245]}.items():
            rd = safe_call("query_risk", f"J{cam[-3:]}")
            clr = "gray"
            pop = f"<b>Junction:</b> J{cam[-3:]}<br><b>Status:</b> No data"
            if rd:
                tier = rd.get("risk_tier","LOW")
                clr = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"yellow","LOW":"green"}.get(tier,"gray")
                pop = f"<b>Junction:</b> J{cam[-3:]}<br><b>Score:</b> {rd.get('risk_score',0):.1f}<br><b>Tier:</b> {tier}"
            folium.CircleMarker(location=coords, radius=10, color=clr, fill=True, fill_color=clr, fill_opacity=0.7, popup=pop).add_to(m)
        col1, col2 = st.columns([3,1])
        with col1: st_folium(m, width=800, height=450)
        with col2: st.markdown("### Risk Legend\n🔴 **CRITICAL** ≥ 8.0\n🟠 **HIGH** 5.0-7.9\n🟡 **MEDIUM** 2.0-4.9\n🟢 **LOW** < 2.0")
    except ImportError:
        st.info("Map requires folium + streamlit-folium.")

with t3:
    st.subheader("Violation Density Hotspots (DBSCAN Clusters)")
    hs = safe_call("query_hotspots")
    if hs:
        try:
            import folium; from folium.plugins import HeatMap; from streamlit_folium import st_folium
            col1, col2 = st.columns([2,1])
            with col1:
                m2 = folium.Map(location=[12.9716,77.5946], zoom_start=12)
                HeatMap([[h["centroid"]["lat"],h["centroid"]["lon"],h["violation_count"]] for h in hs]).add_to(m2)
                st_folium(m2, width=700, height=400)
            with col2:
                st.markdown("**Dominant Violations per Hotspot**")
                if _HAS_PD:
                    df = pd.DataFrame(hs)
                    if not df.empty:
                        import altair as alt
                        st.altair_chart(alt.Chart(df).mark_bar().encode(x="cluster_id:O", y="violation_count:Q", color="dominant_violation:N", tooltip=["cluster_id","violation_count","dominant_violation"]).interactive(), use_container_width=True)
        except ImportError:
            st.warning("Charts require altair/folium.")
    else:
        st.info("No hotspot clusters found.")

with t4:
    st.subheader("Predictive Analytics (Next 24 Hours)")
    jid = st.selectbox("Select Junction for Forecast", ["J001","J002"])
    fr = safe_call("query_forecast", jid, hours=24)
    fd = fr.get("forecast") if isinstance(fr,dict) else fr
    if fd:
        try:
            import altair as alt
            df = pd.DataFrame(fd)
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
            line = alt.Chart(df).mark_line(color="cyan").encode(x="timestamp:T", y="predicted_violations:Q", tooltip=["timestamp","predicted_violations","event_flag"])
            band = alt.Chart(df).mark_errorband(opacity=0.3,color="cyan").encode(x="timestamp:T", y="confidence_interval.lower:Q", y2="confidence_interval.upper:Q")
            ev = df[df["event_flag"].notnull()]
            rules = alt.Chart(ev).mark_rule(color="red",strokeDash=[3,3]).encode(x="timestamp:T")
            text = alt.Chart(ev).mark_text(align="left",dx=5,dy=-150,color="white").encode(x="timestamp:T",text="event_flag")
            st.altair_chart(band+line+rules+text, use_container_width=True)
        except ImportError:
            st.info("Charts require altair.")
    else:
        st.info("No forecast data available.")

with t5:
    st.subheader("Repeat Offender Registry")
    offs = safe_call("query_repeat_offenders")
    if offs and _HAS_PD:
        df = pd.DataFrame(offs)
        def ht(v): return f'background-color: {"#ff4b4b" if v=="HIGH RISK" else "#ffa500"}'
        st.dataframe(df.style.map(ht, subset=["risk_tier"]), use_container_width=True)
    else:
        st.info("No repeat offenders found.")

with t6:
    st.subheader("Daily AI Enforcement Recommendation")
    plan = safe_call("query_enforcement_plan")
    if plan and plan.get("message") != "Plan not generated yet":
        st.success(f"Plan for **{plan.get('date', datetime.today().strftime('%Y-%m-%d'))}**")
        st.metric("Total Officers Required", plan.get("total_officers_needed","N/A"))
        if "recommended_allocations" in plan and _HAS_PD:
            st.table(pd.DataFrame(plan["recommended_allocations"]))
        st.button("Download Plan as PDF", disabled=True)
    else:
        st.info("No enforcement plan generated yet.")

with t7:
    st.subheader("Historical Violation Database")
    with st.form("search_form"):
        c1,c2,_ = st.columns(3)
        sp = c1.text_input("License Plate")
        stp = c2.selectbox("Violation Type", ["All","helmet","triple_riding","wrong_side","red_light","illegal_parking","seatbelt"])
        sb = st.form_submit_button("Search Database")
    vs = safe_call("query_violations", plate=sp or None, violation_type=stp if stp!="All" else None, limit=100)
    if vs and _HAS_PD:
        df = pd.DataFrame(vs)
        cols = [c for c in ["timestamp","plate_text","violation_type","camera_id","fine_amount","is_valid_plate"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
        st.download_button("Export to CSV", df.to_csv(index=False).encode("utf-8"), f"violations_{datetime.now():%Y%m%d}.csv")
    else:
        st.info("No violations match your search criteria.")
