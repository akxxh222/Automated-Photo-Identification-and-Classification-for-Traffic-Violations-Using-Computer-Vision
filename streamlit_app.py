import streamlit as st
st.set_page_config(page_title="Gridlock AI Command Center", layout="wide", page_icon="🚦")

import os
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tempfile
from datetime import datetime
import subprocess
import tarfile
import shutil
import ctypes
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_CV2_FIXED = False
def _ensure_cv2():
    global _CV2_FIXED
    if _CV2_FIXED:
        if "cv2" in sys.modules: del sys.modules["cv2"]
        try:
            import cv2 as _cv2
            return _cv2, True
        except Exception:
            return None, False
    def _try_import():
        if "cv2" in sys.modules: del sys.modules["cv2"]
        try:
            import cv2 as _cv2
            return _cv2, True
        except Exception:
            return None, False
    def _extract_so(lib_dir):
        tarball = os.path.join(Path(__file__).parent, "streamlit_app_files", "glib_libs.tar.gz")
        if not os.path.exists(tarball):
            return False
        if os.path.isdir(lib_dir) and os.listdir(lib_dir):
            return True
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(lib_dir)
        for fname in os.listdir(lib_dir):
            fpath = os.path.join(lib_dir, fname)
            base = fname.rsplit(".", 2)[0]
            if ".so.0" in fname and not fname.endswith(".so.0"):
                simple = base.rsplit(".", 1)[0] if "." in base else base
                link = os.path.join(lib_dir, simple + ".so.0")
                if not os.path.exists(link):
                    try: os.symlink(fname, link)
                    except: shutil.copy2(fpath, link)
        return True
    cv2, ok = _try_import()
    if ok:
        _CV2_FIXED = True
        return cv2, True
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "opencv-python-headless", "--quiet", "--force-reinstall", "--no-deps"],
        capture_output=True, timeout=120
    )
    cv2, ok = _try_import()
    if ok:
        _CV2_FIXED = True
        return cv2, True
    lib_dir = os.path.join(tempfile.gettempdir(), "glib_so")
    if not _extract_so(lib_dir):
        return None, False
    os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    for soname in ["libglib-2.0.so.0", "libgthread-2.0.so.0"]:
        so_path = os.path.join(lib_dir, soname)
        for ver in ["", ".8000.0"]:
            p = so_path + ver if ver else so_path
            if os.path.exists(p):
                try:
                    ctypes.CDLL(os.path.realpath(p), ctypes.RTLD_GLOBAL)
                except Exception:
                    pass
                break
    for _ in range(3):
        cv2, ok = _try_import()
        if ok:
            _CV2_FIXED = True
            return cv2, True
    return None, False

try:
    cv2, _HAS_CV2 = _ensure_cv2()
except Exception as e:
    logger.warning("_ensure_cv2 crashed: %s", e)
    cv2 = None
    _HAS_CV2 = False

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    Image = None
    _HAS_PIL = False

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:
    pd = None
    _HAS_PD = False

try:
    import numpy as np
except ImportError:
    np = None

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
    if not pipeline.load():
        _load_error = pipeline._load_err
        pipeline = None
except Exception as e:
    _load_error = str(e)
    pipeline = None
    logger.warning("Pipeline not loaded: %s", e)

def safe_call(method, *a, **kw):
    if pipeline is None:
        return None
    try:
        return getattr(pipeline, method)(*a, **kw)
    except Exception as e:
        logger.warning("%s failed: %s", method, e)
        return None

_pipeline_active = pipeline is not None

st.sidebar.title("Gridlock AI Command Center")
st.sidebar.selectbox("Camera", ["All", "CAM_001", "CAM_002"])
st.sidebar.checkbox("Auto-Refresh (Live Feed)", value=False)

if not _pipeline_active:
    st.sidebar.error("Pipeline unavailable: " + (_load_error or "unknown"))
    st.sidebar.info("Run locally via Docker:\n`docker build -f Docker/Dockerfile -t gridlock-ai . && docker run -p 8000:8501 gridlock-ai`")

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
            with st.spinner("Analyzing..."):
                r = safe_call("process_image", data, camera_id="CAM_001")
                if r is not None and r.get("error"):
                    st.error(f"Pipeline error: {r['error']}")
                elif r and r.get("events"):
                    st.success(f"Detected {len(r['events'])} violation(s)")
                    if _HAS_PD:
                        st.dataframe(pd.DataFrame([
                            {"Type": e.get("violation_type","N/A"), "Confidence": f"{e.get('confidence',0):.0%}", "Plate": e.get("plate_text","N/A"), "Fine": f"₹{e.get('fine_amount',0)}"}
                            for e in r["events"]]), hide_index=True, use_container_width=True)
                    jr = r.get("junction_risk", {})
                    if jr and jr.get("score") is not None:
                        st.metric("Junction Risk Score", f"{jr['score']:.1f}", jr.get("tier","N/A"))
                else:
                    st.info("No violations detected.")
    elif mode == "Upload Video":
        try:
            import cv2
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
                        if res: vs.extend(res.get("events",[]))
                        pb.progress(min(fi/tf,1))
                    fi += 1
                cap.release(); os.unlink(p)
                st.success(f"Processed {fi} frames — {len(vs)} violation(s)")
                if vs and _HAS_PD:
                    st.dataframe(pd.DataFrame(vs), hide_index=True, use_container_width=True)
        except ImportError:
            st.warning("OpenCV not available on this server — video upload disabled.")
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
            st.dataframe(pd.DataFrame(vs) if isinstance(vs, list) else pd.DataFrame([vs]), hide_index=True, use_container_width=True)
        elif not _pipeline_active:
            st.info("Pipeline not connected — no data available.")
        else:
            st.info("No violations logged recently.")

with t2:
    st.subheader("City-Wide Junction Risk Intelligence")
    if _pipeline_active:
        try:
            import folium; from streamlit_folium import st_folium
            m = folium.Map(location=[12.9716,77.5946], zoom_start=12, tiles="CartoDB dark_matter")
            for cam, coords in {"CAM_001":[12.9716,77.5946],"CAM_002":[12.9352,77.6245]}.items():
                rd = safe_call("query_risk", f"J{cam[-3:]}")
                clr = "gray"; pop = f"<b>{cam}</b><br>No data"
                if rd: clr = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"yellow","LOW":"green"}.get(rd.get("risk_tier","LOW"),"gray"); pop = f"<b>{cam}</b><br>Score: {rd.get('risk_score',0):.1f}<br>Tier: {rd.get('risk_tier','LOW')}"
                folium.CircleMarker(location=coords, radius=10, color=clr, fill=True, fill_color=clr, fill_opacity=0.7, popup=pop).add_to(m)
            col1,_ = st.columns([3,1])
            with col1: st_folium(m, width=800, height=450)
        except ImportError: st.info("Map requires folium/streamlit-folium.")
    else:
        st.info("Pipeline not connected — run locally via Docker.")

with t3:
    st.subheader("Violation Density Hotspots")
    if _pipeline_active:
        hs = safe_call("query_hotspots")
        if hs:
            try:
                import folium; from folium.plugins import HeatMap; from streamlit_folium import st_folium
                col1,_ = st.columns([2,1])
                with col1:
                    m2 = folium.Map(location=[12.9716,77.5946], zoom_start=12)
                    HeatMap([[h["centroid"]["lat"],h["centroid"]["lon"],h["violation_count"]] for h in hs]).add_to(m2)
                    st_folium(m2, width=700, height=400)
            except ImportError: st.info("Charts require folium/altair.")
        else: st.info("No hotspot data.")
    else: st.info("Pipeline not connected — run locally via Docker.")

with t4:
    st.subheader("Predictive Analytics (Next 24 Hours)")
    if _pipeline_active:
        jid = st.selectbox("Junction", ["J001","J002"])
        fr = safe_call("query_forecast", jid, hours=24)
        fd = fr.get("forecast") if isinstance(fr,dict) else fr
        if fd:
            try:
                import altair as alt
                df = pd.DataFrame(fd)
                if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.altair_chart(alt.Chart(df).mark_line(color="cyan").encode(x="timestamp:T",y="predicted_violations:Q",tooltip=["timestamp","predicted_violations"]) + alt.Chart(df[df["event_flag"].notnull()]).mark_rule(color="red",strokeDash=[3,3]).encode(x="timestamp:T"), use_container_width=True)
            except ImportError: st.info("Charts require altair.")
        else: st.info("No forecast data.")
    else: st.info("Pipeline not connected — run locally via Docker.")

with t5:
    st.subheader("Repeat Offender Registry")
    if _pipeline_active:
        offs = safe_call("query_repeat_offenders")
        if offs and _HAS_PD:
            df = pd.DataFrame(offs)
            def ht(v): return f'background-color: {"#ff4b4b" if v=="HIGH RISK" else "#ffa500"}'
            st.dataframe(df.style.map(ht, subset=["risk_tier"]), use_container_width=True)
        else: st.info("No repeat offenders found.")
    else: st.info("Pipeline not connected — run locally via Docker.")

with t6:
    st.subheader("Daily AI Enforcement Plan")
    if _pipeline_active:
        plan = safe_call("query_enforcement_plan")
        if plan and plan.get("message") != "Plan not generated yet":
            st.success(f"Plan for **{plan.get('date', datetime.today().strftime('%Y-%m-%d'))}**")
            st.metric("Total Officers Required", plan.get("total_officers_needed","N/A"))
            if "recommended_allocations" in plan and _HAS_PD:
                st.table(pd.DataFrame(plan["recommended_allocations"]))
        else: st.info("No enforcement plan yet.")
    else: st.info("Pipeline not connected — run locally via Docker.")

with t7:
    st.subheader("Historical Violation Database")
    with st.form("search_form"):
        c1,c2,_ = st.columns(3)
        sp = c1.text_input("License Plate")
        stp = c2.selectbox("Violation Type", ["All","helmet","triple_riding","wrong_side","red_light","illegal_parking","seatbelt"])
        sb = st.form_submit_button("Search")
    vs = safe_call("query_violations", plate=sp or None, violation_type=stp if stp!="All" else None, limit=100) if _pipeline_active else None
    if vs and _HAS_PD:
        df = pd.DataFrame(vs)
        cols = [c for c in ["timestamp","plate_text","violation_type","camera_id","fine_amount","is_valid_plate"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), f"violations_{datetime.now():%Y%m%d}.csv")
    elif _pipeline_active:
        st.info("No violations match your search criteria.")
    else:
        st.info("Pipeline not connected — run locally via Docker.")
