import streamlit as st
st.set_page_config(page_title="Gridlock AI Command Center", layout="wide", page_icon="🚦")

import os
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import tempfile
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_HAS_CV2 = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None

_HAS_PIL = False
try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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

if pipeline is not None and not getattr(pipeline, '_loaded', False):
    pipeline = None

def safe_call(method, *a, **kw):
    if pipeline is None:
        return None
    try:
        return getattr(pipeline, method)(*a, **kw)
    except Exception as e:
        logger.warning("%s failed: %s", method, e)
        return None

def analyze_with_pil(image_bytes: bytes) -> dict:
    if not _HAS_PIL or not Image:
        return {"processed_violations": 0, "events": [], "junction_risk": {"score": 0.0, "tier": "LOW"}, "demo": True}
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = list(edges.getdata()) if _HAS_NP else None
        if _HAS_NP:
            edge_pixels = sum(1 for p in edge_arr if p > 50)
            total_pixels = w * h
            edge_ratio = edge_pixels / total_pixels
            r_channel = np.array(img)[:, :, 0]
            red_pixels = int(np.sum(r_channel > 200))
        else:
            edge_ratio = 0.1
            red_pixels = 0

        events = []
        if edge_ratio > 0.08:
            events.append({"violation_type": "helmet", "confidence": 0.72 + min(edge_ratio * 2, 0.2), "plate_text": "KA01AB1234", "fine_amount": 1000, "evidence_path": None, "summary": "No Helmet Detected"})
            events.append({"violation_type": "triple_riding", "confidence": 0.68, "plate_text": "KA02CD5678", "fine_amount": 1500, "evidence_path": None, "summary": "Triple Riding Detected"})
        if red_pixels > 500:
            events.append({"violation_type": "red_light", "confidence": 0.85, "plate_text": "KA03EF9012", "fine_amount": 2000, "evidence_path": None, "summary": "Red Light Violation"})

        jr_score = min(edge_ratio * 20 + (red_pixels / total_pixels) * 5, 10.0)
        tier = "CRITICAL" if jr_score >= 8 else "HIGH" if jr_score >= 5 else "MEDIUM" if jr_score >= 2 else "LOW"
        return {"processed_violations": len(events), "events": events, "junction_risk": {"score": round(jr_score, 1), "tier": tier}, "demo": True}
    except Exception as e:
        logger.warning("PIL analysis failed: %s", e)
        return {"processed_violations": 0, "events": [], "junction_risk": {"score": 0.0, "tier": "LOW"}, "demo": True}

_DEMO_PLATES = ["KA01AB1234", "KA02CD5678", "KA03EF9012", "KA04GH3456", "KA05IJ7890"]
def demo_violations(limit=12):
    if not _HAS_PD:
        return []
    data = []
    for i in range(min(limit, 10)):
        data.append({"id": i, "timestamp": (datetime.now() - timedelta(minutes=i*15)).isoformat(), "plate_text": _DEMO_PLATES[i % 5], "plate_confidence": 0.85 + (i % 10) * 0.01, "violation_type": ["helmet", "triple_riding", "wrong_side", "red_light", "illegal_parking", "seatbelt"][i % 6], "violation_confidence": 0.72 + (i % 8) * 0.02, "camera_id": f"CAM_00{i%2+1}", "junction_id": f"J00{i%2+1}", "latitude": 12.97 + (i * 0.003), "longitude": 77.59 + (i * 0.002), "fine_amount": [500, 1000, 1500, 2000][i % 4], "is_valid_plate": i % 3 != 0})
    return data

st.sidebar.title("Gridlock AI Command Center")
st.sidebar.selectbox("Camera", ["All", "CAM_001", "CAM_002"])
st.sidebar.checkbox("Auto-Refresh (Live Feed)", value=False)

_is_demo = pipeline is None
if _is_demo:
    st.sidebar.warning("⚠️ Cloud Demo Mode — Full pipeline runs locally via Docker")

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
                if pipeline is not None:
                    r = safe_call("process_image", data, camera_id="CAM_001")
                else:
                    r = analyze_with_pil(data)
                if r and r.get("events"):
                    st.success(f"Detected {len(r['events'])} violation(s)" + (" (demo)" if r.get("demo") else ""))
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
        if not _HAS_CV2:
            st.warning("OpenCV not available on this server — video upload disabled in cloud mode.")
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
                        if res: vs.extend(res.get("events",[]))
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
            if _is_demo:
                d.text((10,90), "DEMO MODE", fill=(255,200,0))
            st.image(img, use_container_width=True)
        vs = safe_call("query_violations", limit=5) if pipeline else None
        if vs is None and _is_demo:
            vs = demo_violations(limit=5)
        if vs and _HAS_PD:
            st.markdown("**Real-Time Violation Log**")
            st.dataframe(pd.DataFrame(vs) if isinstance(vs, list) else pd.DataFrame([vs]), hide_index=True, use_container_width=True)
        elif vs is None or (isinstance(vs, list) and len(vs) == 0):
            st.info("No violations logged recently.")

with t2:
    st.subheader("City-Wide Junction Risk Intelligence")
    if not _is_demo:
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
        except ImportError: st.info("Map requires folium.")
    else:
        st.info("Risk map available with full pipeline — see Docker deployment.")

with t3:
    st.subheader("Violation Density Hotspots")
    hs = None
    if pipeline: hs = safe_call("query_hotspots")
    if not hs and _is_demo:
        hs = [{"cluster_id":i,"centroid":{"lat":12.97+(i*0.005),"lon":77.59+(i*0.003)},"violation_count":random.randint(3,15),"dominant_violation":["helmet","triple_riding","red_light","seatbelt"][i%4]} for i in range(1,4)]
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

with t4:
    st.subheader("Predictive Analytics (Next 24 Hours)")
    jid = st.selectbox("Junction", ["J001","J002"])
    fr = safe_call("query_forecast", jid, hours=24) if pipeline else None
    if not fr and _is_demo:
        import random
        base = datetime.now().replace(minute=0, second=0, microsecond=0)
        fr = {"junction_id":jid,"hours":24,"forecast":[{"timestamp":(base+timedelta(hours=h)).isoformat(),"predicted_violations":max(0,int(np.random.normal(8,3))),"confidence_interval":{"lower":max(0,int(np.random.normal(5,2))),"upper":int(np.random.normal(12,4))},"event_flag":"⏰ Peak" if h in [8,9,17,18] else None} for h in range(24)],"model_status":"demo"}
    fd = fr.get("forecast") if isinstance(fr,dict) else fr
    if fd:
        try:
            import altair as alt
            df = pd.DataFrame(fd)
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.altair_chart(alt.Chart(df).mark_line(color="cyan").encode(x="timestamp:T",y="predicted_violations:Q",tooltip=["timestamp","predicted_violations"]) + alt.Chart(df[df["event_flag"].notnull()]).mark_rule(color="red",strokeDash=[3,3]).encode(x="timestamp:T"), use_container_width=True)
        except ImportError: st.info("Charts require altair.")
    else: st.info("No forecast data.")

with t5:
    st.subheader("Repeat Offender Registry")
    offs = safe_call("query_repeat_offenders") if pipeline else None
    if not offs and _is_demo:
        offs = [{"plate_text":p,"violation_count":random.randint(2,8),"risk_tier":random.choice(["HIGH RISK","MEDIUM RISK"]),"last_seen":(datetime.now()-timedelta(hours=random.randint(1,48))).isoformat()} for p in _DEMO_PLATES[:3]]
    if offs and _HAS_PD:
        df = pd.DataFrame(offs)
        def ht(v): return f'background-color: {"#ff4b4b" if v=="HIGH RISK" else "#ffa500"}'
        st.dataframe(df.style.map(ht, subset=["risk_tier"]), use_container_width=True)
    else: st.info("No repeat offenders found.")

with t6:
    st.subheader("Daily AI Enforcement Plan")
    plan = safe_call("query_enforcement_plan") if pipeline else None
    if not plan and _is_demo:
        plan = {"date":datetime.today().strftime("%Y-%m-%d"),"total_officers_needed":random.randint(8,15),"recommended_allocations":[{"junction":"J001","officers":random.randint(3,6),"priority":"HIGH"},{"junction":"J002","officers":random.randint(2,4),"priority":"MEDIUM"}]}
    if plan and plan.get("message") != "Plan not generated yet":
        st.success(f"Plan for **{plan.get('date', datetime.today().strftime('%Y-%m-%d'))}**")
        st.metric("Total Officers Required", plan.get("total_officers_needed","N/A"))
        if "recommended_allocations" in plan and _HAS_PD:
            st.table(pd.DataFrame(plan["recommended_allocations"]))
    else: st.info("No enforcement plan yet.")

with t7:
    st.subheader("Historical Violation Database")
    with st.form("search_form"):
        c1,c2,_ = st.columns(3)
        sp = c1.text_input("License Plate")
        stp = c2.selectbox("Violation Type", ["All","helmet","triple_riding","wrong_side","red_light","illegal_parking","seatbelt"])
        sb = st.form_submit_button("Search")
    vs = safe_call("query_violations", plate=sp or None, violation_type=stp if stp!="All" else None, limit=100) if pipeline else None
    if not vs and _is_demo:
        vs = demo_violations(limit=100)
    if vs and _HAS_PD:
        df = pd.DataFrame(vs)
        cols = [c for c in ["timestamp","plate_text","violation_type","camera_id","fine_amount","is_valid_plate"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True)
        st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), f"violations_{datetime.now():%Y%m%d}.csv")
    else: st.info("No violations match your search criteria.")
