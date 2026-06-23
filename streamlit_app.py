import streamlit as st
st.set_page_config(page_title="Gridlock AI", layout="wide")

import os, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.title("Test")
st.write("If you see this, basic Streamlit works.")

try:
    import cv2
    st.write("cv2 imported OK")
except Exception as e:
    st.write(f"cv2 import failed: {e}")
