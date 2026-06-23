import streamlit as st
st.set_page_config(page_title="Gridlock AI Test", layout="wide")
st.title("Gridlock AI")
st.write("Booting...")

import sys, os
os.environ["STREAMLIT_TEST"] = "1"
st.write(f"Python: {sys.version}")
st.write(f"CWD: {os.getcwd()}")

try:
    from PIL import Image
    st.write("PIL: OK")
except Exception as e:
    st.write(f"PIL: {e}")

try:
    import cv2
    st.write(f"cv2: OK (version {cv2.__version__})")
except Exception as e:
    st.write(f"cv2: FAILED - {e}")
    import traceback
    st.code(traceback.format_exc())

try:
    import ultralytics
    st.write(f"ultralytics: OK (version {ultralytics.__version__})")
except Exception as e:
    st.write(f"ultralytics: {e}")

st.write("App ready!")
