import os
import streamlit as st
from ultralytics import YOLO

def model_selection():
    # ── (1) Scan your ../models/ folder for all .pt files ────────────────────────
    MODEL_DIR = os.path.join("..", "models")
    model_files = [
        f for f in os.listdir(MODEL_DIR)
        if f.lower().endswith(".pt")
    ]
    if not model_files:
        st.error(f"No *.pt files found in {MODEL_DIR}")
        st.stop()

    # ── (2) In the sidebar, show a dropdown to pick one of those .pt files ───────
    selected_model_filename = st.sidebar.selectbox(
        "Choose YOLO model:",
        model_files,
    )

    # ── (3) Load the chosen model just once ──────────────────────────────────────
    model_path = os.path.join(MODEL_DIR, selected_model_filename)
    st.session_state.model = YOLO(model_path)

    # ── (4) (Optional) Show which model is currently loaded ─────────────────────
    st.sidebar.write(f"Loaded model:  `{selected_model_filename}`")