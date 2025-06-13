import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
from utils import model_selection


# # ── Load YOLO model once ─────────────────────────────────────────────────────
# model = YOLO("../models/model.pt")  # adjust path as needed

st.image("../img/logo.png", width=250)

# ── Streamlit UI ────────────────────────────────────────────────────────────
st.title("🛡 Obscura - Anonymization")

# Sidebar: choose mode and blur strength
mode = st.sidebar.radio(
    "Select input source:",
    ("Webcam", "Upload Video", "Upload Image")
)

blur_strength = st.sidebar.slider(
    "Blur intensity (odd number)", 5, 99, 35, step=2
)

model_selection()

# WebRTC configuration (only used if mode == "Webcam")
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Ensure output directory exists
OUTPUT_DIR = os.path.join("..", "video", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, initial_blur: int):
        # store a local reference to the YOLO model
        self.model = model
        # store blur strength (we'll allow updating via update_blur)
        self.blur_strength = initial_blur

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # un‐mirror

        # Use the model that was passed in at init time
        results = self.model(img)
        res = results[0]

        # (Optional) put a count of detected “person” boxes
        num_people = sum(1 for box in res.boxes if int(box.cls[0]) == 0)
        cv2.putText(
            img,
            f"Detected: {num_people} people",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # Blur each detected face/box
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad = 10
            x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
            x2p = min(x2 + pad, img.shape[1])
            y2p = min(y2 + pad, img.shape[0])

            face = img[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue

            ksize = self.blur_strength
            if ksize % 2 == 0:
                ksize += 1

            face_blur = cv2.GaussianBlur(face, (ksize, ksize), 0)
            img[y1p:y2p, x1p:x2p] = face_blur

        return frame.from_ndarray(img, format="bgr24")

    def update_blur(self, new_blur: int):
        self.blur_strength = new_blur


# ── (3) In run_webcam, pass the session_state.model into the processor factory ──
def run_webcam():
    st.subheader("Webcam Live Blur")

    # Pull the selected model out of session_state once, in the main thread
    # (so that we hand a “frozen” reference into the worker).
    yolo_model = st.session_state.get("model", None)
    if yolo_model is None:
        st.error("No model found in session_state. Please select a model first.")
        return

    # Create the webrtc_streamer, passing our VideoProcessor class a lambda
    # that creates it with the current model & blur strength.
    ctx = webrtc_streamer(
        key="yolo-face-blur",
        video_processor_factory=lambda: VideoProcessor(
            model=yolo_model,
            initial_blur=blur_strength,
        ),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # If the processor is instantiated, update its blur whenever the slider changes.
    if ctx.video_processor:
        ctx.video_processor.update_blur(blur_strength)


# ── Helper: Process a single image ────────────────────────────────────────────
def process_image(image_bgr, blur_strength):
    img = image_bgr.copy()
    results = st.session_state.model(img)
    res = results[0]

    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pad = 10
        x1p, y1p = max(x1 - pad, 0), max(y1 - pad, 0)
        x2p = min(x2 + pad, img.shape[1])
        y2p = min(y2 + pad, img.shape[0])

        face = img[y1p:y2p, x1p:x2p]
        if face.size == 0:
            continue

        ksize = blur_strength
        if ksize % 2 == 0:
            ksize += 1

        face_blur = cv2.GaussianBlur(face, (ksize, ksize), 0)
        img[y1p:y2p, x1p:x2p] = face_blur

    return img


# ── Image Upload Mode ─────────────────────────────────────────────────────────
def run_image_upload():
    st.subheader("Upload an Image for Blurring")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )
    if uploaded_file is not None:
        # Read bytes into numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Error: Could not read image.")
            return

        st.markdown("**Original Image:**")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), channels="RGB")

        with st.spinner("Processing..."):
            blurred_img = process_image(image_bgr, blur_strength)

        st.markdown("**Blurred Output:**")
        st.image(cv2.cvtColor(blurred_img, cv2.COLOR_BGR2RGB), channels="RGB")


# ── Video Upload Mode ─────────────────────────────────────────────────────────
def run_video_upload():
    st.subheader("Upload a Video for Blurring")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        return

    # 1. Save the uploaded bytes under ../video/inputs/
    INPUT_DIR = os.path.join("..", "video", "inputs")
    os.makedirs(INPUT_DIR, exist_ok=True)
    input_path = os.path.join(INPUT_DIR, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # 2. Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return

    # 3. Prepare the output path under ../video/outputs/
    base_name, _ = os.path.splitext(uploaded_file.name)
    output_name = f"{base_name}_blurred.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed = 0

    with st.spinner("Processing video... this may take a while"):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blurred_frame = process_image(frame, blur_strength)
            out.write(blurred_frame)

            processed += 1
            progress_bar.progress(min(processed / frame_count, 1.0))

    cap.release()
    out.release()

    # 4. Verify that the output file exists and is non‐empty
    if not os.path.exists(output_path):
        st.error(f"Error: Output file was not created at {output_path}")
        return

    file_size = os.path.getsize(output_path)
    if file_size < 1000:
        st.warning(
            f"Warning: Output file exists but is very small ({file_size} bytes). "
            "It might be corrupted."
        )

    st.success(f"Processing complete! Saved to `{output_path}` (size: {file_size:,} bytes)")

    # # ── Embed the video using base64 IData URI ──
    # with open(output_path, "rb") as f:
    #     video_bytes = f.read()

    # # Convert to base64 so we can embed directly in HTML
    # b64 = base64.b64encode(video_bytes).decode("utf-8")
    # video_html = f"""
    # <video width="700" controls>
    #     <source src="data:video/mp4;base64,{b64}" type="video/mp4">
    #     Your browser does not support the video tag.
    # </video>
    # """
    # st.markdown(video_html, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if mode == "Webcam":
        run_webcam()
    elif mode == "Upload Video":
        run_video_upload()
    else:  # "Upload Image"
        run_image_upload()


if __name__ == "__main__":
    main()