import cv2
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Charge ton modèle YOLO (modifie le chemin)
model = YOLO("../models/model.pt")

st.title("🛡 Obscura - Anonymization")

# Slider en sidebar pour régler le flou (sera accessible en live)
blur_strength = st.sidebar.slider("Blur intensity (odd number)", 5, 99, 35, step=2)

# Configuration WebRTC (optionnel, utile pour déploiement)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Classe pour traiter les frames webcam
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.blur_strength = blur_strength
        self.last_count = 0  # dans recv


    def recv(self, frame):
        # Convert frame en numpy array BGR
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Détection YOLO (renvoie liste des boîtes)
        results = model(img)
        res = results[0]
        num_people = sum(1 for box in res.boxes if int(box.cls[0]) == 0)
        self.last_count = num_people  # dans recv
        cv2.putText(
                        img,
                        f"Personnes detectees: {num_people}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
        # Floute chaque visage détecté
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            ksize = self.blur_strength
            # ksize doit être impair
            if ksize % 2 == 0:
                ksize += 1
            face_blur = cv2.GaussianBlur(face, (ksize, ksize), 0)
            img[y1:y2, x1:x2] = face_blur

        # Convertir en VideoFrame et renvoyer
        return frame.from_ndarray(img, format="bgr24")

    def update_blur(self, new_blur):
        self.blur_strength = new_blur


def main():
    ctx = webrtc_streamer(
        key="yolo-face-blur",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        # Update blur en temps réel depuis le slider
        ctx.video_processor.update_blur(blur_strength)



if __name__ == "__main__":
    main()
