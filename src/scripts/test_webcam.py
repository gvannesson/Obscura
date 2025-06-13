from ultralytics import YOLO
import cv2

# 1. Load your face‐detection model (if you have a face‐specific weight, use that)
model = YOLO('../models/model.pt') # or 'yolov8n.pt' if you only have COCO (will detect full person)

# 2. Open your default webcam (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# 3. Optionally, set a smaller resolution for faster inference
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── UN-MIRROR THE FRAME ─────────────────────────────────────
    # If your camera feed is mirrored, this flips it back
    frame = cv2.flip(frame, 1)
    # ────────────────────────────────────────────────────────────

    # 4. Run YOLOv8 on this frame
    results = model(frame)    # returns a list; we only passed one frame
    res = results[0]

    # 5. For each detected box, extract ROI, blur it, and paste back
    for box in res.boxes:
        cls_id = int(box.cls[0])
        # If using COCO weights and you only want to blur people:
        # if cls_id != 0:
        #     continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Add padding so the blur covers a bit more than just the box
        pad = 10
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        roi = frame[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1:y2, x1:x2] = blurred_roi

    # 6. Show the blurred video frame
    cv2.imshow('Webcam Face Blurring', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Cleanup
cap.release()
cv2.destroyAllWindows()
