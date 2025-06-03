from ultralytics import YOLO
import cv2

# ── CONFIG ─────────────────────────────────────────────────────
INPUT_PATH  = '../video/crowd.mp4'           # change to your input video file
OUTPUT_PATH = '../video/outputs/output_blurred_video.mp4'  # desired output path

# If you have a YOLOv8 model fine-tuned on faces, use it here. Otherwise, using
# 'yolov8n.pt' will detect full persons (COCO class 0).
MODEL_WEIGHTS = '../models/model.pt'  # or 'yolov8n.pt' for person blurring
# ────────────────────────────────────────────────────────────────

# 1. Load YOLOv8
model = YOLO(MODEL_WEIGHTS)

# 2. Open the input video
cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_PATH}")
    exit()

# 3. Gather input video properties
fps         = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 output

# 4. Set up the VideoWriter for the output
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
if not out.isOpened():
    print(f"Error: Could not open video writer for {OUTPUT_PATH}")
    cap.release()
    exit()

print(f"Processing {INPUT_PATH} → {OUTPUT_PATH}")
print(f"Resolution: {width}×{height}, FPS: {fps}")

# 5. Process frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    # 5a. Run YOLOv8 inference on the current frame
    results = model(frame)    # returns a list; we only passed one frame
    res = results[0]

    # 5b. Iterate over each detected box and apply blur
    for box in res.boxes:
        cls_id = int(box.cls[0])
        # If using COCO weights and you only want to blur 'person' (class 0):
        # if cls_id != 0:
        #     continue

        # Get bounding-box coords and convert to ints
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # (Optional) Add padding so you don’t cut off edges of the face
        pad = 10
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        # Extract the ROI, blur it, and place it back
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue  # skip invalid boxes
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1:y2, x1:x2] = blurred_roi

    # 5c. Write the processed frame to the output video
    out.write(frame)

# 6. Release resources
cap.release()
out.release()
print("Done. Video saved to:", OUTPUT_PATH)
