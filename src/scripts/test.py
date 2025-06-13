from ultralytics import YOLO
import cv2

# Load a pre-trained model (assumes this is already fine-tuned for faces;
# if you’re still using the default COCO weights, “cls 0” = “person,” not necessarily just the face)

# load model
model = YOLO("../models/model.pt")
results = model('../img/strato.jpg')      # returns a list, so results[0] is our Results object

# Read the same image with OpenCV
img = cv2.imread('../img/strato.jpg')
res = results[0]

# Loop over each detected box
for box in res.boxes:
    cls_id = int(box.cls[0])        # class index (e.g. 0 for “person”)
    # If you only want to blur faces (and your weights are truly a face model),
    # you can skip this check. If using COCO, you might want to do:
    # if cls_id != 0:
    #     continue

    # Get the absolute pixel coordinates of the bounding box
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    # (Optional) Expand the box slightly so you don't cut off ears/edges
    pad = 10
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, img.shape[1])
    y2 = min(y2 + pad, img.shape[0])

    # Extract ROI and blur it
    roi = img[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)

    # Put the blurred region back into the image
    img[y1:y2, x1:x2] = blurred_roi

# Show (or save) the final result
cv2.imshow('Blurred Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

pass

#This is for testing the workflow

