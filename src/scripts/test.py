from ultralytics import YOLO
import cv2

# Load a pre-trained model
model = YOLO('yolov8n.pt')

# Run inference on an image
results = model('../crowd.jpeg')

# Draw bounding boxes & display
img = cv2.imread('../crowd.jpeg')
annotated = results[0].plot()      # returns an ndarray with boxes drawn
cv2.imshow('YOLOv8 Detection', annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
