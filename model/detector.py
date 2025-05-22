# from ultralytics import YOLO
# import numpy as np

# model = YOLO(r"model\best.pt")  # Update with actual path

# def detect_heads_and_segment(image: np.ndarray, return_annotated=False):
#     results = model(image)
#     boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
#     total_heads = len(boxes)
#     annotated = results[0].plot() if return_annotated else None
#     return total_heads, annotated


import numpy as np
import cv2
from asone import ASOne, YOLOV9_C, BYTETRACK

# Initialize ASOne with YOLOv9C detector and ByteTrack tracker
model = ASOne(tracker=BYTETRACK, detector=YOLOV9_C, use_cuda=True)

def detect_heads_and_segment(image: np.ndarray, return_annotated=False):
    # Run detection and tracking on a single image frame
    model_output = model.track(image, filter_classes=['wheat'])  # Assuming heads = people

    # Count the number of detections (heads)
    total_heads = len(model_output)

    # Draw annotations if requested
    annotated = ASOne.draw(model_output, image.copy()) if return_annotated else None

    return total_heads, annotated
