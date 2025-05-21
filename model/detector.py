from ultralytics import YOLO
import numpy as np

model = YOLO(r"model\best.pt")  # Update with actual path

def detect_heads_and_segment(image: np.ndarray, return_annotated=False):
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
    total_heads = len(boxes)
    annotated = results[0].plot() if return_annotated else None
    return total_heads, annotated
