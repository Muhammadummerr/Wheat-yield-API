from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import shutil, os, uuid, cv2, base64
from model.stitcher import stitch_images
from model.patcher import divide_into_patches
from model.detector import detect_heads_and_segment
from yield_formula import estimate_yield
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import uuid
import os
import shutil
import base64
import cv2
import numpy as np
from asone import ASOne, YOLOV9_C, BYTETRACK

# Initialize ASOne model
model = ASOne(tracker=BYTETRACK, detector=YOLOV9_C, use_cuda=True)

app = FastAPI()

def detect_heads_and_segment(image: np.ndarray, return_annotated=False):
    model_output = model.track(image, filter_classes=['wheat'])
    total_heads = len(model_output)
    annotated = ASOne.draw(model_output, image.copy()) if return_annotated else None
    return total_heads, annotated

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    input_dir = f"temp_{uuid.uuid4()}"
    os.makedirs(input_dir, exist_ok=True)
    image_paths = []

    for file in files:
        path = os.path.join(input_dir, file.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image_paths.append(path)

    try:
        # Load and stitch the images
        stitched = stitch_images(image_paths)  # Define this function
        patches = divide_into_patches(stitched)  # Define this function

        total_heads = 0
        annotated_stitched = stitched.copy()

        for (i, j), patch in patches:
            heads, _ = detect_heads_and_segment(patch)
            total_heads += heads

            model_output = model.track(patch, filter_classes=['wheat'])
            for box in model_output:
                x1, y1, x2, y2 = map(int, box['bbox'])
                x1, y1, x2, y2 = x1 + j, y1 + i, x2 + j, y2 + i
                cv2.rectangle(annotated_stitched, (x1, y1), (x2, y2), (0, 255, 0), 2)

        yield_estimate = estimate_yield(total_heads)  # Define this function

        # Encode final stitched annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_stitched)
        stitched_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={
            "total_heads": total_heads,
            "estimated_yield_kg_per_ha": round(yield_estimate, 2),
            "stitched_image_base64": stitched_base64
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        shutil.rmtree(input_dir)
