from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import shutil, os, uuid, cv2, base64
from model.stitcher import stitch_images
from model.patcher import divide_into_patches
from model.detector import detect_heads_and_segment
from yield_formula import estimate_yield
from ultralytics import YOLO
app = FastAPI()
model = YOLO("model/best.pt") 
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
        stitched = stitch_images(image_paths)
        patches = divide_into_patches(stitched)

        total_heads = 0
        annotated_stitched = stitched.copy()

        for (i, j), patch in patches:
            heads, _ = detect_heads_and_segment(patch)
            total_heads += heads

            results = model(patch)
            if results[0].boxes:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1 + j), int(y1 + i), int(x2 + j), int(y2 + i)
                    cv2.rectangle(annotated_stitched, (x1, y1), (x2, y2), (0, 255, 0), 2)

        yield_estimate = estimate_yield(total_heads)

        # Encode image to base64 for return
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
