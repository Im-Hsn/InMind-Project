from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse, Response
from typing import List
import os
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import io

app = FastAPI()

@app.get("/")
def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")

MODEL_DIR = "../Part_2/yolov5/runs/train/"
MODEL_PATH = os.path.join(MODEL_DIR, "normal_train/weights/best.onnx")

session = ort.InferenceSession(MODEL_PATH)

@app.get("/models", response_model=List[str])
def list_models():
    """Endpoint to list available models."""
    models = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".onnx"):
                rel_path = os.path.relpath(os.path.join(root, file), MODEL_DIR)
                models.append(rel_path)
    return models

CLASS_MAPPING = {
    0: "tugger",
    1: "cabinet",
    2: "str",
    3: "box",
    4: "forklift"
}


def process_yolo_output(output, conf_threshold=0.25):
    predictions = output[0]
    valid_detections = []
    
    for prediction in predictions:
        confidence = prediction[4]
        if confidence < conf_threshold:
            continue
            
        class_scores = prediction[5:]
        class_id = np.argmax(class_scores)
        
        if class_id not in CLASS_MAPPING:
            continue
            
        x, y, w, h = prediction[0:4]
        
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        valid_detections.append({
            "confidence": float(confidence),
            "class_id": int(class_id),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h)
        })
    
    return valid_detections

@app.post("/inference")
def run_inference(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(file.file.read()))
    original_width, original_height = image.size
    image = image.resize((640, 640))
    image_data = np.array(image).astype('float32')
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    image_data /= 255.0

    inputs = {session.get_inputs()[0].name: image_data}
    outputs = session.run(None, inputs)
    
    detections = process_yolo_output(outputs[0])
    
    results = []
    detection_id = 172779
    
    for detection in detections:
        class_id = detection["class_id"]
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        
        left = max(0, min(int(x1), int(x2), 640))
        right = min(640, max(int(x1), int(x2), 0))
        top = max(0, min(int(y1), int(y2), 640))
        bottom = min(640, max(int(y1), int(y2), 0))
        
        x_center = detection["x"] / 640
        y_center = detection["y"] / 640
        width = detection["w"] / 640
        height = detection["h"] / 640
        
        results.append({
            "Id": detection_id,
            "ObjectClassName": CLASS_MAPPING[class_id],
            "ObjectClassId": class_id + 1,
            "Left": left,
            "Top": top,
            "Right": right,
            "Bottom": bottom,
            "x_center": float(x_center),
            "y_center": float(y_center),
            "width": float(width),
            "height": float(height),
            "Confidence": float(detection["confidence"])
        })
        detection_id += 1
    
    return results



# To run the app, use: uvicorn main:app --reload