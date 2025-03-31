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

@app.post("/inference_image")
async def inference_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    original_width, original_height = image.size
    
    resized_image = image.resize((640, 640))
    image_data = np.array(resized_image).astype('float32')
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    image_data /= 255.0

    inputs = {session.get_inputs()[0].name: image_data}
    outputs = session.run(None, inputs)
    
    detections = process_yolo_output(outputs[0])
    
    draw = ImageDraw.Draw(resized_image)
    
    for detection in detections:
        class_id = detection["class_id"]
        x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
        
        left = max(0, min(int(x1), int(x2), 639))
        right = min(639, max(int(x1), int(x2), 0))
        top = max(0, min(int(y1), int(y2), 639))
        bottom = min(639, max(int(y1), int(y2), 0))
        
        class_name = CLASS_MAPPING[class_id]
        confidence = detection["confidence"]
        
        class_colors = {
            0: (188, 60, 160),  # tugger - bc3ca0
            1: (196, 94, 105),  # cabinet - c45e69
            2: (51, 191, 45),   # str - 33bf2d
            3: (22, 74, 176),   # box - 164ab0
            4: (46, 179, 213)   # forklift - 2eb3d5
        }
        
        color = class_colors.get(class_id, (255, 0, 0))
        
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        
        label = f"{class_name} {confidence:.2f}"
        draw.text((left, top - 10), label, fill=color)
    
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Response(content=img_byte_arr.read(), media_type="image/png")

# To run the app, use: uvicorn main:app --reload