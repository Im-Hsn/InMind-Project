from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from typing import List
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
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
                models.append(file)
    return models

@app.post("/inference")
def run_inference(file: UploadFile = File(...)):
    """Endpoint to run inference on an uploaded image."""
    image = Image.open(io.BytesIO(file.file.read()))
    image = image.resize((640, 640))
    image_data = np.array(image).astype('float32')
    image_data = np.transpose(image_data, (2, 0, 1)) 
    image_data = np.expand_dims(image_data, axis=0)
    image_data /= 255.0

    inputs = {session.get_inputs()[0].name: image_data}
    outputs = session.run(None, inputs)

    results = [{"bbox": [0, 0, 100, 100], "label": "example", "score": 0.9}]

    return {"results": results}

# To run the app, use: uvicorn main:app --reload