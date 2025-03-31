from fastapi import FastAPI
from typing import List
import os

app = FastAPI()

MODEL_DIR = "runs/train/"

@app.get("/models", response_model=List[str])
def list_models():
    """Endpoint to list available models."""
    models = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith(".onnx"):
                models.append(file)
    return models