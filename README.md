# InMind Project: Industrial Object Detection

This repository contains the implementation of an object detection system for industrial environments using YOLOv5. The project focuses on detecting objects such as tuggers, cabinets, STRs, boxes, and forklifts.

## Table of Contents
- [Project Overview](#project-overview)
- [Part 1: Data Preparation & Visualization](#part-1-data-preparation--visualization)
- [Part 2: Model Training & Evaluation](#part-2-model-training--evaluation)
- [Part 3: Model Deployment & Inference](#part-3-model-deployment--inference)
- [Requirements](#requirements)
- [Results](#results)

## Project Overview

This project implements an object detection system for industrial environments. I train and evaluate YOLOv5 models with different hyperparameters to identify the best configuration for detecting industrial objects.

## Part 1: Data Preparation & Visualization

### 1. Loading the Dataset

I used PyTorch DataLoader to load my dataset and labels for efficient batch processing. The implementation can be found in `Project/Part_1/load_dataset.py`.

### 2. Visualization of Labeled Images

A function was developed to visualize the dataset with bounding box annotations to verify the correctness of my labels. This is implemented in `Project/Part_1/visualize.py`.

### 3. Dataset Augmentation

I explored dataset augmentation using Albumentations library to improve model robustness. The augmentation includes horizontal flips, rotations, and contrast adjustments. Implementation is in `Project/Part_1/augment_dataset.py`.

![Augmented Image Example 1](./assets/aug_im_bbox.png)
*Augmented image with preserved bounding boxes*

![Augmented Image Example 2](./assets/aug_im_bbox2.png)
*Another example of augmentation applied to my dataset*

### 4. Dataset Splitting

The dataset was split into training and validation sets with an 80/20 ratio to ensure proper model evaluation. The implementation is available in `Project/Part_1/split_dataset.py`.

## Part 2: Model Training & Evaluation

### 1. YOLOv5 Model Training

I used YOLOv5 for object detection. First, I cloned the YOLOv5 repository and installed its dependencies:

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install --no-deps -r requirements.txt
```

#### Dataset Preparation

YOLOv5 requires annotations in a specific format. Each image has a corresponding .txt file with annotations in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

My dataset was organized in the following structure:
```
/dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            val_image1.jpg
            val_image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            val_image1.txt
            val_image2.txt
            ...
```

A custom YAML file (`Project/Part_2/dataset.yaml`) was created to define my dataset:

```yaml
path: ../../dataset/yolo_data
train: images/train
val: images/val

# Classes
nc: 5
names: 
  0: tugger
  1: cabinet
  2: str
  3: box
  4: forklift
```

#### Training Process

I trained the model using the following command:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data ../../dataset/yolo_data/dataset.yaml --weights yolov5s.pt --cache --device 0
```

Parameters explanation:
- `--img 640`: Image size (640x640 pixels)
- `--batch 16`: Batch size for training
- `--epochs 50`: Number of training epochs
- `--data`: Path to the dataset YAML file
- `--weights yolov5s.pt`: Pre-trained weights (YOLOv5 small model)
- `--cache`: Cache images for faster training
- `--device 0`: Use GPU for training

![Training Command](./assets/training_cmd.png)
*Training command execution and initial output*

![Training Model Summary](./assets/training_modelsummary.png)
*Model architecture summary for YOLOv5s*

![Training Epochs](./assets/training_epochs.png)
*Training progress across epochs (dataset: 80 training images, 20 validation images)*

![Training Complete](./assets/training_complete.png)
*Training completion output with final metrics*

### 2. Model Evaluation and Hyperparameter Tuning

After training the initial model, I evaluated its performance on the test dataset using:

```bash
python val.py --weights runs/train/exp/weights/best.pt --data ../../dataset/yolo_data/dataset.yaml --img 640
```

![Model Evaluation](./assets/evaluate_model.png)
*Evaluation results of the trained model on test dataset*

I then retrained the model with different hyperparameters to improve performance:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data ../../dataset/yolo_data/dataset.yaml --weights yolov5s.pt --cache --device 0 --hyp data/hyps/hyp.scratch-high.yaml
```

The `hyp.scratch-high.yaml` file contains hyperparameters with higher augmentation settings to potentially improve model generalization.

![Hyperparameter Training](./assets/hyperparameter_train.png)
*Training with modified hyperparameters*

![Hyperparameter Training Complete](./assets/hyperpar_train_done.png)
*Completion output of hyperparameter-tuned training*

### 3. TensorBoard Visualization

To compare the performance of both models, I saved the training logs and used TensorBoard for visualization:

```bash
# For the standard model
python train.py --img 640 --batch 16 --epochs 50 --data ../../dataset/yolo_data/dataset.yaml --weights yolov5s.pt --cache --device 0 --project runs/train --name normal_train

# For the model with tuned hyperparameters
python train.py --img 640 --batch 16 --epochs 50 --data ../../dataset/yolo_data/dataset.yaml --weights yolov5s.pt --cache --device 0 --hyp data/hyps/hyp.scratch-high.yaml --project runs/train --name hyper_train
```

To visualize the metrics:

```bash
tensorboard --logdir=runs/train
```

This starts a local server at http://localhost:6006/ where you can view training metrics in real-time.

### 4. Model Performance Comparison

I compared both models using various metrics:

#### F1 Score Curves

![F1 Score (Normal Model)](./assets/f1_normal.png)
*F1 score curve for the standard model*

![F1 Score (Hyperparameter-tuned Model)](./assets/f1_hyper.png)
*F1 score curve for the hyperparameter-tuned model*

#### Mean Average Precision (mAP)

![mAP Comparison](./assets/comparison_map.png)
*Comparison of mAP@0.5 and mAP@0.5:0.95 between both models*

#### Loss Comparison

![Training Loss Comparison](./assets/comparison_loss.png)
*Box loss, class loss, and object loss during training for both models*

![Validation Loss Comparison](./assets/comp_val_loss.png)
*Validation losses for both models*

#### Learning Rate

![Learning Rate Comparison](./assets/compar_lr.png)
*Learning rate schedules for both models*

#### Confusion Matrices

![Confusion Matrix (Normal Model)](./assets/confusion_normal.png)
*Confusion matrix for the standard model*

![Confusion Matrix (Hyperparameter-tuned Model)](./assets/confusion_hyper.png)
*Confusion matrix for the hyperparameter-tuned model*

## Part 3: Model Deployment & Inference

### 1. Export Models to ONNX Format

I exported both trained models to ONNX format for efficient inference deployment. For the standard model:

```bash
python export.py --weights runs/train/normal_train/weights/best.pt --img-size 640 --batch-size 1 --dynamic --include onnx
```

And for the hyperparameter-tuned model:

```bash
python export.py --weights runs/train/hyper_high_train/weights/best.pt --img-size 640 --batch-size 1 --dynamic --include onnx
```

Parameters explanation:
- `--weights`: Path to the trained model weights
- `--img-size`: Input image size for inference (matching training size)
- `--batch-size`: Batch size for the ONNX export (typically 1 for inference)
- `--dynamic`: Enables dynamic axes for variable input sizes
- `--include onnx`: Specifies ONNX as the export format

After running these commands, the ONNX files were saved in the `runs/train/normal_train/weights/` & `runs/train/hyper_high_train/weights/` directory.

### 2. Network Visualization with Netron

I used Netron, a web-based neural network visualization tool, to inspect the exported models:

1. I visited the Netron website at https://netron.app
2. Uploaded my ONNX model files by clicking "Open" in the top-left corner
3. Navigated to the exported model files in the directory (`best.onnx`) and selected them

Netron displayed the complete architecture of the models, showing all layers, parameters, and connections.

![Netron (Normal Model)](./assets/view_ONNX_model.png)
*Netron visualization of the standard YOLOv5s model architecture*

![Netron (Tuned Model)](./assets/view_ONNX_model2.png)
*Netron visualization of the hyperparameter-tuned model architecture*

### 3. REST API Implementation with FastAPI

I developed a REST API using FastAPI to serve the trained object detection models. The API allows for real-time inference on uploaded images, returning both JSON output and visualized results.

#### API Endpoints

The API offers the following endpoints:

1. **`GET /`**: Redirects to the API documentation
2. **`GET /models`**: Lists all available ONNX models
3. **`POST /inference`**: Accepts an image and returns detected objects as JSON
4. **`POST /inference_image`**: Accepts an image and returns the same image with bounding boxes drawn

#### Architecture

The API is built using:
- **FastAPI**: For creating the REST API endpoints
- **ONNX Runtime**: For efficient model inference
- **Pillow**: For image manipulation and drawing bounding boxes
- **NumPy**: For numerical operations on image data

Key components of the implementation include:

- **Non-Maximum Suppression (NMS)**: Implemented to filter out redundant bounding boxes, ensuring only the highest confidence detection for each object is returned
- **YOLOv5 Output Processing**: Custom logic to decode the model's output tensor (shape [1, 25200, 10]) into usable bounding boxes, class predictions, and confidence scores
- **Visualization**: Drawing functionality to display detection results directly on the image

#### Sample JSON Output

```json
[
  {
    "Id": 172779,
    "ObjectClassName": "cabinet",
    "ObjectClassId": 2,
    "Left": 398,
    "Top": 23,
    "Right": 652,
    "Bottom": 427,
    "x_center": 0.2734,
    "y_center": 0.2083,
    "width": 0.1323,
    "height": 0.3741,
    "Confidence": 0.8976
  }
]
```

### Endpoints Testing Overview

![FastAPI docs](./assets/fastapi_endpoints.png)
*FastAPI docs page showing project endpoints*

![Models Endpoint](./assets/models_ep.png)
*'/models' endpoint returning the available models*

![Inference Endpoint](./assets/inference_ep.png)
*'/inference' endpoint showing the JSON output which includes the bounding boxes*

![Image Inference Endpoint](./assets/image_inference_ep.png)
*'/inference_image' endpoint showing the image and the returned bounding boxes*

### 4. Docker Containerization

I containerized the API for easy deployment and portability using Docker. The containerization includes:

#### Dockerfile

The Dockerfile specifies a Python 3.10 slim base image and sets up the application:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

# Create a directory for model path
RUN mkdir -p ../Part_2/yolov5/runs/train/normal_train/weights/

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

A docker-compose.yml file simplifies the deployment process:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../Part_2/yolov5/runs/train/:/app/Part_2/yolov5/runs/train/
    environment:
      - MODEL_DIR=Part_2/yolov5/runs/train/
```

This configuration:
- Maps port 8000 from the container to the host
- Mounts the model directory as a volume
- Sets the model directory via environment variable

#### Running the Containerized API

To run the API in a Docker container:

```bash
cd Project/Part_3
docker-compose up --build
```

![Docker Build Starting](./assets/docker_start.png)
*Docker Compose build process initiation*

![Docker Build Complete](./assets/docker_end.png)
*Successful completion of the Docker container build*

![Docker Application Logs](./assets/docker_app.png)
*Container logs showing API endpoint requests and responses*

Access the API documentation at http://localhost:8000/docs to test the endpoints.

## Requirements

The project dependencies are specified in `Project/requirements.txt`. The main dependencies include:
- torch==2.1.2+cu118
- torchvision==0.16.2+cu118
- matplotlib
- albumentations
- numpy
- tensorboard
- onnx
- onnxruntime
- fastapi
- uvicorn
- python-multipart

## Results

The hyperparameter-tuned model showed imbalanced little to no improved performance over the standard model, particularly in terms of mAP and F1 score. This is due to my small dataset size and because it is not balanced.
