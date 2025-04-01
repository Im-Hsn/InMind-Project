import albumentations as A
from load_dataset import BMWObjectDataset
import os
import json
from PIL import Image
import numpy as np

def augment_dataset(image_dir, label_dir, output_dir):
    dataset = BMWObjectDataset(image_dir, label_dir)
    
    transformations = [
        ("horizontal_flip", A.HorizontalFlip(p=1.0)),
        ("rotate", A.Rotate(limit=30, p=1.0)),
        ("brightness", A.RandomBrightnessContrast(p=1.0))
    ]
    
    output_image_dir = os.path.join(output_dir, "images")
    output_label_dir = os.path.join(output_dir, "labels", "json")
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    image_count = 0
    
    for _, (image, labels) in enumerate(dataset):
        orig_image_path = os.path.join(output_image_dir, f"{image_count}.png")
        image.save(orig_image_path)
        
        orig_label_path = os.path.join(output_label_dir, f"{image_count}.json")
        with open(orig_label_path, "w") as f:
            json.dump(labels, f)
        
        image_count += 1
        
        image_array = np.array(image)
        bboxes = [[label["Left"], label["Top"], label["Right"], label["Bottom"]] for label in labels]
        category_ids = [label["ObjectClassId"] for label in labels]
        
        for name, transform in transformations:
            bbox_params = A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
            processor = A.Compose([transform], bbox_params=bbox_params)
            
            augmented = processor(image=image_array, bboxes=bboxes, category_ids=category_ids)
            
            aug_image_path = os.path.join(output_image_dir, f"{image_count}.png")
            Image.fromarray(augmented["image"]).save(aug_image_path)
            
            aug_labels = []
            for bbox, class_id in zip(augmented["bboxes"], augmented["category_ids"]):
                original_label = next(label for label in labels if label["ObjectClassId"] == class_id)
                
                aug_label = original_label.copy()
                aug_label["Left"] = int(bbox[0])
                aug_label["Top"] = int(bbox[1])
                aug_label["Right"] = int(bbox[2])
                aug_label["Bottom"] = int(bbox[3])
                aug_labels.append(aug_label)
            
            aug_label_path = os.path.join(output_label_dir, f"{image_count}.json")
            with open(aug_label_path, "w") as f:
                json.dump(aug_labels, f)
            
            image_count += 1

if __name__ == "__main__":
    image_dir = "../dataset/data/images"
    label_dir = "../dataset/data/labels/json"
    output_dir = "../dataset/aug_data"
    
    augment_dataset(image_dir, label_dir, output_dir)