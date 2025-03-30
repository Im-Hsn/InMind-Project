import os
import json
import shutil
from PIL import Image

def convert_to_yolo_format(image_width, image_height, bbox):
    x_center = ((bbox["Left"] + bbox["Right"]) / 2) / image_width
    y_center = ((bbox["Top"] + bbox["Bottom"]) / 2) / image_height
    width = (bbox["Right"] - bbox["Left"]) / image_width
    height = (bbox["Bottom"] - bbox["Top"]) / image_height
    return x_center, y_center, width, height

def prepare_yolo_dataset(source_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    
    with open("../dataset/data/objectclasses.json", "r", encoding="utf-16") as f:
        classes = json.load(f)
    
    class_mapping = {cls["Id"]: i for i, cls in enumerate(classes)}
    class_names = [cls["Name"] for cls in classes]
    
    for split in ["train", "val"]:
        images_dir = os.path.join(source_dir, split, "images")
        labels_dir = os.path.join(source_dir, split, "labels", "json")
        
        for img_file in os.listdir(images_dir):
            if not img_file.endswith(".png"):
                continue
                
            img_path = os.path.join(images_dir, img_file)
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            shutil.copy(img_path, os.path.join(output_dir, "images", split, img_file))
            
            label_file = os.path.join(labels_dir, img_file.replace(".png", ".json"))
            with open(label_file, "r") as f:
                labels = json.load(f)
            
            yolo_label_path = os.path.join(output_dir, "labels", split, os.path.splitext(img_file)[0] + ".txt")
            with open(yolo_label_path, "w") as f:
                for obj in labels:
                    class_id = class_mapping[obj["ObjectClassId"]]
                    x_center, y_center, width, height = convert_to_yolo_format(img_width, img_height, obj)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml_str = f"path: {os.path.abspath(output_dir)}\n"
        yaml_str += f"train: images/train\n"
        yaml_str += f"val: images/val\n"
        yaml_str += f"nc: {len(classes)}\n"
        yaml_str += "names:\n"
        for i, name in enumerate(class_names):
            yaml_str += f"  {i}: {name}\n"
        f.write(yaml_str)
    
    print(f"Dataset prepared in YOLO format at {output_dir}")

if __name__ == "__main__":
    source_dir = "../dataset/split_data"
    output_dir = "../dataset/yolo_data"
    prepare_yolo_dataset(source_dir, output_dir)