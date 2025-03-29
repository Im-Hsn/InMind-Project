import os
import random
import shutil
from load_dataset import BMWObjectDataset

def split_dataset(data_dir, output_dir, train_ratio=0.8):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels", "json")
    
    dataset = BMWObjectDataset(image_dir, label_dir)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels", "json"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels", "json"), exist_ok=True)
    
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    for idx in train_indices:
        src_img = os.path.join(image_dir, dataset.image_files[idx])
        dst_img = os.path.join(train_dir, "images", dataset.image_files[idx])
        shutil.copy(src_img, dst_img)
        
        label_file = dataset.image_files[idx].replace(".png", ".json")
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(train_dir, "labels", "json", label_file)
        shutil.copy(src_label, dst_label)
    
    for idx in val_indices:
        src_img = os.path.join(image_dir, dataset.image_files[idx])
        dst_img = os.path.join(val_dir, "images", dataset.image_files[idx])
        shutil.copy(src_img, dst_img)
        
        label_file = dataset.image_files[idx].replace(".png", ".json")
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(val_dir, "labels", "json", label_file)
        shutil.copy(src_label, dst_label)
    
    print(f"Split: {len(train_indices)} training, {len(val_indices)} validation samples")

if __name__ == "__main__":
    split_dataset("../aug_data", "../split_data", 0.8)