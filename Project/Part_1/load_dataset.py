import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class BMWObjectDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace(".png", ".json"))
        
        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            labels = json.load(f)
        
        return image, labels

image_dir = "../data/images"
label_dir = "../data/labels/json"
dataset = BMWObjectDataset(image_dir, label_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)