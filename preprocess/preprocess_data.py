from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom PyTorch Dataset class
class ReviewDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["ID"]) + ".jpg")
        text = row["text"]
        label = int(row["label"])

        # Load image
        image = image_transform(Image.open(img_path).convert("RGB"))

        # Tokenize text
        tokens = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }
