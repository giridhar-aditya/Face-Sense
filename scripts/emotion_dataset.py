import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Map string labels to ints
        self.label_map = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sad": 4,
            "surprise": 5,
            "neutral": 6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        label_str = self.data.iloc[idx]['emotion']
        label = self.label_map[label_str]

        # Open image (PIL) and convert grayscale or RGB as needed
        image = Image.open(img_path).convert('L')  # original FER2013 images are grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
