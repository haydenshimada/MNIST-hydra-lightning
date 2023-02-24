import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_file, dir = None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform
        self.dir = dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        if self.dir is not None:
          img_path = os.path.join(self.dir, str(img_path))
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def get_labels(self):
        return self.img_labels['label']