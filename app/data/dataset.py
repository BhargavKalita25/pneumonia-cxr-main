import os, glob, re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def default_transforms(img_size, train):
    if train:
        aug = [
            transforms.RandomResizedCrop(img_size, scale=(0.9,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7)
        ]
    else:
        aug = [transforms.Resize((img_size,img_size))]
    aug += [
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    return transforms.Compose(aug)

def class_from_path(path): 
    return 1 if "PNEUMONIA" in path.upper() else 0

def patient_id_from_filename(path):
    stem = os.path.basename(path).split('.')[0]
    return re.findall(r'\d+', stem)[0] if re.findall(r'\d+', stem) else stem

class CXRFolder(Dataset):
    def __init__(self, root, split, img_size, train):
        self.items = sorted(glob.glob(os.path.join(root, split, "*", "*")))
        if not self.items: raise RuntimeError(f"No images in {root}/{split}")
        self.tfm = default_transforms(img_size, train)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = self.items[i]
        img = Image.open(p).convert("RGB")
        y = class_from_path(p)
        return self.tfm(img), torch.tensor(y, dtype=torch.float32), p, patient_id_from_filename(p)
