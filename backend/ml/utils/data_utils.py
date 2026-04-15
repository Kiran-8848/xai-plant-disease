"""
Data utilities for PlantVillage dataset.
Handles loading, splitting, augmentation and DataLoader creation.
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import yaml


# ─────────────────────────────────────────────
#  PlantVillage class names (all 38 classes)
# ─────────────────────────────────────────────
PLANT_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ─────────────────────────────────────────────
#  Transform factories
# ─────────────────────────────────────────────
def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """Same as val — used during XAI and API inference."""
    return get_val_transforms(image_size)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised tensor back to a uint8 HWC numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class PlantDiseaseDataset(Dataset):
    """
    Expects ImageFolder structure:
        root/
          ClassName1/img1.jpg
          ClassName2/img2.jpg
          ...
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None
    ):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        self._load_dataset(class_names)

    def _load_dataset(self, class_names: Optional[List[str]]):
        """Scan root directory, build class mapping and sample list."""
        dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if class_names:
            dirs = [d for d in dirs if d.name in class_names]

        for idx, d in enumerate(dirs):
            self.class_to_idx[d.name] = idx
            self.idx_to_class[idx] = d.name
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            for img_path in d.iterdir():
                if img_path.suffix.lower() in exts:
                    self.samples.append((img_path, idx))

        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def get_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, label in self.samples:
            name = self.idx_to_class[label]
            counts[name] = counts.get(name, 0) + 1
        return counts

    def save_class_mapping(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "class_to_idx": self.class_to_idx,
                "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()}
            }, f, indent=2)


# ─────────────────────────────────────────────
#  DataLoader builder
# ─────────────────────────────────────────────
def build_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.70,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, PlantDiseaseDataset]:
    """
    Returns (train_loader, val_loader, test_loader, full_dataset).
    The full_dataset carries the class mapping.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load without transforms to get indices first
    base_dataset = PlantDiseaseDataset(data_root)
    n = len(base_dataset)

    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val

    indices = list(range(n))
    random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    # Apply appropriate transforms via wrapper
    train_dataset = _TransformDataset(base_dataset, get_train_transforms(image_size), train_idx)
    val_dataset   = _TransformDataset(base_dataset, get_val_transforms(image_size),   val_idx)
    test_dataset  = _TransformDataset(base_dataset, get_val_transforms(image_size),   test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[Data] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"[Data] Classes: {base_dataset.num_classes}")

    return train_loader, val_loader, test_loader, base_dataset


class _TransformDataset(Dataset):
    """Wraps a base dataset with a specific transform applied at a subset of indices."""
    def __init__(self, dataset: PlantDiseaseDataset, transform, indices: List[int]):
        self.dataset   = dataset
        self.transform = transform
        self.indices   = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
