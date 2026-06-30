"""
Dataset / DataLoader utilities for the NEU surface defect classification task.
"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 224

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    data_dir = Path(data_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=TRAIN_TRANSFORMS)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=EVAL_TRANSFORMS)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=EVAL_TRANSFORMS)

    # Sanity check: class-to-index mapping must match across splits
    assert train_ds.classes == val_ds.classes == test_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes
