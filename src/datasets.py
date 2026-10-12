import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}


def get_files_and_labels(root_dir):
    paths, labels = [], []
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for img in sorted(os.listdir(cls_dir)):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(cls_dir, img))
                labels.append(CLASS_TO_IDX[cls_name])
    return np.array(paths), np.array(labels)


class BRISCDataset(Dataset):
    """
    Brain Image Segmentation and Classification (BRISC-2025) dataset.
    Pre-loads all images into RAM for fast iteration during training.

    Dataset: https://www.kaggle.com/datasets/briscdataset/brisc2025
    """

    def __init__(self, paths, labels, transform=None):
        self.labels = labels
        self.transform = transform
        self.images = [
            cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            for p in paths
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(image=img)['image']
        return img, self.labels[idx]


def build_transforms(image_size=224):
    """Return (train_transform, test_transform) albumentations pipelines."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    test_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return train_transform, test_transform


def build_dataloaders(
    dataset_dir,
    batch_size=64,
    val_split=0.2,
    seed=42,
    num_workers=0,
    image_size=224,
):
    """
    Build train / val / test DataLoaders for BRISC-2025.

    Args:
        dataset_dir: Path to the ``classification_task`` directory containing
                     ``train/`` and ``test/`` subdirectories.
        batch_size:  Number of samples per batch.
        val_split:   Fraction of training data held out for validation.
        seed:        Random seed for reproducible splits.
        num_workers: Number of worker processes for DataLoader.
        image_size:  Height and width to resize images to.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_transform, test_transform = build_transforms(image_size)

    full_train_paths, full_train_labels = get_files_and_labels(
        os.path.join(dataset_dir, 'train')
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        full_train_paths, full_train_labels,
        test_size=val_split,
        stratify=full_train_labels,
        random_state=seed,
    )
    test_paths, test_labels = get_files_and_labels(os.path.join(dataset_dir, 'test'))

    train_loader = DataLoader(
        BRISCDataset(train_paths, train_labels, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        BRISCDataset(val_paths, val_labels, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        BRISCDataset(test_paths, test_labels, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
