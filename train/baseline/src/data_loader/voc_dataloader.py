"""VOC2012 Segmentation DataModule for PyTorch Lightning.

This module handles loading and preprocessing of the PASCAL VOC 2012 dataset
for semantic segmentation tasks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from .base_datamodule import BaseDataModule

if TYPE_CHECKING:
    from typing import Any


class VOC2012Dataset(Dataset):
    """PASCAL VOC 2012 Segmentation Dataset.
    
    Args:
        root: Path to VOC2012 root directory (containing JPEGImages, SegmentationClass, etc.)
        split: 'train', 'val', or 'trainval'
        image_size: Target size for resizing images and masks
        augment: Whether to apply data augmentation (only for training)
        num_classes: Number of classes (21 for full VOC, 1 for binary, 2 for cat/background)
        target_classes: Optional list of class indices to extract (for binary/multi-class subset)
    """
    
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
        'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root: str | Path,
        split: str = 'train',
        image_size: int = 256,
        augment: bool = False,
        num_classes: int = 21,
        target_classes: Optional[list[int]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.num_classes = num_classes
        self.target_classes = target_classes
        
        # Load image IDs from split file
        split_file = self.root / 'ImageSets' / 'Segmentation' / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]
        
        self.images_dir = self.root / 'JPEGImages'
        self.masks_dir = self.root / 'SegmentationClass'
        
        # Verify directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.images_dir / f'{image_id}.jpg'
        image = Image.open(image_path).convert('RGB')
        
        # Load mask
        mask_path = self.masks_dir / f'{image_id}.png'
        mask = Image.open(mask_path)
        
        # Apply transforms
        image, mask = self._transform(image, mask)
        
        return image, mask
    
    def _transform(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply transformations to image and mask."""
        
        # Convert mask to numpy for processing
        mask_np = np.array(mask, dtype=np.int64)
        
        # Handle boundary pixels (255) -> set to 0 (background) or ignore
        mask_np[mask_np == 255] = 0
        
        # If we have target_classes, remap mask to binary or subset
        if self.target_classes is not None:
            new_mask = np.zeros_like(mask_np)
            for new_idx, orig_idx in enumerate(self.target_classes, start=1):
                new_mask[mask_np == orig_idx] = new_idx
            mask_np = new_mask
        
        # Clip mask values to valid range
        mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        
        # Convert back to PIL for transforms
        mask = Image.fromarray(mask_np.astype(np.uint8), mode='L')
        
        # Resize
        image = TF.resize(image, (self.image_size, self.image_size), 
                         interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), 
                        interpolation=InterpolationMode.NEAREST)
        
        # Data augmentation (only for training)
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random rotation
            if torch.rand(1) > 0.5:
                angle = torch.randint(-15, 15, (1,)).item()
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
            # Color jitter (only for image)
            if torch.rand(1) > 0.5:
                image = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)(image)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        
        # Normalize image (ImageNet stats)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert mask to tensor
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image, mask


class VOC2012DataModule(BaseDataModule):
    """PyTorch Lightning DataModule for VOC2012 Segmentation.

    Args:
        root: Path to VOC2012 root directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target size for images
        num_classes: Number of classes (21 for full VOC)
        target_classes: Optional list of class indices to extract
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to keep workers alive
    """

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 256,
        train_split: str = 'train',
        val_split: str = 'val',
        num_classes: int = 21,
        target_classes: Optional[list[int]] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_classes = num_classes
        self.target_classes = target_classes
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.train_dataset = None
        self.val_dataset = None

    @classmethod
    def from_config(cls, config: Any) -> VOC2012DataModule:
        """
        Factory method to create VOC2012DataModule from configuration.

        Args:
            config: ExperimentConfig object containing data and model configuration

        Returns:
            Initialized VOC2012DataModule instance
        """
        # Extract parameters from config
        data_cfg = config.data

        # Build kwargs for __init__
        kwargs = {
            'root': data_cfg.root,
            'batch_size': data_cfg.batch_size,
            'num_workers': data_cfg.num_workers,
            'image_size': data_cfg.image_size,
            'pin_memory': data_cfg.pin_memory,
            'persistent_workers': data_cfg.persistent_workers,
            'num_classes': config.model.classes,
        }

        # Add any VOC-specific parameters from datamodule_kwargs
        if hasattr(data_cfg, 'datamodule_kwargs') and data_cfg.datamodule_kwargs:
            for key, value in data_cfg.datamodule_kwargs.items():
                kwargs[key] = value

        return cls(**kwargs)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        if stage in (None, 'fit'):
            if self.train_dataset is None:
                self.train_dataset = VOC2012Dataset(
                    root=self.root,
                    split=self.train_split,
                    image_size=self.image_size,
                    augment=True,
                    num_classes=self.num_classes,
                    target_classes=self.target_classes,
                )

        if stage in (None, 'fit', 'validate'):
            if self.val_dataset is None:
                self.val_dataset = VOC2012Dataset(
                    root=self.root,
                    split=self.val_split,
                    image_size=self.image_size,
                    augment=False,
                    num_classes=self.num_classes,
                    target_classes=self.target_classes,
                )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            self.setup(stage='fit')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        if self.val_dataset is None:
            self.setup(stage='validate')
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
