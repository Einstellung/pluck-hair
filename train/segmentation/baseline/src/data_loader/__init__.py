"""Data loader package for semantic segmentation datasets."""

from .base_datamodule import BaseDataModule
from .voc_dataloader import VOC2012DataModule

__all__ = [
    "BaseDataModule",
    "VOC2012DataModule",
]
