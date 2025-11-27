"""Preprocessing steps for vision pipeline."""

import logging
from typing import Tuple

import cv2
import numpy as np

from ..types import PipelineContext
from .base import ProcessStep

logger = logging.getLogger(__name__)


class ResizeStep(ProcessStep):
    """Resize image to specified size.
    
    Parameters:
        size: Target size as [width, height].
        keep_aspect: If True, maintain aspect ratio with padding.
        interpolation: OpenCV interpolation method.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.size = tuple(self.params.get("size", [640, 640]))
        self.keep_aspect = self.params.get("keep_aspect", True)
        self.interpolation = self.params.get("interpolation", cv2.INTER_LINEAR)

    @property
    def name(self) -> str:
        return "resize"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Resize the processed image."""
        image = ctx.processed_image
        original_shape = image.shape[:2]
        
        if self.keep_aspect:
            resized, scale, padding = self._resize_with_padding(image)
            ctx.metadata["resize_scale"] = scale
            ctx.metadata["resize_padding"] = padding
        else:
            resized = cv2.resize(
                image,
                self.size,
                interpolation=self.interpolation
            )
            ctx.metadata["resize_scale"] = (
                self.size[0] / original_shape[1],
                self.size[1] / original_shape[0]
            )
        
        ctx.processed_image = resized
        ctx.metadata["original_shape"] = original_shape
        ctx.metadata["resized_shape"] = resized.shape[:2]
        
        logger.debug(
            f"Resized from {original_shape} to {resized.shape[:2]}"
        )
        
        return ctx

    def _resize_with_padding(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float, Tuple[int, int, int, int]]:
        """Resize image maintaining aspect ratio with padding.
        
        Returns:
            Tuple of (resized_image, scale, (top, bottom, left, right) padding).
        """
        h, w = image.shape[:2]
        target_w, target_h = self.size
        
        # Calculate scale to fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=self.interpolation
        )
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Apply padding
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)  # YOLO default padding color
        )
        
        return padded, scale, (top, bottom, left, right)


class NormalizeStep(ProcessStep):
    """Normalize image pixel values.
    
    Parameters:
        mean: Mean values per channel [B, G, R].
        std: Standard deviation per channel [B, G, R].
        scale: Scale factor applied before normalization.
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.mean = np.array(self.params.get("mean", [0, 0, 0]), dtype=np.float32)
        self.std = np.array(self.params.get("std", [1, 1, 1]), dtype=np.float32)
        self.scale = self.params.get("scale", 1.0 / 255.0)

    @property
    def name(self) -> str:
        return "normalize"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Normalize the processed image."""
        image = ctx.processed_image.astype(np.float32)
        image = image * self.scale
        image = (image - self.mean) / self.std
        ctx.processed_image = image
        
        logger.debug("Applied normalization")
        return ctx


class EnhanceStep(ProcessStep):
    """Apply image enhancement.
    
    Parameters:
        brightness: Brightness adjustment (-1 to 1).
        contrast: Contrast adjustment factor.
        denoise: Apply denoising filter.
        denoise_strength: Denoising strength (h parameter).
    """

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.brightness = self.params.get("brightness", 0)
        self.contrast = self.params.get("contrast", 1.0)
        self.denoise = self.params.get("denoise", False)
        self.denoise_strength = self.params.get("denoise_strength", 10)

    @property
    def name(self) -> str:
        return "enhance"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Apply enhancement to the processed image."""
        image = ctx.processed_image
        
        # Brightness and contrast
        if self.brightness != 0 or self.contrast != 1.0:
            image = cv2.convertScaleAbs(
                image,
                alpha=self.contrast,
                beta=self.brightness * 255
            )
        
        # Denoising
        if self.denoise:
            image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                self.denoise_strength,
                self.denoise_strength,
                7,
                21
            )
        
        ctx.processed_image = image
        logger.debug("Applied enhancement")
        return ctx


