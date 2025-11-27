"""YOLO detection step for vision pipeline."""

import logging
from typing import Dict, Optional

from ..types import BoundingBox, Detection, ObjectType, PipelineContext
from .base import ProcessStep

logger = logging.getLogger(__name__)


class YOLODetectStep(ProcessStep):
    """YOLO object detection step.
    
    Uses Ultralytics YOLO for object detection. Supports various
    YOLO versions (v8, v11, etc.) and model sizes.
    
    Parameters:
        model: Path to model file (.pt or .onnx).
        conf: Confidence threshold (0-1).
        iou: IoU threshold for NMS.
        device: Device to run inference ('cuda', 'cpu', 'auto').
        half: Use FP16 inference (GPU only).
        classes: List of class indices to detect (None for all).
    """

    # Default mapping from class index to ObjectType
    # This should match the training class order
    DEFAULT_CLASS_MAP: Dict[int, ObjectType] = {
        0: ObjectType.HAIR,
        1: ObjectType.BLACK_SPOT,
        2: ObjectType.YELLOW_SPOT,
    }

    def __init__(self, params: dict = None):
        super().__init__(params)
        
        self.model_path = self.params.get("model", "models/best.pt")
        self.confidence = self.params.get("conf", 0.5)
        self.iou_threshold = self.params.get("iou", 0.45)
        self._device_config = self.params.get("device", "auto")
        self.half = self.params.get("half", False)
        self.classes = self.params.get("classes", None)
        
        # Custom class mapping if provided
        self.class_map = self.params.get("class_map", self.DEFAULT_CLASS_MAP)
        
        # Lazy load model
        self._model = None
        self._resolved_device: Optional[str] = None

    @property
    def name(self) -> str:
        return "yolo_detect"

    def _resolve_device(self) -> str:
        """Resolve the actual device to use."""
        if self._resolved_device is not None:
            return self._resolved_device
        
        import torch
        
        if self._device_config == "auto":
            if torch.cuda.is_available():
                self._resolved_device = "cuda"
                logger.info("Auto-detected CUDA, using GPU")
            else:
                self._resolved_device = "cpu"
                logger.info("CUDA not available, using CPU")
        elif self._device_config == "cuda":
            if torch.cuda.is_available():
                self._resolved_device = "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self._resolved_device = "cpu"
        else:
            self._resolved_device = self._device_config
        
        return self._resolved_device

    @property
    def model(self):
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            
            logger.info(f"Loading YOLO model: {self.model_path}")
            self._model = YOLO(self.model_path)
            
            device = self._resolve_device()
            logger.info(f"YOLO model loaded, will use device: {device}")
        
        return self._model

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run YOLO detection on the image."""
        device = self._resolve_device()
        
        # FP16 only works on CUDA
        use_half = self.half and device == "cuda"
        if self.half and device != "cuda":
            logger.warning("FP16 (half) requested but device is not CUDA, disabling")
        
        # Run inference on processed image with correct device
        results = self.model.predict(
            ctx.processed_image,
            conf=self.confidence,
            iou=self.iou_threshold,
            half=use_half,
            classes=self.classes,
            device=device,  # Pass device to predict
            verbose=False,
        )
        
        # Get scale and padding info for coordinate transformation
        scale = ctx.metadata.get("resize_scale", 1.0)
        padding = ctx.metadata.get("resize_padding", (0, 0, 0, 0))
        original_shape = ctx.metadata.get(
            "original_shape",
            ctx.original_image.shape[:2]
        )
        
        # Parse results
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Get coordinates in resized image space
                xyxy = box.xyxy[0].cpu().numpy()
                cls_idx = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Transform coordinates back to original image space
                x1, y1, x2, y2 = self._transform_coordinates(
                    xyxy, scale, padding, original_shape
                )
                
                # Map class index to ObjectType
                object_type = self.class_map.get(cls_idx, ObjectType.UNKNOWN)
                
                detection = Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    object_type=object_type,
                    confidence=conf,
                )
                
                ctx.detections.append(detection)
        
        ctx.metadata["yolo_detection_count"] = len(ctx.detections)
        
        logger.debug(f"YOLO detected {len(ctx.detections)} objects")
        return ctx

    def _transform_coordinates(
        self,
        xyxy,
        scale,
        padding,
        original_shape,
    ):
        """Transform coordinates from resized to original image space.
        
        Args:
            xyxy: [x1, y1, x2, y2] in resized image.
            scale: Resize scale factor.
            padding: (top, bottom, left, right) padding.
            original_shape: (height, width) of original image.
            
        Returns:
            Transformed (x1, y1, x2, y2) in original image space.
        """
        x1, y1, x2, y2 = xyxy
        
        # If we have padding info, remove it
        if isinstance(padding, tuple) and len(padding) == 4:
            top, bottom, left, right = padding
            x1 = x1 - left
            y1 = y1 - top
            x2 = x2 - left
            y2 = y2 - top
        
        # Scale back to original size
        if isinstance(scale, (int, float)):
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
        elif isinstance(scale, tuple):
            x1 = x1 / scale[0]
            y1 = y1 / scale[1]
            x2 = x2 / scale[0]
            y2 = y2 / scale[1]
        
        # Clip to image bounds
        h, w = original_shape
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return float(x1), float(y1), float(x2), float(y2)

