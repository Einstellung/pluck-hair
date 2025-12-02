"""Tests for vision pipeline module."""

import numpy as np
import pytest


class TestVisionTypes:
    """Tests for vision data types."""

    def test_bounding_box(self):
        """Test BoundingBox properties."""
        from src.core.vision.types import BoundingBox
        
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)
        
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.center == (60, 70)
        assert bbox.area == 10000
        assert bbox.to_xyxy() == (10, 20, 110, 120)
        assert bbox.to_xywh() == (10, 20, 100, 100)

    def test_detection(self):
        """Test Detection creation and serialization."""
        from src.core.vision.types import BoundingBox, Detection
        
        det = Detection(
            bbox=BoundingBox(x1=10, y1=20, x2=30, y2=40),
            object_type="debris",
            confidence=0.95,
        )
        
        assert det.object_type == "debris"
        assert det.confidence == 0.95
        
        d = det.to_dict()
        assert d["object_type"] == "debris"
        assert d["confidence"] == 0.95
        assert d["bbox"]["x1"] == 10


class TestPipelineSteps:
    """Tests for pipeline steps."""

    def test_tile_step(self):
        """Test TileStep slices image into tiles."""
        from src.core.vision.steps import TileStep
        from src.core.vision.types import PipelineContext
        
        # Create a 1000x800 image
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        step = TileStep({"tile_size": 640, "overlap": 0.2})
        ctx = PipelineContext(original_image=image)
        
        result = step.process(ctx)
        
        assert "tiles" in result.metadata
        assert result.metadata["tile_count"] > 0
        tiles = result.metadata["tiles"]
        assert len(tiles) >= 2
        for tile in tiles:
            assert tile.image.shape == (640, 640, 3)

    def test_nms_step(self):
        """Test NMSStep removes overlapping detections."""
        from src.core.vision.steps import NMSStep
        from src.core.vision.types import (
            BoundingBox, Detection, PipelineContext
        )
        
        # Create overlapping detections
        det1 = Detection(
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            object_type="debris",
            confidence=0.9,
        )
        det2 = Detection(
            bbox=BoundingBox(x1=15, y1=15, x2=55, y2=55),  # High overlap with det1
            object_type="debris",
            confidence=0.8,
        )
        det3 = Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=150),  # No overlap
            object_type="debris",
            confidence=0.7,
        )
        
        ctx = PipelineContext(
            original_image=np.zeros((200, 200, 3), dtype=np.uint8),
            detections=[det1, det2, det3]
        )
        
        step = NMSStep({"iou_threshold": 0.3})
        result = step.process(ctx)
        
        # det2 should be removed due to overlap with det1
        assert len(result.detections) == 2
        assert result.detections[0].confidence == 0.9  # det1 kept
        assert result.detections[1].confidence == 0.7  # det3 kept


class TestPipeline:
    """Tests for VisionPipeline."""

    def test_empty_pipeline(self, sample_image):
        """Test pipeline with no steps."""
        from src.core.vision.pipeline import VisionPipeline
        
        pipeline = VisionPipeline()
        result = pipeline.run(sample_image)
        
        assert result.detections == []
        assert result.processing_time_ms >= 0

    def test_pipeline_with_tile(self, sample_image):
        """Test pipeline with tile step."""
        from src.core.vision.pipeline import VisionPipeline
        from src.core.vision.steps import TileStep
        
        pipeline = VisionPipeline()
        pipeline.add_step(TileStep({"tile_size": 320, "overlap": 0.2}))
        
        result = pipeline.run(sample_image)
        
        assert "tile" in pipeline.step_names
        assert result.processing_time_ms >= 0

    def test_pipeline_from_config(self):
        """Test building pipeline from config."""
        from src.core.vision.pipeline import VisionPipeline
        
        config = {
            "pipeline": {
                "steps": [
                    {"type": "tile", "params": {"tile_size": 640}},
                    {"type": "nms", "params": {"iou_threshold": 0.4}},
                ]
            }
        }
        
        pipeline = VisionPipeline.from_config(config)
        
        assert pipeline.step_count == 2

    def test_create_step_unknown_type(self):
        """Test create_step raises error for unknown type."""
        from src.core.vision.steps import create_step
        
        with pytest.raises(ValueError, match="Unknown step type"):
            create_step({"type": "unknown_step"})

    def test_list_available_steps(self):
        """Test listing available step types."""
        from src.core.vision.steps import list_available_steps
        
        steps = list_available_steps()
        
        assert "tile" in steps
        assert "merge_tiles" in steps
        assert "yolo" in steps
        assert "nms" in steps
        assert "filter" in steps


