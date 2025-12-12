"""Detection task with optional ByteTrack tracking."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.core.vision.pipeline import VisionPipeline
from src.core.vision.types import BoundingBox, Detection
from src.scheduler.tasks.base import Task, TaskIterationResult
from src.scheduler.tasks.conditions import DoneCondition, AlwaysFalseCondition
from src.scheduler.tasks.stats import TaskStats

if TYPE_CHECKING:
    from src.config import TrackerConfig, SmoothingConfig

logger = logging.getLogger(__name__)


class DetectionTask(Task):
    """Detection task with optional ByteTrack tracking.
    
    This task wraps a VisionPipeline and optionally applies ByteTrack
    to filter flickering detections and stabilize bounding boxes.
    
    The tracker state is maintained in memory and reset when:
    - task.reset() is called (region/session switch)
    - The task is re-created
    
    Pipeline remains stateless - tracker state belongs to Task layer.
    """

    def __init__(
        self,
        pipeline: VisionPipeline,
        *,
        tracker_config: Optional["TrackerConfig"] = None,
        smoothing_config: Optional["SmoothingConfig"] = None,
        done_condition: Optional[DoneCondition] = None,
        stats: Optional[TaskStats] = None,
        name: str = "detection",
    ):
        super().__init__(
            pipeline,
            done_condition=done_condition or AlwaysFalseCondition(),
            stats=stats,
        )
        self._name = name
        self._frame_count = 0
        
        # Tracker (optional)
        self._tracker = None
        self._tracker_config = tracker_config
        self._min_hits = 1  # Default: no filtering
        self._smoothing_config = smoothing_config
        self._prev_detections: List[Detection] = []
        
        if tracker_config and tracker_config.enabled:
            self._tracker = self._create_tracker(tracker_config)
            self._min_hits = tracker_config.min_hits
            logger.info(
                f"ByteTrack enabled: track_thresh={tracker_config.track_thresh}, "
                f"min_hits={tracker_config.min_hits}, buffer={tracker_config.track_buffer}"
            )
        
        if smoothing_config and smoothing_config.enabled:
            logger.info(
                "Temporal smoothing enabled: alpha=%.3f, min_iou=%.2f, max_center_jump=%.1f",
                smoothing_config.alpha,
                smoothing_config.min_iou,
                smoothing_config.max_center_jump,
            )

    @property
    def name(self) -> str:
        return self._name

    def _create_tracker(self, config: "TrackerConfig"):
        """Create ByteTracker instance from config."""
        try:
            from boxmot import ByteTrack
        except ImportError:
            raise ImportError(
                "boxmot is required for tracking. Install with: pip install boxmot"
            )
        
        return ByteTrack(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            frame_rate=config.frame_rate,
        )

    def run_iteration(self, image: np.ndarray) -> TaskIterationResult:
        """Process a single frame.
        
        1. Run pipeline to get raw detections
        2. Apply tracking (optional) to filter and stabilize
        3. Record stats and check done condition
        """
        # 1. Run pipeline (stateless)
        pipeline_result = self.pipeline.run(image)
        self._frame_count += 1
        
        raw_detections = pipeline_result.detections
        raw_count = len(raw_detections)
        
        # 2. Apply tracking (optional)
        confirmed_count = 0
        if self._tracker is not None:
            detections, confirmed_count = self._apply_tracking(
                raw_detections, image
            )
        else:
            detections = raw_detections
            confirmed_count = raw_count

        # 2.1 Lightweight temporal smoothing to reduce jitter
        if self._smoothing_config and self._smoothing_config.enabled:
            detections = self._apply_smoothing(detections)
        
        # 3. Build result
        result = TaskIterationResult(
            detections=detections,
            is_done=False,
            metadata={
                "pipeline_time_ms": pipeline_result.processing_time_ms,
                "raw_detection_count": raw_count,
                "detection_count": len(detections),
                "confirmed_count": confirmed_count,
                "frame_in_task": self._frame_count,
                "tracker_enabled": self._tracker is not None,
                "smoothing_enabled": bool(self._smoothing_config and self._smoothing_config.enabled),
            },
        )
        
        # 4. Record stats and check done
        self.stats.record(result)
        result.is_done = self.done_condition.check(self.stats, result)
        
        return result

    def _apply_tracking(
        self, 
        detections: List[Detection], 
        image: np.ndarray
    ) -> Tuple[List[Detection], int]:
        """Apply ByteTrack to detections.
        
        Args:
            detections: Raw detections from pipeline
            image: Original frame array
            
        Returns:
            Tuple of (filtered detections, confirmed count)
        """
        if not detections:
            # Still update tracker with empty detections
            empty_dets = np.empty((0, 6))
            self._tracker.update(empty_dets, image)
            return [], 0
        
        # Convert detections to numpy array: [x1, y1, x2, y2, conf, class_id]
        dets_array = np.array([
            [
                d.bbox.x1,
                d.bbox.y1,
                d.bbox.x2,
                d.bbox.y2,
                d.confidence,
                0,  # class_id (single class for now)
            ]
            for d in detections
        ])
        
        # Update tracker
        tracks = self._tracker.update(dets_array, image)
        
        # Convert tracks back to Detection objects
        # Filter by min_hits (only keep confirmed tracks)
        tracked_detections = []
        confirmed_count = 0
        
        if len(tracks) > 0:
            for track in tracks:
                # boxmot BYTETracker output format: [x1, y1, x2, y2, id, conf, cls, idx]
                # or may vary by version, handle common formats
                if len(track) >= 6:
                    x1, y1, x2, y2, track_id = track[:5]
                    conf = track[5] if len(track) > 5 else 1.0
                else:
                    continue
                
                # Find original detection to get object_type
                obj_type = self._find_original_type(x1, y1, x2, y2, detections)
                
                # Create tracked detection with track_id
                det = Detection(
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                    object_type=obj_type,
                    confidence=float(conf),
                    detection_id=f"track_{int(track_id)}",
                )
                tracked_detections.append(det)
                confirmed_count += 1
        
        logger.debug(
            f"Tracking: {len(detections)} raw -> {len(tracked_detections)} tracked "
            f"(frame {self._frame_count})"
        )
        
        return tracked_detections, confirmed_count

    def _apply_smoothing(self, detections: List[Detection]) -> List[Detection]:
        """Apply exponential smoothing on bounding boxes to reduce jitter.
        
        Matches current detections to previous smoothed results by IoU
        (and optional center distance) and blends boxes using alpha.
        """
        if not detections:
            self._prev_detections = []
            return []

        smoothed: List[Detection] = []
        used_prev: set[int] = set()

        for det in detections:
            match_idx = self._find_smoothing_match(det, used_prev)
            detection_id = det.detection_id

            if match_idx is not None:
                prev_det = self._prev_detections[match_idx]
                blended_bbox = self._blend_bbox(prev_det.bbox, det.bbox)
                # Prefer stable ID when available
                detection_id = detection_id or prev_det.detection_id
                used_prev.add(match_idx)
            else:
                blended_bbox = det.bbox

            smoothed.append(
                Detection(
                    bbox=blended_bbox,
                    object_type=det.object_type,
                    confidence=det.confidence,
                    detection_id=detection_id,
                )
            )

        # Update history for next frame
        self._prev_detections = smoothed
        return smoothed

    def _find_smoothing_match(
        self,
        det: Detection,
        used_prev: set[int],
    ) -> Optional[int]:
        """Find best previous detection to blend with current one."""
        if not self._prev_detections or not self._smoothing_config:
            return None

        best_idx: Optional[int] = None
        best_iou = 0.0

        for idx, prev_det in enumerate(self._prev_detections):
            if idx in used_prev:
                continue
            if det.object_type != prev_det.object_type:
                continue

            iou = self._calculate_iou(
                det.bbox.to_xyxy(),
                prev_det.bbox.to_xyxy(),
            )

            center_jump = self._center_distance(det.bbox, prev_det.bbox)

            # Require either reasonable IoU or small jump
            if (
                iou < self._smoothing_config.min_iou
                and center_jump > self._smoothing_config.max_center_jump
            ):
                continue

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        return best_idx

    def _blend_bbox(
        self,
        prev_box: BoundingBox,
        new_box: BoundingBox,
    ) -> BoundingBox:
        """Blend two boxes using exponential smoothing."""
        alpha = max(0.0, min(1.0, self._smoothing_config.alpha if self._smoothing_config else 0.0))

        def _lerp(a: float, b: float) -> float:
            return (1.0 - alpha) * a + alpha * b

        return BoundingBox(
            x1=_lerp(prev_box.x1, new_box.x1),
            y1=_lerp(prev_box.y1, new_box.y1),
            x2=_lerp(prev_box.x2, new_box.x2),
            y2=_lerp(prev_box.y2, new_box.y2),
        )

    def _center_distance(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Euclidean distance between two box centers."""
        c1x, c1y = box1.center
        c2x, c2y = box2.center
        return math.hypot(c1x - c2x, c1y - c2y)

    def _find_original_type(
        self, 
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float,
        detections: List[Detection]
    ) -> str:
        """Find the object_type from original detections by matching bbox.
        
        Uses IoU to find the best matching original detection.
        """
        if not detections:
            return "unknown"
        
        best_iou = 0.0
        best_type = detections[0].object_type  # Fallback
        
        for det in detections:
            iou = self._calculate_iou(
                (x1, y1, x2, y2),
                (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
            )
            if iou > best_iou:
                best_iou = iou
                best_type = det.object_type
        
        return best_type

    def _calculate_iou(
        self, 
        box1: Tuple[float, float, float, float], 
        box2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def reset(self) -> None:
        """Reset task state for new session/region."""
        self._frame_count = 0
        self.stats.reset()
        self.done_condition.reset()
        self._prev_detections = []
        
        # Recreate tracker to clear all track history
        if self._tracker_config and self._tracker_config.enabled:
            self._tracker = self._create_tracker(self._tracker_config)
            logger.info("Tracker reset: cleared all track history")
