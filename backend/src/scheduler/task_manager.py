"""Main loop scheduler for the detection system."""

import logging
import signal
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol

import cv2
import numpy as np

from src.core.camera.base import CameraBase
from src.core.vision.pipeline import VisionPipeline
from src.core.vision.types import Detection
from src.config import VideoStreamConfig
from src.storage.interfaces import Database, DetectionRecord, ImageStorage, SessionRecord
from src.scheduler.storage_saver import StorageSaver, StorageSaverConfig
from src.scheduler.tasks import DetectionTask, Task

logger = logging.getLogger(__name__)


@dataclass
class TaskManagerConfig:
    """Configuration for TaskManager.
    
    Attributes:
        loop_delay_ms: Delay between frames in milliseconds.
        max_errors: Maximum consecutive errors before stopping.
        save_images: Whether to persist raw images and detection records.
        save_annotated: Whether to save annotated images.
        show_preview: Show real-time preview window with OpenCV.
        async_storage: Use async storage with thread pool.
        storage_workers: Number of storage worker threads.
        max_pending_saves: Max pending save operations before blocking.
        storage_retry_count: Number of retries for failed storage ops.
        annotation_color_map: Colors for different object types.
    """
    loop_delay_ms: int = 100
    max_errors: int = 10
    save_images: bool = True
    save_annotated: bool = True
    show_preview: bool = True  # Show real-time detection preview
    async_storage: bool = True
    storage_workers: int = 4
    max_pending_saves: int = 100
    storage_retry_count: int = 3
    annotation_color_map: dict = field(default_factory=lambda: {
        "hair": (255, 0, 0),        # Blue (BGR
        "black_spot": (0, 0, 255),  # Red
        "yellow_spot": (0, 255, 255),  # Yellow
        "unknown": (255, 0, 0),  # Blue (BGR
    })


class DetectionEventPublisher(Protocol):
    """Protocol for publishing detection events."""

    def publish(self, payload: dict) -> None:
        ...


class FramePublisher(Protocol):
    """Protocol for publishing encoded frames."""

    def publish(self, frame_bytes: bytes, frame_id: str, timestamp: str) -> None:
        ...


class TaskManager:
    """Main loop scheduler for the detection system.
    
    Manages the continuous cycle of:
    1. Capture image from camera
    2. Run detection pipeline
    3. Save image to storage (async)
    4. Save detection records to database (async)
    
    Features:
    - Async storage operations via thread pool (non-blocking)
    - Retry logic for failed storage operations
    - Graceful shutdown with pending operation completion
    
    Example:
        >>> manager = TaskManager(
        ...     camera=camera,
        ...     pipeline=pipeline,
        ...     image_storage=storage,
        ...     database=db,
        ... )
        >>> manager.start()  # Blocks until stopped
    """

    def __init__(
        self,
        camera: CameraBase,
        pipeline: VisionPipeline,
        image_storage: ImageStorage,
        database: Database,
        task: Optional[Task] = None,
        config: Optional[TaskManagerConfig] = None,
        event_publisher: Optional[DetectionEventPublisher] = None,
        frame_publisher: Optional[FramePublisher] = None,
        video_stream_config: Optional[VideoStreamConfig] = None,
        register_signals: bool = True,
    ):
        """Initialize TaskManager.
        
        Args:
            camera: Camera instance for image capture.
            pipeline: Vision pipeline for detection.
            image_storage: Storage for images.
            database: Database for detection records.
            config: Optional configuration.
            event_publisher: Optional publisher for detection events.
            frame_publisher: Optional publisher for encoded frames.
            video_stream_config: Optional video streaming configuration.
        """
        self.camera = camera
        self.pipeline = pipeline
        self.image_storage = image_storage
        self.database = database
        self.task: Task = task or DetectionTask(pipeline=self.pipeline)
        self.config = config or TaskManagerConfig()
        self.event_publisher = event_publisher
        self.frame_publisher = frame_publisher
        self.video_stream_config = video_stream_config
        self.storage_saver = StorageSaver(
            image_storage=self.image_storage,
            database=self.database,
            config=StorageSaverConfig(
                save_images=self.config.save_images,
                save_annotated=self.config.save_annotated,
                async_storage=self.config.async_storage,
                storage_workers=self.config.storage_workers,
                max_pending_saves=self.config.max_pending_saves,
                storage_retry_count=self.config.storage_retry_count,
            ),
            event_callback=self._publish_event,
        )
        
        # State
        self._running = False
        self._error_count = 0
        self._frame_count = 0
        self._total_detections = 0
        self._session_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        self._last_frame_push_time: float = 0.0
        
        # Register signal handlers for graceful shutdown when allowed
        if register_signals and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            logger.debug("Skipping signal registration (not main thread or disabled)")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self.stop()

    @property
    def is_running(self) -> bool:
        """Check if task manager is running."""
        return self._running

    @property
    def session_id(self) -> Optional[str]:
        """Current session ID."""
        return self._session_id

    @property
    def frame_count(self) -> int:
        """Number of frames processed in current session."""
        return self._frame_count

    @property
    def total_detections(self) -> int:
        """Total detections in current session."""
        return self._total_detections

    def start(self):
        """Start the main processing loop.
        
        This method blocks until stop() is called or max errors reached.
        """
        logger.info("Starting TaskManager...")
        
        # Open camera
        if not self.camera.open():
            raise RuntimeError("Failed to open camera")
        
        # Initialize storage saver (starts async executor if configured)
        self.storage_saver.start()
        
        # Create new session
        self._session_id = str(uuid.uuid4())
        self._start_time = datetime.now()
        self._running = True
        self._frame_count = 0
        self._total_detections = 0
        self._error_count = 0
        self._last_frame_push_time = 0.0

        # Reset task state for new session
        try:
            self.task.reset()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to reset task state: %s", exc)
        
        # Create session in database
        session = SessionRecord(
            id=self._session_id,
            start_time=self._start_time,
            status="running",
        )
        self.database.create_session(session)
        
        logger.info(f"Session started: {self._session_id}")
        
        try:
            self._main_loop()
        finally:
            self._cleanup()

    def stop(self):
        """Stop the main processing loop."""
        logger.info("Stopping TaskManager...")
        self._running = False

    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            loop_start = time.time()
            
            try:
                self._process_frame()
                self._error_count = 0  # Reset on success
                
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Error processing frame (attempt {self._error_count}): {e}",
                    exc_info=True
                )
                
                if self._error_count >= self.config.max_errors:
                    logger.critical(
                        f"Max consecutive errors ({self.config.max_errors}) "
                        "reached, stopping"
                    )
                    self.stop()
                    break
            
            # Calculate remaining delay
            elapsed_ms = (time.time() - loop_start) * 1000
            remaining_ms = self.config.loop_delay_ms - elapsed_ms
            
            if remaining_ms > 0:
                time.sleep(remaining_ms / 1000.0)

    def _process_frame(self):
        """Process a single frame."""
        timestamp = datetime.now()
        
        # 1. Capture image
        capture_start = time.time()
        image = self.camera.capture()
        capture_time = (time.time() - capture_start) * 1000
        
        self._frame_count += 1
        logger.debug(
            f"Frame {self._frame_count}: captured in {capture_time:.1f}ms"
        )
        
        # 2. Run task iteration (wraps pipeline)
        result = self.task.run_iteration(image)
        
        detection_count = len(result.detections)
        self._total_detections += detection_count
        
        logger.info(
            f"Frame {self._frame_count}: "
            f"{detection_count} detections, "
            f"{result.metadata.get('pipeline_time_ms', 0):.1f}ms processing"
        )
        
        # 2.5 Real-time preview with OpenCV (if enabled)
        if self.config.show_preview:
            if result.detections:
                preview_image = self._draw_detections(image, result.detections)
            else:
                preview_image = image.copy()
            
            # Resize for display if image is too large
            max_display_width = 1280
            h, w = preview_image.shape[:2]
            if w > max_display_width:
                scale = max_display_width / w
                preview_image = cv2.resize(
                    preview_image,
                    (int(w * scale), int(h * scale))
                )
            
            cv2.imshow("Detection Preview", preview_image)
            key = cv2.waitKey(1) & 0xFF
            # Press 'q' to quit
            if key == ord('q'):
                logger.info("User pressed 'q', stopping...")
                self.stop()
        
        # 3. Generate image path
        date_path = timestamp.strftime("%Y/%m/%d")
        time_str = timestamp.strftime("%H%M%S_%f")[:-3]  # HHMMSSmmm
        image_name = f"{time_str}_{self._frame_count:06d}.jpg"
        image_path = f"{date_path}/{image_name}"
        
        # 4. Prepare annotated image if needed
        annotated_image = None
        annotated_path = None
        if self.config.save_annotated and result.detections:
            annotated_image = self._draw_detections(image, result.detections)
            annotated_path = image_path.replace(".jpg", "_annotated.jpg")

        # Frame for streaming (draw boxes if present, otherwise raw)
        frame_for_stream = None
        if result.detections:
            frame_for_stream = (
                annotated_image
                if annotated_image is not None
                else self._draw_detections(image, result.detections)
            )
        else:
            frame_for_stream = image
        
        # 5. Save images/detections
        self.storage_saver.save(
            image=image,
            image_path=image_path,
            detections=result.detections,
            timestamp=timestamp,
            annotated_image=annotated_image,
            annotated_path=annotated_path,
            session_id=self._session_id,
        )

        # 5.5 Publish frame to stream (honoring fps cap)
        self._publish_frame(
            frame_for_stream,
            frame_id=f"{self._session_id}-{self._frame_count:06d}",
            timestamp=timestamp,
        )

        # 5.6 Handle task completion
        if result.is_done:
            logger.info(
                "Task '%s' reported completion after %s frames in session %s",
                self.task.name,
                self._frame_count,
                self._session_id,
            )
            # Stop main loop; future workflow can switch tasks here.
            self.stop()
        
        # 6. Update session periodically
        if self._frame_count % 100 == 0:
            self._update_session()
        
        # 7. Clean up completed futures
        self.storage_saver.cleanup_futures()

    def _publish_event(
        self,
        records: List[DetectionRecord],
        image_path: str,
        annotated_path: Optional[str],
        timestamp: datetime,
    ):
        """Publish detection event to external subscribers (if configured).
        
        NOTE: bbox data is NOT included in the event payload because:
        - Bounding boxes are already drawn on the MJPEG video stream
        - This keeps WebSocket messages lightweight (JSON only)
        - Historical bbox data can be queried via REST API if needed
        """
        if not self.event_publisher or not records:
            return

        # Count detections by type for statistics
        by_type: dict[str, int] = {}
        for record in records:
            by_type[record.object_type] = by_type.get(record.object_type, 0) + 1

        payload = {
            "type": "detection",
            "session_id": self._session_id,
            "frame": self._frame_count,
            "timestamp": timestamp.isoformat() + "Z",
            "image_path": image_path,
            "annotated_path": annotated_path,
            # Lightweight detection summary (no bbox - already drawn on frame)
            "detection_count": len(records),
            "by_type": by_type,
            "total_detections": self._total_detections,
        }

        try:
            self.event_publisher.publish(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish detection event: %s", exc)

    def _publish_frame(
        self,
        frame: Optional[np.ndarray],
        frame_id: str,
        timestamp: datetime,
    ) -> None:
        """Publish encoded frame for MJPEG streaming."""
        if (
            frame is None
            or not self.frame_publisher
            or not self.video_stream_config
            or not self.video_stream_config.enabled
        ):
            return

        now = time.time()
        min_interval = 1.0 / self.video_stream_config.fps_limit if self.video_stream_config.fps_limit > 0 else 0
        if min_interval and (now - self._last_frame_push_time) < min_interval:
            return

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.video_stream_config.jpeg_quality)]
        success, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            logger.warning("Failed to encode frame for streaming (id=%s)", frame_id)
            return

        try:
            self.frame_publisher.publish(
                buffer.tobytes(),
                frame_id=frame_id,
                timestamp=timestamp.isoformat() + "Z",
            )
            self._last_frame_push_time = now
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to publish frame: %s", exc)


    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw detection boxes on image.
        
        Args:
            image: Original image.
            detections: List of detections.
            
        Returns:
            Annotated image copy.
        """
        result = image.copy()
        
        for det in detections:
            # object_type is a string, not an enum
            obj_type = det.object_type if isinstance(det.object_type, str) else det.object_type.value
            color = self.config.annotation_color_map.get(
                obj_type,
                (255, 0, 0)
            )
            
            # Draw bounding box
            pt1 = (int(det.bbox.x1), int(det.bbox.y1))
            pt2 = (int(det.bbox.x2), int(det.bbox.y2))
            cv2.rectangle(result, pt1, pt2, color, 2)
            
            # Draw label
            label = f"{obj_type}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            # Label background
            cv2.rectangle(
                result,
                (pt1[0], pt1[1] - label_size[1] - 10),
                (pt1[0] + label_size[0], pt1[1]),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                result,
                label,
                (pt1[0], pt1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return result

    def _update_session(self):
        """Update session record in database."""
        session = SessionRecord(
            id=self._session_id,
            start_time=self._start_time,
            total_frames=self._frame_count,
            total_detections=self._total_detections,
            status="running",
        )
        self.database.update_session(session)

    def _cleanup(self):
        """Cleanup resources and finalize session."""
        logger.info("Cleaning up TaskManager...")
        
        # Close OpenCV windows if preview was enabled
        if self.config.show_preview:
            cv2.destroyAllWindows()
        
        # Close camera
        try:
            self.camera.close()
        except Exception as e:
            logger.warning(f"Error closing camera: {e}")
        
        # Wait for pending storage operations
        self.storage_saver.shutdown()
        
        # Finalize session
        if self._session_id:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds()
            
            session = SessionRecord(
                id=self._session_id,
                start_time=self._start_time,
                end_time=end_time,
                total_frames=self._frame_count,
                total_detections=self._total_detections,
                status="completed" if self._error_count < self.config.max_errors else "failed",
            )
            
            try:
                self.database.update_session(session)
            except Exception as e:
                logger.error(f"Failed to update session: {e}")
            
            fps_str = f"{self._frame_count / duration:.1f}" if duration > 0 else "N/A"
            logger.info(
                f"Session ended: {self._session_id}\n"
                f"  Duration: {duration:.1f}s\n"
                f"  Frames: {self._frame_count}\n"
                f"  Detections: {self._total_detections}\n"
                f"  Storage errors: {self.storage_saver.storage_errors}\n"
                f"  FPS: {fps_str}"
            )
