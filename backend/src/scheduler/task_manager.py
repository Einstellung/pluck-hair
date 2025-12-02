"""Main loop scheduler for the detection system."""

import logging
import queue
import signal
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np

from src.core.camera.base import CameraBase
from src.core.vision.pipeline import VisionPipeline
from src.core.vision.types import Detection
from src.storage.interfaces import Database, DetectionRecord, ImageStorage, SessionRecord

logger = logging.getLogger(__name__)


@dataclass
class TaskManagerConfig:
    """Configuration for TaskManager.
    
    Attributes:
        loop_delay_ms: Delay between frames in milliseconds.
        max_errors: Maximum consecutive errors before stopping.
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
        config: Optional[TaskManagerConfig] = None,
        register_signals: bool = True,
    ):
        """Initialize TaskManager.
        
        Args:
            camera: Camera instance for image capture.
            pipeline: Vision pipeline for detection.
            image_storage: Storage for images.
            database: Database for detection records.
            config: Optional configuration.
        """
        self.camera = camera
        self.pipeline = pipeline
        self.image_storage = image_storage
        self.database = database
        self.config = config or TaskManagerConfig()
        
        # State
        self._running = False
        self._error_count = 0
        self._frame_count = 0
        self._total_detections = 0
        self._session_id: Optional[str] = None
        self._start_time: Optional[datetime] = None
        
        # Async storage
        self._storage_executor: Optional[ThreadPoolExecutor] = None
        self._pending_futures: List[Future] = []
        self._storage_errors = 0
        self._storage_lock = threading.Lock()
        
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
        
        # Initialize async storage executor
        if self.config.async_storage:
            self._storage_executor = ThreadPoolExecutor(
                max_workers=self.config.storage_workers,
                thread_name_prefix="storage"
            )
            logger.info(
                f"Async storage enabled with {self.config.storage_workers} workers"
            )
        
        # Create new session
        self._session_id = str(uuid.uuid4())
        self._start_time = datetime.utcnow()
        self._running = True
        self._frame_count = 0
        self._total_detections = 0
        self._error_count = 0
        self._storage_errors = 0
        self._pending_futures = []
        
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
        timestamp = datetime.utcnow()
        
        # 1. Capture image
        capture_start = time.time()
        image = self.camera.capture()
        capture_time = (time.time() - capture_start) * 1000
        
        self._frame_count += 1
        logger.debug(
            f"Frame {self._frame_count}: captured in {capture_time:.1f}ms"
        )
        
        # 2. Run detection pipeline
        result = self.pipeline.run(image)
        
        detection_count = len(result.detections)
        self._total_detections += detection_count
        
        logger.info(
            f"Frame {self._frame_count}: "
            f"{detection_count} detections, "
            f"{result.processing_time_ms:.1f}ms processing"
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
        
        # 5. Save images and detection records (async or sync)
        if self.config.async_storage and self._storage_executor:
            self._save_async(
                image, image_path,
                result.detections, timestamp,
                annotated_image, annotated_path
            )
        else:
            self._save_sync(
                image, image_path,
                result.detections, timestamp,
                annotated_image, annotated_path
            )
        
        # 6. Update session periodically
        if self._frame_count % 100 == 0:
            self._update_session()
        
        # 7. Clean up completed futures
        self._cleanup_futures()

    def _save_sync(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
    ):
        """Save results synchronously (blocking)."""
        # Save original image
        full_path = self.image_storage.save(image, image_path)
        
        # Save detection records
        if detections:
            records = [
                self._to_detection_record(det, full_path, timestamp)
                for det in detections
            ]
            self.database.save_detections_batch(records)
        
        # Save annotated image
        if annotated_image is not None and annotated_path:
            self.image_storage.save(annotated_image, annotated_path)

    def _save_async(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
    ):
        """Save results asynchronously (non-blocking)."""
        # Check if we have too many pending operations
        with self._storage_lock:
            pending_count = len([f for f in self._pending_futures if not f.done()])
            
        if pending_count >= self.config.max_pending_saves:
            logger.warning(
                f"Storage queue full ({pending_count} pending), "
                "falling back to sync save"
            )
            self._save_sync(
                image, image_path, detections, timestamp,
                annotated_image, annotated_path
            )
            return
        
        # Submit async save task
        future = self._storage_executor.submit(
            self._save_with_retry,
            image.copy(),  # Copy to avoid race conditions
            image_path,
            detections,
            timestamp,
            annotated_image.copy() if annotated_image is not None else None,
            annotated_path,
        )
        
        with self._storage_lock:
            self._pending_futures.append(future)

    def _save_with_retry(
        self,
        image: np.ndarray,
        image_path: str,
        detections: List[Detection],
        timestamp: datetime,
        annotated_image: Optional[np.ndarray],
        annotated_path: Optional[str],
    ):
        """Save with retry logic for resilience."""
        last_error = None
        
        for attempt in range(self.config.storage_retry_count):
            try:
                # Save original image
                full_path = self.image_storage.save(image, image_path)
                
                # Save detection records
                if detections:
                    records = [
                        self._to_detection_record(det, full_path, timestamp)
                        for det in detections
                    ]
                    self.database.save_detections_batch(records)
                
                # Save annotated image
                if annotated_image is not None and annotated_path:
                    self.image_storage.save(annotated_image, annotated_path)
                
                # Success
                return
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Storage attempt {attempt + 1}/{self.config.storage_retry_count} "
                    f"failed: {e}"
                )
                if attempt < self.config.storage_retry_count - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        with self._storage_lock:
            self._storage_errors += 1
        logger.error(
            f"Storage failed after {self.config.storage_retry_count} attempts: "
            f"{last_error}"
        )

    def _cleanup_futures(self):
        """Clean up completed futures and check for errors."""
        with self._storage_lock:
            # Remove completed futures
            still_pending = []
            for future in self._pending_futures:
                if future.done():
                    # Check if there was an exception
                    try:
                        future.result()  # Will raise if there was an error
                    except Exception as e:
                        logger.error(f"Async storage task failed: {e}")
                else:
                    still_pending.append(future)
            
            self._pending_futures = still_pending

    def _to_detection_record(
        self,
        detection: Detection,
        image_path: str,
        timestamp: datetime,
    ) -> DetectionRecord:
        """Convert Detection to DetectionRecord."""
        # object_type is a string, not an enum
        obj_type = detection.object_type if isinstance(detection.object_type, str) else detection.object_type.value
        return DetectionRecord(
            id=str(uuid.uuid4()),
            image_path=image_path,
            bbox_x1=detection.bbox.x1,
            bbox_y1=detection.bbox.y1,
            bbox_x2=detection.bbox.x2,
            bbox_y2=detection.bbox.y2,
            object_type=obj_type,
            confidence=detection.confidence,
            created_at=timestamp,
            session_id=self._session_id,
        )

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
        if self._storage_executor:
            pending_count = len([f for f in self._pending_futures if not f.done()])
            if pending_count > 0:
                logger.info(f"Waiting for {pending_count} pending storage operations...")
            
            # Wait for all pending futures with timeout
            for future in self._pending_futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logger.error(f"Pending storage task failed: {e}")
            
            # Shutdown executor
            self._storage_executor.shutdown(wait=True)
            self._storage_executor = None
            logger.info("Storage executor shut down")
        
        # Finalize session
        if self._session_id:
            end_time = datetime.utcnow()
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
                f"  Storage errors: {self._storage_errors}\n"
                f"  FPS: {fps_str}"
            )

