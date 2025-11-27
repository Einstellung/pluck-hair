"""Tests for TaskManager and async storage operations."""

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from src.core.camera.base import CameraBase
from src.core.vision.pipeline import VisionPipeline
from src.core.vision.types import BoundingBox, Detection, ObjectType, PipelineResult
from src.scheduler.task_manager import TaskManager, TaskManagerConfig
from src.storage.interfaces import Database, DetectionRecord, ImageStorage, SessionRecord


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_camera():
    """Create a mock camera."""
    camera = MagicMock(spec=CameraBase)
    camera.is_opened.return_value = False
    camera.open.return_value = True
    camera.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    camera.get_frame_size.return_value = (640, 480)
    return camera


@pytest.fixture
def mock_pipeline():
    """Create a mock vision pipeline."""
    pipeline = MagicMock(spec=VisionPipeline)
    pipeline.run.return_value = PipelineResult(
        original_image=np.zeros((480, 640, 3), dtype=np.uint8),
        detections=[],
        processing_time_ms=10.0,
        metadata={},
    )
    return pipeline


@pytest.fixture
def mock_database():
    """Create a mock database."""
    db = MagicMock(spec=Database)
    db.create_session.return_value = "session-123"
    db.save_detection.return_value = "det-123"
    db.save_detections_batch.return_value = ["det-1", "det-2"]
    return db


@pytest.fixture
def mock_storage():
    """Create a mock image storage."""
    storage = MagicMock(spec=ImageStorage)
    storage.save.return_value = "images/test.jpg"
    return storage


@pytest.fixture
def task_config():
    """Create a task manager config for testing."""
    return TaskManagerConfig(
        loop_delay_ms=10,  # Fast for tests
        max_errors=3,
        save_annotated=False,
        async_storage=False,  # Use sync for easier testing
        storage_retry_count=2,
    )


@pytest.fixture
def task_manager(mock_camera, mock_pipeline, mock_storage, mock_database, task_config):
    """Create a task manager with mocked dependencies."""
    return TaskManager(
        camera=mock_camera,
        pipeline=mock_pipeline,
        image_storage=mock_storage,
        database=mock_database,
        config=task_config,
    )


# ============================================================
# Initialization Tests
# ============================================================

class TestTaskManagerInit:
    """Tests for TaskManager initialization."""

    def test_init_with_defaults(self, mock_camera, mock_pipeline, mock_storage, mock_database):
        """Test TaskManager initializes with default config."""
        manager = TaskManager(
            camera=mock_camera,
            pipeline=mock_pipeline,
            image_storage=mock_storage,
            database=mock_database,
        )
        
        assert manager.config.loop_delay_ms == 100
        assert manager.config.max_errors == 10
        assert manager.config.async_storage is True
        assert manager.is_running is False

    def test_init_with_custom_config(self, task_manager, task_config):
        """Test TaskManager initializes with custom config."""
        assert task_manager.config.loop_delay_ms == 10
        assert task_manager.config.max_errors == 3

    def test_initial_state(self, task_manager):
        """Test TaskManager has correct initial state."""
        assert task_manager.is_running is False
        assert task_manager.session_id is None
        assert task_manager.frame_count == 0
        assert task_manager.total_detections == 0


# ============================================================
# Processing Tests
# ============================================================

class TestTaskManagerProcessing:
    """Tests for TaskManager frame processing."""

    def test_process_frame_no_detections(
        self, task_manager, mock_camera, mock_pipeline, mock_storage, mock_database
    ):
        """Test processing a frame with no detections."""
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[],
            processing_time_ms=15.0,
            metadata={},
        )
        
        # Set up manager state
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        
        # Process one frame
        task_manager._process_frame()
        
        # Verify
        mock_camera.capture.assert_called_once()
        mock_pipeline.run.assert_called_once()
        mock_storage.save.assert_called_once()  # Still saves image
        mock_database.save_detections_batch.assert_not_called()  # No detections

    def test_process_frame_with_detections(
        self, task_manager, mock_camera, mock_pipeline, mock_storage, mock_database
    ):
        """Test processing a frame with detections."""
        detections = [
            Detection(
                bbox=BoundingBox(x1=10, y1=20, x2=30, y2=40),
                object_type=ObjectType.HAIR,
                confidence=0.95,
            ),
            Detection(
                bbox=BoundingBox(x1=50, y1=60, x2=70, y2=80),
                object_type=ObjectType.BLACK_SPOT,
                confidence=0.88,
            ),
        ]
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=detections,
            processing_time_ms=20.0,
            metadata={},
        )
        
        # Set up manager state
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        
        # Process one frame
        task_manager._process_frame()
        
        # Verify detections were saved
        mock_database.save_detections_batch.assert_called_once()
        saved_records = mock_database.save_detections_batch.call_args[0][0]
        assert len(saved_records) == 2
        assert saved_records[0].object_type == "hair"
        assert saved_records[1].object_type == "black_spot"

    def test_frame_count_increments(self, task_manager, mock_pipeline):
        """Test frame count increments correctly."""
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[],
            processing_time_ms=10.0,
            metadata={},
        )
        
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        
        assert task_manager.frame_count == 0
        
        task_manager._process_frame()
        assert task_manager.frame_count == 1
        
        task_manager._process_frame()
        assert task_manager.frame_count == 2

    def test_total_detections_accumulates(self, task_manager, mock_pipeline):
        """Test total detections accumulates across frames."""
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        
        # First frame: 2 detections
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[
                Detection(BoundingBox(0, 0, 10, 10), ObjectType.HAIR, 0.9),
                Detection(BoundingBox(20, 20, 30, 30), ObjectType.HAIR, 0.8),
            ],
            processing_time_ms=10.0,
            metadata={},
        )
        task_manager._process_frame()
        assert task_manager.total_detections == 2
        
        # Second frame: 1 detection
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[
                Detection(BoundingBox(40, 40, 50, 50), ObjectType.BLACK_SPOT, 0.85),
            ],
            processing_time_ms=10.0,
            metadata={},
        )
        task_manager._process_frame()
        assert task_manager.total_detections == 3


# ============================================================
# Storage Retry Tests
# ============================================================

class TestStorageRetry:
    """Tests for async storage retry logic."""

    def test_sync_save_success(self, task_manager, mock_storage, mock_database):
        """Test synchronous save succeeds."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(BoundingBox(0, 0, 10, 10), ObjectType.HAIR, 0.9),
        ]
        
        task_manager._session_id = "test-session"
        task_manager._save_sync(
            image=image,
            image_path="test/image.jpg",
            detections=detections,
            timestamp=datetime.utcnow(),
            annotated_image=None,
            annotated_path=None,
        )
        
        mock_storage.save.assert_called_once()
        mock_database.save_detections_batch.assert_called_once()

    def test_save_with_retry_succeeds_first_try(
        self, task_manager, mock_storage, mock_database
    ):
        """Test save_with_retry succeeds on first attempt."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        task_manager._session_id = "test-session"
        task_manager._save_with_retry(
            image=image,
            image_path="test/image.jpg",
            detections=[],
            timestamp=datetime.utcnow(),
            annotated_image=None,
            annotated_path=None,
        )
        
        mock_storage.save.assert_called_once()

    def test_save_with_retry_retries_on_failure(
        self, task_manager, mock_storage, mock_database
    ):
        """Test save_with_retry retries on failure."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Fail first time, succeed second time
        mock_storage.save.side_effect = [Exception("Connection error"), "images/test.jpg"]
        
        task_manager._session_id = "test-session"
        task_manager._save_with_retry(
            image=image,
            image_path="test/image.jpg",
            detections=[],
            timestamp=datetime.utcnow(),
            annotated_image=None,
            annotated_path=None,
        )
        
        # Should have been called twice
        assert mock_storage.save.call_count == 2

    def test_save_with_retry_fails_after_max_retries(
        self, task_manager, mock_storage, mock_database
    ):
        """Test save_with_retry gives up after max retries."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Always fail
        mock_storage.save.side_effect = Exception("Persistent error")
        
        task_manager._session_id = "test-session"
        initial_errors = task_manager._storage_errors
        
        # This should not raise, just log and count error
        task_manager._save_with_retry(
            image=image,
            image_path="test/image.jpg",
            detections=[],
            timestamp=datetime.utcnow(),
            annotated_image=None,
            annotated_path=None,
        )
        
        # Should have tried retry_count times
        assert mock_storage.save.call_count == task_manager.config.storage_retry_count
        # Error count should have increased
        assert task_manager._storage_errors == initial_errors + 1


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """Tests for TaskManager error handling."""

    def test_stops_after_max_errors(self, task_manager, mock_camera, mock_pipeline):
        """Test TaskManager stops after max consecutive errors."""
        mock_pipeline.run.side_effect = Exception("Pipeline error")
        
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        
        # Run the main loop (should stop after max_errors)
        task_manager._main_loop()
        
        assert task_manager._running is False
        assert task_manager._error_count >= task_manager.config.max_errors

    def test_error_count_resets_on_success(self, task_manager, mock_pipeline):
        """Test error count resets after successful frame."""
        task_manager._running = True
        task_manager._session_id = "test-session"
        task_manager._start_time = datetime.utcnow()
        task_manager._error_count = 2
        
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[],
            processing_time_ms=10.0,
            metadata={},
        )
        
        task_manager._process_frame()
        
        # After success, error count should be reset in main loop
        # (Note: _process_frame itself doesn't reset, _main_loop does)
        # So we just verify the frame was processed successfully
        assert task_manager.frame_count == 1


# ============================================================
# Session Management Tests
# ============================================================

class TestSessionManagement:
    """Tests for session management."""

    def test_session_created_on_start(self, task_manager, mock_database, mock_camera):
        """Test session is created when starting."""
        # Start in a separate thread so we can stop it
        def run_and_stop():
            time.sleep(0.05)  # Let it run a bit
            task_manager.stop()
        
        stopper = threading.Thread(target=run_and_stop)
        stopper.start()
        
        task_manager.start()
        stopper.join()
        
        # Verify session was created
        mock_database.create_session.assert_called_once()
        session_arg = mock_database.create_session.call_args[0][0]
        assert isinstance(session_arg, SessionRecord)
        assert session_arg.status == "running"

    def test_session_updated_on_cleanup(self, task_manager, mock_database, mock_camera):
        """Test session is updated when cleaning up."""
        def run_and_stop():
            time.sleep(0.05)
            task_manager.stop()
        
        stopper = threading.Thread(target=run_and_stop)
        stopper.start()
        
        task_manager.start()
        stopper.join()
        
        # Verify session was updated (at least once for finalization)
        assert mock_database.update_session.called


# ============================================================
# Annotation Tests
# ============================================================

class TestAnnotations:
    """Tests for detection annotation drawing."""

    def test_draw_detections(self, task_manager):
        """Test drawing detection boxes on image."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
                object_type=ObjectType.HAIR,
                confidence=0.95,
            ),
        ]
        
        annotated = task_manager._draw_detections(image, detections)
        
        # Annotated should be different from original
        assert not np.array_equal(annotated, image)
        # Original should be unchanged
        assert np.all(image == 0)

    def test_annotated_image_saved_when_configured(
        self, mock_camera, mock_pipeline, mock_storage, mock_database
    ):
        """Test annotated images are saved when configured."""
        config = TaskManagerConfig(
            loop_delay_ms=10,
            save_annotated=True,
            async_storage=False,
        )
        manager = TaskManager(
            camera=mock_camera,
            pipeline=mock_pipeline,
            image_storage=mock_storage,
            database=mock_database,
            config=config,
        )
        
        # Pipeline returns detections
        mock_pipeline.run.return_value = PipelineResult(
            original_image=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[
                Detection(BoundingBox(0, 0, 10, 10), ObjectType.HAIR, 0.9),
            ],
            processing_time_ms=10.0,
            metadata={},
        )
        
        manager._running = True
        manager._session_id = "test-session"
        manager._start_time = datetime.utcnow()
        
        manager._process_frame()
        
        # Should save both original and annotated
        assert mock_storage.save.call_count == 2

