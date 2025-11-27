"""Tests for storage module."""

from datetime import datetime

import numpy as np
import pytest


class TestLocalStorage:
    """Tests for LocalStorage."""

    def test_save_and_load_image(self, local_storage, sample_image):
        """Test saving and loading an image."""
        path = "test/image.jpg"
        
        saved_path = local_storage.save(sample_image, path)
        assert local_storage.exists(saved_path)
        
        loaded = local_storage.load(saved_path)
        
        # JPEG compression changes the image slightly
        assert loaded.shape == sample_image.shape

    def test_save_bytes(self, local_storage):
        """Test saving raw bytes."""
        data = b"test data content"
        path = "test/file.txt"
        
        saved_path = local_storage.save_bytes(data, path, "text/plain")
        loaded = local_storage.load_bytes(saved_path)
        
        assert loaded == data

    def test_delete(self, local_storage, sample_image):
        """Test deleting a file."""
        path = "test/to_delete.jpg"
        local_storage.save(sample_image, path)
        
        assert local_storage.exists(local_storage.base_path / path)
        
        result = local_storage.delete(str(local_storage.base_path / path))
        
        assert result is True
        assert not local_storage.exists(str(local_storage.base_path / path))

    def test_delete_nonexistent(self, local_storage):
        """Test deleting non-existent file."""
        result = local_storage.delete("nonexistent/file.jpg")
        assert result is False

    def test_list_objects(self, local_storage, sample_image):
        """Test listing objects."""
        # Save some files
        local_storage.save(sample_image, "dir1/a.jpg")
        local_storage.save(sample_image, "dir1/b.jpg")
        local_storage.save(sample_image, "dir2/c.jpg")
        
        # List all
        all_files = local_storage.list_objects()
        assert len(all_files) == 3
        
        # List with prefix
        dir1_files = local_storage.list_objects("dir1")
        assert len(dir1_files) == 2


class TestSQLiteDatabase:
    """Tests for SQLiteDatabase."""

    def test_save_and_get_detection(self, sqlite_db):
        """Test saving and retrieving a detection."""
        from src.storage.interfaces import DetectionRecord
        
        record = DetectionRecord(
            image_path="test/image.jpg",
            bbox_x1=10.0,
            bbox_y1=20.0,
            bbox_x2=30.0,
            bbox_y2=40.0,
            object_type="hair",
            confidence=0.95,
        )
        
        record_id = sqlite_db.save_detection(record)
        assert record_id is not None
        
        loaded = sqlite_db.get_detection(record_id)
        
        assert loaded is not None
        assert loaded.image_path == "test/image.jpg"
        assert loaded.object_type == "hair"
        assert loaded.confidence == 0.95

    def test_save_detections_batch(self, sqlite_db):
        """Test batch saving detections."""
        from src.storage.interfaces import DetectionRecord
        
        records = [
            DetectionRecord(
                image_path=f"test/image_{i}.jpg",
                bbox_x1=float(i * 10),
                bbox_y1=float(i * 10),
                bbox_x2=float(i * 10 + 20),
                bbox_y2=float(i * 10 + 20),
                object_type="hair",
                confidence=0.9,
            )
            for i in range(5)
        ]
        
        ids = sqlite_db.save_detections_batch(records)
        
        assert len(ids) == 5
        
        # Verify all saved
        count = sqlite_db.count_detections()
        assert count == 5

    def test_query_detections(self, sqlite_db):
        """Test querying detections with filters."""
        from src.storage.interfaces import DetectionRecord
        
        # Save some records
        for obj_type in ["hair", "hair", "black_spot"]:
            record = DetectionRecord(
                image_path="test/image.jpg",
                bbox_x1=10.0, bbox_y1=20.0, bbox_x2=30.0, bbox_y2=40.0,
                object_type=obj_type,
                confidence=0.9,
            )
            sqlite_db.save_detection(record)
        
        # Query by type
        hair_detections = sqlite_db.query_detections(object_type="hair")
        assert len(hair_detections) == 2
        
        spot_detections = sqlite_db.query_detections(object_type="black_spot")
        assert len(spot_detections) == 1

    def test_session_operations(self, sqlite_db):
        """Test session CRUD operations."""
        from src.storage.interfaces import SessionRecord
        
        # Create session
        session = SessionRecord(
            status="running",
            total_frames=0,
            total_detections=0,
        )
        session_id = sqlite_db.create_session(session)
        assert session_id is not None
        
        # Get session
        loaded = sqlite_db.get_session(session_id)
        assert loaded is not None
        assert loaded.status == "running"
        
        # Update session
        session.id = session_id
        session.total_frames = 100
        session.total_detections = 50
        session.status = "completed"
        session.end_time = datetime.utcnow()
        
        result = sqlite_db.update_session(session)
        assert result is True
        
        # Verify update
        updated = sqlite_db.get_session(session_id)
        assert updated.total_frames == 100
        assert updated.status == "completed"

    def test_count_detections_with_filters(self, sqlite_db):
        """Test counting detections with various filters."""
        from src.storage.interfaces import DetectionRecord, SessionRecord
        
        # Create a session
        session = SessionRecord(status="running")
        session_id = sqlite_db.create_session(session)
        
        # Save detections with session
        for i in range(3):
            record = DetectionRecord(
                image_path="test/image.jpg",
                bbox_x1=10.0, bbox_y1=20.0, bbox_x2=30.0, bbox_y2=40.0,
                object_type="hair",
                confidence=0.9,
                session_id=session_id,
            )
            sqlite_db.save_detection(record)
        
        # Count by session
        count = sqlite_db.count_detections(session_id=session_id)
        assert count == 3
        
        # Count by type
        count = sqlite_db.count_detections(object_type="hair")
        assert count == 3


