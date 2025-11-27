"""Tests for API endpoints and dependency injection."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import AppConfig
from src.storage.interfaces import Database, DetectionRecord, ImageStorage, SessionRecord


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_config():
    """Create a mock AppConfig."""
    return AppConfig()


@pytest.fixture
def mock_database():
    """Create a mock database."""
    db = MagicMock(spec=Database)
    db.count_detections.return_value = 0
    db.query_detections.return_value = []
    db.get_detection.return_value = None
    return db


@pytest.fixture
def mock_storage():
    """Create a mock image storage."""
    storage = MagicMock(spec=ImageStorage)
    storage.list_objects.return_value = []
    return storage


@pytest.fixture
def client(mock_config, mock_database, mock_storage):
    """Create test client with mocked dependencies."""
    app = create_app(
        config=mock_config,
        database=mock_database,
        image_storage=mock_storage,
    )
    return TestClient(app)


@pytest.fixture
def client_no_deps():
    """Create test client without dependencies (for testing error cases)."""
    app = create_app()
    return TestClient(app)


# ============================================================
# Health Endpoint Tests
# ============================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check returns healthy."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_check(self, client):
        """Test liveness probe returns alive."""
        response = client.get("/api/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True
        assert "timestamp" in data

    def test_readiness_check_all_healthy(self, client, mock_database, mock_storage):
        """Test readiness check when all deps are healthy."""
        response = client.get("/api/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["database"] == "ok"
        assert data["storage"] == "ok"

    def test_readiness_check_database_error(self, client, mock_database):
        """Test readiness check when database fails."""
        mock_database.count_detections.side_effect = Exception("Connection failed")
        
        response = client.get("/api/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "error" in data["database"]

    def test_readiness_check_storage_error(self, client, mock_storage):
        """Test readiness check when storage fails."""
        mock_storage.list_objects.side_effect = Exception("MinIO unavailable")
        
        response = client.get("/api/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert "error" in data["storage"]

    def test_readiness_no_deps_configured(self, client_no_deps):
        """Test readiness check when no dependencies are injected."""
        response = client_no_deps.get("/api/ready")
        
        assert response.status_code == 503
        data = response.json()
        assert data["ready"] is False
        assert data["database"] == "not_configured"
        assert data["storage"] == "not_configured"


# ============================================================
# Detection Endpoint Tests
# ============================================================

class TestDetectionEndpoints:
    """Tests for detection API endpoints."""

    def test_list_detections_empty(self, client, mock_database):
        """Test listing detections when empty."""
        mock_database.count_detections.return_value = 0
        mock_database.query_detections.return_value = []
        
        response = client.get("/api/detections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["offset"] == 0
        assert data["limit"] == 100

    def test_list_detections_with_data(self, client, mock_database):
        """Test listing detections with data."""
        records = [
            DetectionRecord(
                id="det-1",
                image_path="test/image1.jpg",
                bbox_x1=10.0,
                bbox_y1=20.0,
                bbox_x2=30.0,
                bbox_y2=40.0,
                object_type="hair",
                confidence=0.95,
                created_at=datetime(2024, 1, 1, 12, 0, 0),
            ),
            DetectionRecord(
                id="det-2",
                image_path="test/image2.jpg",
                bbox_x1=50.0,
                bbox_y1=60.0,
                bbox_x2=70.0,
                bbox_y2=80.0,
                object_type="black_spot",
                confidence=0.88,
                created_at=datetime(2024, 1, 1, 12, 0, 1),
            ),
        ]
        mock_database.count_detections.return_value = 2
        mock_database.query_detections.return_value = records
        
        response = client.get("/api/detections")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        
        # Check first item
        item1 = data["items"][0]
        assert item1["id"] == "det-1"
        assert item1["object_type"] == "hair"
        assert item1["confidence"] == 0.95
        assert item1["bbox"]["x1"] == 10.0

    def test_list_detections_with_filters(self, client, mock_database):
        """Test listing detections with query filters."""
        mock_database.count_detections.return_value = 5
        mock_database.query_detections.return_value = []
        
        response = client.get(
            "/api/detections",
            params={
                "object_type": "hair",
                "limit": 50,
                "offset": 10,
            }
        )
        
        assert response.status_code == 200
        
        # Verify the database was called with correct filters
        mock_database.query_detections.assert_called_once()
        call_kwargs = mock_database.query_detections.call_args.kwargs
        assert call_kwargs["object_type"] == "hair"
        assert call_kwargs["limit"] == 50
        assert call_kwargs["offset"] == 10

    def test_list_detections_pagination(self, client, mock_database):
        """Test pagination parameters."""
        response = client.get(
            "/api/detections",
            params={"limit": 10, "offset": 20}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["offset"] == 20
        assert data["limit"] == 10

    def test_get_detection_found(self, client, mock_database):
        """Test getting a specific detection."""
        record = DetectionRecord(
            id="det-123",
            image_path="test/image.jpg",
            bbox_x1=10.0,
            bbox_y1=20.0,
            bbox_x2=30.0,
            bbox_y2=40.0,
            object_type="hair",
            confidence=0.95,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        mock_database.get_detection.return_value = record
        
        response = client.get("/api/detections/det-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "det-123"
        assert data["object_type"] == "hair"

    def test_get_detection_not_found(self, client, mock_database):
        """Test getting non-existent detection."""
        mock_database.get_detection.return_value = None
        
        response = client.get("/api/detections/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_detection_stats(self, client, mock_database):
        """Test getting detection statistics."""
        mock_database.count_detections.return_value = 100
        
        response = client.get("/api/detections/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "by_type" in data
        assert "time_range" in data

    def test_detections_no_database(self, client_no_deps):
        """Test detection endpoints fail gracefully without database."""
        response = client_no_deps.get("/api/detections")
        
        assert response.status_code == 503
        assert "not available" in response.json()["detail"].lower()


# ============================================================
# Dependency Injection Tests
# ============================================================

class TestDependencyInjection:
    """Tests for dependency injection system."""

    def test_app_state_initialized(self, mock_config, mock_database, mock_storage):
        """Test app state is properly initialized."""
        app = create_app(
            config=mock_config,
            database=mock_database,
            image_storage=mock_storage,
        )
        
        assert hasattr(app.state, "app_state")
        assert app.state.app_state.config is mock_config
        assert app.state.app_state.database is mock_database
        assert app.state.app_state.image_storage is mock_storage

    def test_app_without_deps(self):
        """Test app can be created without dependencies."""
        app = create_app()
        
        assert hasattr(app.state, "app_state")
        assert app.state.app_state.config is None
        assert app.state.app_state.database is None

    def test_version_from_config(self, mock_config):
        """Test API version comes from config."""
        mock_config.app.version = "1.2.3"
        
        app = create_app(config=mock_config)
        
        assert app.version == "1.2.3"


# ============================================================
# Service Layer Tests
# ============================================================

class TestDetectionService:
    """Tests for DetectionService."""

    def test_list_detections(self, mock_database):
        """Test service list_detections method."""
        from src.services.detection_service import DetectionService
        
        records = [
            DetectionRecord(
                id="det-1",
                image_path="test/image.jpg",
                bbox_x1=10.0,
                bbox_y1=20.0,
                bbox_x2=30.0,
                bbox_y2=40.0,
                object_type="hair",
                confidence=0.9,
            )
        ]
        mock_database.count_detections.return_value = 1
        mock_database.query_detections.return_value = records
        
        service = DetectionService(database=mock_database)
        result = service.list_detections(limit=10, offset=0)
        
        assert result.total == 1
        assert len(result.items) == 1
        assert result.items[0].id == "det-1"
        assert result.items[0].object_type == "hair"

    def test_get_detection(self, mock_database):
        """Test service get_detection method."""
        from src.services.detection_service import DetectionService
        
        record = DetectionRecord(
            id="det-123",
            image_path="test/image.jpg",
            bbox_x1=10.0,
            bbox_y1=20.0,
            bbox_x2=30.0,
            bbox_y2=40.0,
            object_type="hair",
            confidence=0.95,
        )
        mock_database.get_detection.return_value = record
        
        service = DetectionService(database=mock_database)
        result = service.get_detection("det-123")
        
        assert result is not None
        assert result.id == "det-123"

    def test_get_detection_not_found(self, mock_database):
        """Test service returns None for missing detection."""
        from src.services.detection_service import DetectionService
        
        mock_database.get_detection.return_value = None
        
        service = DetectionService(database=mock_database)
        result = service.get_detection("nonexistent")
        
        assert result is None

    def test_get_stats(self, mock_database):
        """Test service get_stats method."""
        from src.services.detection_service import DetectionService
        
        # Mock count returns based on object_type
        def count_side_effect(**kwargs):
            obj_type = kwargs.get("object_type")
            if obj_type == "hair":
                return 50
            elif obj_type == "black_spot":
                return 30
            elif obj_type is None:
                return 100
            return 0
        
        mock_database.count_detections.side_effect = count_side_effect
        
        service = DetectionService(database=mock_database)
        stats = service.get_stats()
        
        assert stats.total_detections == 100
        assert stats.by_type["hair"] == 50
        assert stats.by_type["black_spot"] == 30


