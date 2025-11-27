"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# Image Fixtures
# ============================================================

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # 640x480 BGR image with some random content
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale test image."""
    return np.random.randint(0, 256, (480, 640), dtype=np.uint8)


# ============================================================
# Storage Fixtures
# ============================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sqlite_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    from src.storage.sqlite_db import SQLiteDatabase
    
    db_path = str(tmp_path / "test.db")
    return SQLiteDatabase(db_path)


@pytest.fixture
def local_storage(tmp_path):
    """Create a local storage instance for testing."""
    from src.storage.local_storage import LocalStorage
    
    storage_path = str(tmp_path / "images")
    return LocalStorage(storage_path)


# ============================================================
# Camera Fixtures
# ============================================================

@pytest.fixture
def mock_camera():
    """Create a mock camera for testing."""
    from src.core.camera.mock import MockCamera
    from src.core.camera.base import CameraConfig
    
    config = CameraConfig()
    return MockCamera(config, mode="random", width=640, height=480)


# ============================================================
# Config Fixtures
# ============================================================

@pytest.fixture
def app_config():
    """Create an AppConfig instance for testing."""
    from src.config import AppConfig
    return AppConfig()


@pytest.fixture
def dev_config(tmp_path):
    """Create a development config for testing."""
    from src.config import AppConfig, CameraConfig, StorageConfig, ImageStorageConfig, DatabaseConfig
    
    return AppConfig(
        camera=CameraConfig(type="mock", width=640, height=480),
        storage=StorageConfig(
            images=ImageStorageConfig(type="local", path=str(tmp_path / "images")),
            database=DatabaseConfig(type="sqlite", path=str(tmp_path / "test.db")),
        ),
    )

