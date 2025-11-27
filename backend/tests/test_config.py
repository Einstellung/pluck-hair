"""Tests for configuration management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AppConfig,
    AppSettings,
    CameraConfig,
    DatabaseConfig,
    ImageStorageConfig,
    SchedulerConfig,
    StorageConfig,
    VisionConfig,
    load_config,
    merge_configs,
    get_config_value,
)


# ============================================================
# AppConfig Tests
# ============================================================

class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_values(self):
        """Test AppConfig has sensible defaults."""
        config = AppConfig()
        
        assert config.app.name == "pluck-backend"
        assert config.app.version == "0.1.0"
        assert config.camera.type == "daheng"
        assert config.storage.database.type == "postgres"

    def test_from_dict(self):
        """Test creating AppConfig from dictionary."""
        data = {
            "app": {"name": "test-app", "version": "2.0.0"},
            "camera": {"type": "mock", "width": 1280, "height": 720},
            "storage": {
                "database": {"type": "sqlite", "path": "./test.db"},
                "images": {"type": "local", "path": "./images"},
            },
        }
        
        config = AppConfig.from_dict(data)
        
        assert config.app.name == "test-app"
        assert config.app.version == "2.0.0"
        assert config.camera.type == "mock"
        assert config.camera.width == 1280
        assert config.storage.database.type == "sqlite"
        assert config.storage.images.type == "local"

    def test_from_yaml(self, tmp_path):
        """Test loading AppConfig from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_content = """
app:
  name: yaml-test
  version: 3.0.0
  log_level: DEBUG

camera:
  type: mock
  width: 800
  height: 600

storage:
  images:
    type: local
    path: ./test_images
  database:
    type: sqlite
    path: ./test.db
"""
        config_file.write_text(config_content)
        
        config = AppConfig.from_yaml(str(config_file))
        
        assert config.app.name == "yaml-test"
        assert config.app.version == "3.0.0"
        assert config.app.log_level == "DEBUG"
        assert config.camera.type == "mock"
        assert config.storage.images.type == "local"

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            AppConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_to_dict(self):
        """Test converting AppConfig back to dictionary."""
        config = AppConfig(
            app=AppSettings(name="test", version="1.0.0"),
            camera=CameraConfig(type="mock"),
        )
        
        data = config.to_dict()
        
        assert data["app"]["name"] == "test"
        assert data["app"]["version"] == "1.0.0"
        assert data["camera"]["type"] == "mock"

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Test environment variable substitution in YAML."""
        monkeypatch.setenv("TEST_DB_HOST", "mydbserver")
        monkeypatch.setenv("TEST_DB_PASSWORD", "secret123")
        
        config_file = tmp_path / "env_config.yaml"
        config_content = """
storage:
  database:
    type: postgres
    connection_string: "postgresql://user:${TEST_DB_PASSWORD}@${TEST_DB_HOST}:5432/db"
"""
        config_file.write_text(config_content)
        
        config = AppConfig.from_yaml(str(config_file))
        
        assert "mydbserver" in config.storage.database.connection_string
        assert "secret123" in config.storage.database.connection_string


# ============================================================
# Individual Config Tests
# ============================================================

class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_defaults(self):
        """Test default camera configuration."""
        config = CameraConfig()
        
        assert config.type == "daheng"
        assert config.device_index == 1
        assert config.exposure_auto is False
        assert config.width == 640
        assert config.height == 480

    def test_mock_camera_config(self):
        """Test mock camera configuration."""
        config = CameraConfig(
            type="mock",
            mode="directory",
            image_dir="/path/to/images",
            width=1920,
            height=1080,
        )
        
        assert config.type == "mock"
        assert config.mode == "directory"
        assert config.image_dir == "/path/to/images"


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_from_dict_minio(self):
        """Test creating MinIO storage config."""
        data = {
            "images": {
                "type": "minio",
                "endpoint": "minio.example.com:9000",
                "bucket": "my-bucket",
            },
            "database": {
                "type": "postgres",
                "connection_string": "postgresql://user:pass@host/db",
            },
        }
        
        config = StorageConfig.from_dict(data)
        
        assert config.images.type == "minio"
        assert config.images.endpoint == "minio.example.com:9000"
        assert config.database.type == "postgres"

    def test_from_dict_local(self):
        """Test creating local storage config."""
        data = {
            "images": {
                "type": "local",
                "path": "/data/images",
            },
            "database": {
                "type": "sqlite",
                "path": "/data/app.db",
            },
        }
        
        config = StorageConfig.from_dict(data)
        
        assert config.images.type == "local"
        assert config.images.path == "/data/images"
        assert config.database.type == "sqlite"


class TestVisionConfig:
    """Tests for VisionConfig."""

    def test_from_dict_with_steps(self):
        """Test creating vision config with pipeline steps."""
        data = {
            "pipeline": {
                "steps": [
                    {"name": "resize", "type": "resize", "params": {"size": [640, 640]}},
                    {"name": "detect", "type": "yolo", "params": {"model": "best.pt"}},
                    {"name": "nms", "type": "nms", "params": {"iou_threshold": 0.4}},
                ]
            }
        }
        
        config = VisionConfig.from_dict(data)
        
        assert len(config.steps) == 3
        assert config.steps[0].type == "resize"
        assert config.steps[1].type == "yolo"
        assert config.steps[2].params["iou_threshold"] == 0.4


class TestSchedulerConfig:
    """Tests for SchedulerConfig."""

    def test_defaults(self):
        """Test default scheduler configuration."""
        config = SchedulerConfig()
        
        assert config.loop_delay_ms == 100
        assert config.max_errors == 10
        assert config.async_storage is True
        assert config.storage_workers == 4


# ============================================================
# Legacy Function Tests
# ============================================================

class TestLegacyFunctions:
    """Tests for legacy config functions."""

    def test_load_config(self, tmp_path):
        """Test legacy load_config function."""
        config_file = tmp_path / "legacy_config.yaml"
        config_file.write_text("""
app:
  name: legacy-test
camera:
  type: mock
""")
        
        data = load_config(str(config_file))
        
        assert isinstance(data, dict)
        assert data["app"]["name"] == "legacy-test"
        assert data["camera"]["type"] == "mock"

    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        base = {
            "app": {"name": "base", "version": "1.0"},
            "camera": {"type": "daheng"},
        }
        override = {
            "app": {"version": "2.0"},
            "storage": {"type": "local"},
        }
        
        result = merge_configs(base, override)
        
        assert result["app"]["name"] == "base"  # Preserved
        assert result["app"]["version"] == "2.0"  # Overridden
        assert result["camera"]["type"] == "daheng"  # Preserved
        assert result["storage"]["type"] == "local"  # Added

    def test_get_config_value(self):
        """Test getting nested config values."""
        config = {
            "app": {
                "settings": {
                    "debug": True,
                }
            }
        }
        
        assert get_config_value(config, "app.settings.debug") is True
        assert get_config_value(config, "app.settings.missing", "default") == "default"
        assert get_config_value(config, "nonexistent.path") is None


