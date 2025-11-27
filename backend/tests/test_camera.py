"""Tests for camera module."""

import numpy as np
import pytest


class TestMockCamera:
    """Tests for MockCamera."""

    def test_open_close(self, mock_camera):
        """Test camera open and close."""
        assert not mock_camera.is_opened()
        
        result = mock_camera.open()
        assert result is True
        assert mock_camera.is_opened()
        
        mock_camera.close()
        assert not mock_camera.is_opened()

    def test_capture(self, mock_camera):
        """Test image capture."""
        mock_camera.open()
        
        image = mock_camera.capture()
        
        assert isinstance(image, np.ndarray)
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        
        mock_camera.close()

    def test_capture_not_opened(self, mock_camera):
        """Test capture raises error when not opened."""
        with pytest.raises(RuntimeError, match="not opened"):
            mock_camera.capture()

    def test_get_frame_size(self, mock_camera):
        """Test getting frame size."""
        mock_camera.open()
        
        width, height = mock_camera.get_frame_size()
        
        assert width == 640
        assert height == 480
        
        mock_camera.close()

    def test_context_manager(self, mock_camera):
        """Test camera as context manager."""
        with mock_camera:
            assert mock_camera.is_opened()
            image = mock_camera.capture()
            assert image is not None
        
        assert not mock_camera.is_opened()


class TestCameraConfig:
    """Tests for CameraConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        from src.core.camera.base import CameraConfig
        
        config = CameraConfig()
        
        assert config.device_index == 1
        assert config.exposure_auto is False
        assert config.gain_auto is False
        assert config.exposure_time is None
        assert config.gain is None

    def test_custom_values(self):
        """Test custom configuration values."""
        from src.core.camera.base import CameraConfig
        
        config = CameraConfig(
            device_index=2,
            exposure_auto=True,
            gain_auto=True,
            exposure_time=10000,
            gain=2.0,
        )
        
        assert config.device_index == 2
        assert config.exposure_auto is True
        assert config.gain_auto is True
        assert config.exposure_time == 10000
        assert config.gain == 2.0


