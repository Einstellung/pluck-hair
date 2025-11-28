"""Daheng Imaging camera implementation using gxipy SDK."""

import logging
from typing import Optional, Tuple

import numpy as np

from .base import CameraBase, CameraConfig

logger = logging.getLogger(__name__)


class DahengCamera(CameraBase):
    """Daheng Imaging camera implementation.
    
    Uses the gxipy SDK (Galaxy SDK Python bindings) to interface
    with Daheng industrial cameras connected via USB3.
    
    Example:
        >>> config = CameraConfig(device_index=1)
        >>> with DahengCamera(config) as camera:
        ...     image = camera.capture()
        ...     print(f"Captured image: {image.shape}")
    """

    def __init__(self, config: CameraConfig):
        """Initialize Daheng camera.
        
        Args:
            config: Camera configuration.
        """
        self.config = config
        self._dev_mgr = None
        self._cam = None
        self._is_opened = False
        self._gx = None

    def open(self) -> bool:
        """Open the camera device.
        
        Returns:
            True if opened successfully.
            
        Raises:
            RuntimeError: If no camera found or failed to open.
        """
        try:
            # Import gxipy here to avoid import errors when SDK not installed
            import gxipy as gx
            self._gx = gx
            
            # Initialize device manager
            self._dev_mgr = gx.DeviceManager()
            num, dev_info_list = self._dev_mgr.update_device_list()
            
            if num == 0:
                raise RuntimeError("No Daheng camera found")
            
            logger.info(f"Found {num} Daheng camera(s)")
            
            # Open device by index (1-based)
            self._cam = self._dev_mgr.open_device_by_index(
                self.config.device_index
            )
            
            # Configure camera parameters
            self._configure_camera()
            
            # Start streaming
            self._cam.stream_on()
            self._is_opened = True
            
            width, height = self.get_frame_size()
            logger.info(
                f"Camera opened: device_index={self.config.device_index}, "
                f"resolution={width}x{height}"
            )
            
            return True
            
        except Exception as e:
            self._is_opened = False
            logger.error(f"Failed to open camera: {e}")
            raise RuntimeError(f"Failed to open camera: {e}")

    def _configure_camera(self) -> None:
        """Configure camera parameters based on config."""
        # Auto exposure
        if hasattr(self._cam, 'ExposureAuto'):
            self._cam.ExposureAuto.set(self.config.exposure_auto)
            logger.debug(f"ExposureAuto set to {self.config.exposure_auto}")
        
        # Auto gain
        if hasattr(self._cam, 'GainAuto'):
            self._cam.GainAuto.set(self.config.gain_auto)
            logger.debug(f"GainAuto set to {self.config.gain_auto}")
        
        # Manual exposure time
        if self.config.exposure_time is not None:
            if hasattr(self._cam, 'ExposureTime'):
                self._cam.ExposureTime.set(self.config.exposure_time)
                logger.debug(f"ExposureTime set to {self.config.exposure_time}us")
        
        # Manual gain
        if self.config.gain is not None:
            if hasattr(self._cam, 'Gain'):
                self._cam.Gain.set(self.config.gain)
                logger.debug(f"Gain set to {self.config.gain}")

        # Gamma
        self._configure_gamma()
        
        # White balance
        self._configure_white_balance()

    def _configure_gamma(self) -> None:
        """Configure gamma based on config."""
        if not (self.config.gamma_enable or self.config.gamma_value):
            return
        cam = self._cam
        if cam is None:
            return
        try:
            if hasattr(cam, "GammaEnable"):
                cam.GammaEnable.set(True)
                logger.debug("GammaEnable set to True")
            value = self.config.gamma_value
            if value is not None:
                # Some SDKs expose GammaParam or Gamma
                if hasattr(cam, "GammaParam"):
                    cam.GammaParam.set(value)
                    logger.debug(f"GammaParam set to {value}")
                elif hasattr(cam, "Gamma"):
                    cam.Gamma.set(value)
                    logger.debug(f"Gamma set to {value}")
        except Exception as e:
            logger.warning(f"Failed to configure gamma: {e}")

    def _configure_white_balance(self) -> None:
        """Configure white balance based on config."""
        gx = self._gx
        if gx is None:
            logger.debug("gxipy not loaded; skipping white balance configuration")
            return
        mode = (self.config.white_balance_mode or "auto").lower()
        
        # Mapping to SDK enum
        mode_map = {
            "off": gx.GxAutoEntry.OFF,
            "auto": gx.GxAutoEntry.CONTINUOUS,
            "continuous": gx.GxAutoEntry.CONTINUOUS,
            "once": gx.GxAutoEntry.ONCE,
            "manual": gx.GxAutoEntry.OFF,
        }
        if hasattr(self._cam, "BalanceWhiteAuto"):
            if mode in mode_map:
                try:
                    self._cam.BalanceWhiteAuto.set(mode_map[mode])
                    logger.debug(f"BalanceWhiteAuto set to {mode}")
                except Exception as e:
                    logger.warning(f"Failed to set BalanceWhiteAuto to {mode}: {e}")
            else:
                logger.warning(f"Unknown white_balance_mode '{mode}', skipping")
        
        # Manual ratios only when requested and supported
        if mode == "manual":
            if not (hasattr(self._cam, "BalanceRatioSelector") and hasattr(self._cam, "BalanceRatio")):
                logger.warning("Manual white balance not supported by camera")
                return
            ratios = [
                ("red", gx.GxBalanceRatioSelectorEntry.RED, self.config.white_balance_red),
                ("green", gx.GxBalanceRatioSelectorEntry.GREEN, self.config.white_balance_green),
                ("blue", gx.GxBalanceRatioSelectorEntry.BLUE, self.config.white_balance_blue),
            ]
            for name, selector_entry, value in ratios:
                if value is None:
                    continue
                try:
                    self._cam.BalanceRatioSelector.set(selector_entry)
                    self._cam.BalanceRatio.set(value)
                    logger.debug(f"BalanceRatio {name} set to {value}")
                except Exception as e:
                    logger.warning(f"Failed to set BalanceRatio {name}: {e}")
        elif mode == "once":
            # Some SDKs perform "once" automatically on set; no extra call needed.
            logger.debug("White balance once mode requested")

    def close(self) -> None:
        """Close the camera device."""
        if self._cam is not None:
            try:
                self._cam.stream_off()
                self._cam.close_device()
                logger.info("Camera closed")
            except Exception as e:
                logger.warning(f"Error closing camera: {e}")
            finally:
                self._cam = None
        
        self._is_opened = False

    def capture(self) -> np.ndarray:
        """Capture a single frame.
        
        Returns:
            Image as numpy array (H, W, C) or (H, W) for mono.
            
        Raises:
            RuntimeError: If camera not opened or capture fails.
        """
        if not self._is_opened:
            raise RuntimeError("Camera not opened")
        
        # Get image from data stream
        raw_image = self._cam.data_stream[0].get_image()
        
        if raw_image is None:
            raise RuntimeError("Failed to capture image")
        
        # Convert to numpy array
        image = raw_image.get_numpy_array()
        
        if image is None:
            raise RuntimeError("Failed to convert image to numpy array")
        
        return image

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened

    def get_frame_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        if not self._is_opened:
            raise RuntimeError("Camera not opened")
        
        width = self._cam.Width.get()
        height = self._cam.Height.get()
        return (width, height)

    def set_exposure(self, exposure_time: float) -> None:
        """Set exposure time.
        
        Args:
            exposure_time: Exposure time in microseconds.
        """
        if not self._is_opened:
            raise RuntimeError("Camera not opened")
        
        if hasattr(self._cam, 'ExposureTime'):
            self._cam.ExposureTime.set(exposure_time)
            logger.debug(f"ExposureTime set to {exposure_time}us")

    def set_gain(self, gain: float) -> None:
        """Set gain value.
        
        Args:
            gain: Gain value.
        """
        if not self._is_opened:
            raise RuntimeError("Camera not opened")
        
        if hasattr(self._cam, 'Gain'):
            self._cam.Gain.set(gain)
            logger.debug(f"Gain set to {gain}")
