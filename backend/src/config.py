"""Configuration loading and management.

This module provides unified configuration management with:
- Type-safe configuration classes using dataclasses
- Environment variable substitution
- Single source of truth for all components
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# Configuration Data Classes
# ============================================================

@dataclass
class CameraConfig:
    """Camera configuration."""
    type: str = "daheng"
    device_index: int = 1
    exposure_auto: bool = False
    gain_auto: bool = False
    exposure_time: Optional[float] = None
    gain: Optional[float] = None
    white_balance_mode: str = "once"  # auto, once, manual, off
    white_balance_red: Optional[float] = None
    white_balance_green: Optional[float] = None
    white_balance_blue: Optional[float] = None
    gamma_enable: bool = False
    gamma_value: Optional[float] = None
    # For mock camera
    mode: str = "random"
    image_dir: Optional[str] = None
    width: int = 640
    height: int = 480


@dataclass
class PipelineStepConfig:
    """Single pipeline step configuration."""
    name: str = ""
    type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionConfig:
    """Vision pipeline configuration."""
    steps: List[PipelineStepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "VisionConfig":
        """Create VisionConfig from dictionary."""
        pipeline_data = data.get("pipeline", {})
        steps_data = pipeline_data.get("steps", [])
        steps = [
            PipelineStepConfig(
                name=s.get("name", ""),
                type=s.get("type", ""),
                params=s.get("params", {}),
            )
            for s in steps_data
        ]
        return cls(steps=steps)


@dataclass
class ImageStorageConfig:
    """Image storage configuration."""
    type: str = "minio"
    # MinIO settings
    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "pluck-images"
    secure: bool = False
    # Local storage settings
    path: str = "./data/images"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "postgres"
    connection_string: str = "postgresql://pluck:pluck123@localhost:5432/pluck"
    pool_size: int = 5
    echo: bool = False
    # SQLite settings
    path: str = "./data/pluck.db"


@dataclass
class StorageConfig:
    """Combined storage configuration."""
    images: ImageStorageConfig = field(default_factory=ImageStorageConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "StorageConfig":
        """Create StorageConfig from dictionary."""
        images_data = data.get("images", {})
        db_data = data.get("database", {})
        return cls(
            images=ImageStorageConfig(**images_data),
            database=DatabaseConfig(**db_data),
        )


@dataclass
class SchedulerConfig:
    """Scheduler/TaskManager configuration."""
    loop_delay_ms: int = 100
    max_errors: int = 10
    save_images: bool = True
    save_annotated: bool = True
    show_preview: bool = True  # Show real-time OpenCV preview window
    async_storage: bool = True
    storage_workers: int = 4
    max_pending_saves: int = 100
    storage_retry_count: int = 3


@dataclass
class TrackerConfig:
    """ByteTrack tracker configuration."""
    enabled: bool = False
    track_thresh: float = 0.5       # High confidence threshold for tracking
    track_buffer: int = 30          # Frames to keep lost tracks before deletion
    match_thresh: float = 0.8       # IoU threshold for matching
    min_hits: int = 3               # Consecutive hits before track is confirmed
    frame_rate: int = 30            # Expected frame rate (affects Kalman filter)


@dataclass
class SmoothingConfig:
    """Simple temporal smoothing for static objects."""
    enabled: bool = False
    alpha: float = 0.2              # Low-pass factor; smaller = smoother, slower
    min_iou: float = 0.1            # Match threshold to reuse previous box
    max_center_jump: float = 50.0   # Pixels; reset smoothing if jump is too large


@dataclass
class DoneConditionConfig:
    """Task completion condition configuration."""
    consecutive_empty: Optional[int] = None    # Stop after N consecutive empty frames
    max_iterations: Optional[int] = None       # Stop after N iterations
    timeout_seconds: Optional[float] = None    # Stop after timeout


@dataclass
class TaskConfig:
    """Task configuration."""
    name: str = "detection"
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    done_condition: DoneConditionConfig = field(default_factory=DoneConditionConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskConfig":
        """Create TaskConfig from dictionary."""
        tracker_data = data.get("tracker", {})
        smoothing_data = data.get("smoothing", {})
        done_condition_data = data.get("done_condition", {})
        return cls(
            name=data.get("name", "detection"),
            tracker=TrackerConfig(**tracker_data),
            smoothing=SmoothingConfig(**smoothing_data),
            done_condition=DoneConditionConfig(**done_condition_data),
        )


@dataclass
class RedisConfig:
    """Redis settings for inter-process messaging."""
    url: str = "redis://localhost:6379/0"
    stream: str = "pluck:detections"
    consumer_group: str = "api"
    consumer_name: str = "api-1"
    maxlen: int = 10000
    read_count: int = 10
    block_ms: int = 5000  # Max time (ms) to block XREAD when waiting for detection events
    enabled: bool = True  # whether to enable Redis Streams


@dataclass
class VideoStreamConfig:
    """Video streaming (MJPEG) settings."""
    enabled: bool = True
    stream: str = "pluck:frames"
    maxlen: int = 50
    fps_limit: float = 15.0
    jpeg_quality: int = 80
    block_ms: int = 2000  # Max time (ms) to block XREAD when waiting for new video frames


@dataclass
class CorsConfig:
    """CORS configuration."""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ApiConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors: CorsConfig = field(default_factory=CorsConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "ApiConfig":
        """Create ApiConfig from dictionary."""
        cors_data = data.get("cors", {})
        return cls(
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 8000),
            cors=CorsConfig(**cors_data) if cors_data else CorsConfig(),
        )


@dataclass
class AppSettings:
    """Application-level settings."""
    name: str = "pluck-backend"
    version: str = "0.1.0"
    log_level: str = "INFO"


@dataclass
class AppConfig:
    """Main application configuration container.
    
    This is the single source of truth for all configuration.
    Create once at startup and inject into all components.
    
    Example:
        >>> config = AppConfig.from_yaml("config/settings.yaml")
        >>> app = create_app(config)
        >>> storage = create_storage(config)
    """
    app: AppSettings = field(default_factory=AppSettings)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    video_stream: VideoStreamConfig = field(default_factory=VideoStreamConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    
    # Path to the config file (for reference/logging)
    _config_path: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, data: dict, config_path: Optional[str] = None) -> "AppConfig":
        """Create AppConfig from dictionary.
        
        Args:
            data: Configuration dictionary.
            config_path: Optional path for logging purposes.
            
        Returns:
            Populated AppConfig instance.
        """
        app_data = data.get("app", {})
        camera_data = data.get("camera", {})
        vision_data = data.get("vision", {})
        storage_data = data.get("storage", {})
        scheduler_data = data.get("scheduler", {})
        task_data = data.get("task", {})
        api_data = data.get("api", {})
        video_stream_data = data.get("video_stream", {})
        redis_data = data.get("redis", {})
        
        return cls(
            app=AppSettings(**app_data),
            camera=CameraConfig(**camera_data),
            vision=VisionConfig.from_dict(vision_data),
            storage=StorageConfig.from_dict(storage_data),
            scheduler=SchedulerConfig(**scheduler_data),
            task=TaskConfig.from_dict(task_data),
            api=ApiConfig.from_dict(api_data),
            video_stream=VideoStreamConfig(**video_stream_data),
            redis=RedisConfig(**redis_data),
            _config_path=config_path,
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "AppConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
            
        Returns:
            Populated AppConfig instance.
            
        Raises:
            FileNotFoundError: If config file not found.
            yaml.YAMLError: If YAML parsing fails.
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path) as f:
            content = f.read()
        
        # Substitute environment variables
        content = _substitute_env_vars(content)
        
        data = yaml.safe_load(content) or {}
        
        logger.info(f"Loaded configuration from {config_path}")
        return cls.from_dict(data, config_path=str(path.absolute()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (for legacy compatibility)."""
        return {
            "app": {
                "name": self.app.name,
                "version": self.app.version,
                "log_level": self.app.log_level,
            },
            "camera": {
                "type": self.camera.type,
                "device_index": self.camera.device_index,
            "exposure_auto": self.camera.exposure_auto,
            "gain_auto": self.camera.gain_auto,
            "exposure_time": self.camera.exposure_time,
            "gain": self.camera.gain,
            "white_balance_mode": self.camera.white_balance_mode,
            "white_balance_red": self.camera.white_balance_red,
            "white_balance_green": self.camera.white_balance_green,
            "white_balance_blue": self.camera.white_balance_blue,
            "gamma_enable": self.camera.gamma_enable,
            "gamma_value": self.camera.gamma_value,
            "mode": self.camera.mode,
            "image_dir": self.camera.image_dir,
            "width": self.camera.width,
            "height": self.camera.height,
        },
            "vision": {
                "pipeline": {
                    "steps": [
                        {"name": s.name, "type": s.type, "params": s.params}
                        for s in self.vision.steps
                    ]
                }
            },
            "storage": {
                "images": {
                    "type": self.storage.images.type,
                    "endpoint": self.storage.images.endpoint,
                    "access_key": self.storage.images.access_key,
                    "secret_key": self.storage.images.secret_key,
                    "bucket": self.storage.images.bucket,
                    "secure": self.storage.images.secure,
                    "path": self.storage.images.path,
                },
                "database": {
                    "type": self.storage.database.type,
                    "connection_string": self.storage.database.connection_string,
                    "pool_size": self.storage.database.pool_size,
                    "echo": self.storage.database.echo,
                    "path": self.storage.database.path,
                },
            },
            "scheduler": {
                "loop_delay_ms": self.scheduler.loop_delay_ms,
                "max_errors": self.scheduler.max_errors,
                "save_images": self.scheduler.save_images,
                "save_annotated": self.scheduler.save_annotated,
                "async_storage": self.scheduler.async_storage,
                "storage_workers": self.scheduler.storage_workers,
                "max_pending_saves": self.scheduler.max_pending_saves,
                "storage_retry_count": self.scheduler.storage_retry_count,
            },
            "task": {
                "name": self.task.name,
                "tracker": {
                    "enabled": self.task.tracker.enabled,
                    "track_thresh": self.task.tracker.track_thresh,
                    "track_buffer": self.task.tracker.track_buffer,
                    "match_thresh": self.task.tracker.match_thresh,
                    "min_hits": self.task.tracker.min_hits,
                    "frame_rate": self.task.tracker.frame_rate,
                },
                "smoothing": {
                    "enabled": self.task.smoothing.enabled,
                    "alpha": self.task.smoothing.alpha,
                    "min_iou": self.task.smoothing.min_iou,
                    "max_center_jump": self.task.smoothing.max_center_jump,
                },
                "done_condition": {
                    "consecutive_empty": self.task.done_condition.consecutive_empty,
                    "max_iterations": self.task.done_condition.max_iterations,
                    "timeout_seconds": self.task.done_condition.timeout_seconds,
                },
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "cors": {
                    "allow_origins": self.api.cors.allow_origins,
                    "allow_credentials": self.api.cors.allow_credentials,
                    "allow_methods": self.api.cors.allow_methods,
                    "allow_headers": self.api.cors.allow_headers,
                },
            },
            "video_stream": {
                "enabled": self.video_stream.enabled,
                "stream": self.video_stream.stream,
                "maxlen": self.video_stream.maxlen,
                "fps_limit": self.video_stream.fps_limit,
                "jpeg_quality": self.video_stream.jpeg_quality,
                "block_ms": self.video_stream.block_ms,
            },
            "redis": {
                "url": self.redis.url,
                "stream": self.redis.stream,
                "consumer_group": self.redis.consumer_group,
                "consumer_name": self.redis.consumer_name,
                "maxlen": self.redis.maxlen,
                "read_count": self.redis.read_count,
                "block_ms": self.redis.block_ms,
                "enabled": self.redis.enabled,
            },
        }


# ============================================================
# Helper Functions
# ============================================================

def _substitute_env_vars(content: str) -> str:
    """Substitute ${VAR_NAME} patterns with environment variable values.
    
    Args:
        content: String content with potential ${VAR} patterns.
        
    Returns:
        String with environment variables substituted.
    """
    pattern = r'\$\{([^}]+)\}'
    
    def replacer(match):
        var_name = match.group(1)
        value = os.environ.get(var_name, "")
        if not value:
            logger.warning(f"Environment variable not set: {var_name}")
        return value
    
    return re.sub(pattern, replacer, content)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file (legacy function).
    
    DEPRECATED: Use AppConfig.from_yaml() instead for type-safe config.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    config = AppConfig.from_yaml(config_path)
    return config.to_dict()


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base: Base configuration.
        override: Override configuration.
        
    Returns:
        Merged configuration.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "camera.device_index").
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
    
    Example:
        >>> config = {"camera": {"type": "daheng"}}
        >>> get_config_value(config, "camera.type")
        'daheng'
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
