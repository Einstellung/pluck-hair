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
    save_annotated: bool = True
    async_storage: bool = True
    storage_workers: int = 4
    max_pending_saves: int = 100
    storage_retry_count: int = 3


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
    api: ApiConfig = field(default_factory=ApiConfig)
    
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
        api_data = data.get("api", {})
        
        return cls(
            app=AppSettings(**app_data),
            camera=CameraConfig(**camera_data),
            vision=VisionConfig.from_dict(vision_data),
            storage=StorageConfig.from_dict(storage_data),
            scheduler=SchedulerConfig(**scheduler_data),
            api=ApiConfig.from_dict(api_data),
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
                "save_annotated": self.scheduler.save_annotated,
                "async_storage": self.scheduler.async_storage,
                "storage_workers": self.scheduler.storage_workers,
                "max_pending_saves": self.scheduler.max_pending_saves,
                "storage_retry_count": self.scheduler.storage_retry_count,
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
