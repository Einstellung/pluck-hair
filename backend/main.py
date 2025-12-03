#!/usr/bin/env python3
"""Main entry point for Pluck Backend.

Usage:
    # Run detection loop
    python main.py --config config/settings.yaml --mode run
    
    # Run API server only
    python main.py --config config/settings.yaml --mode api
    
    # Development mode with mock camera
    python main.py --config config/settings.dev.yaml --mode run
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import AppConfig


def setup_logging(level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_camera(config: AppConfig):
    """Create camera instance from configuration.
    
    Args:
        config: Application configuration.
        
    Returns:
        Camera instance.
    """
    camera_config = config.camera
    
    if camera_config.type == "daheng":
        from src.core.camera.daheng import DahengCamera
        from src.core.camera.base import CameraConfig as BaseCameraConfig
        base_config = BaseCameraConfig(
            device_index=camera_config.device_index,
            exposure_auto=camera_config.exposure_auto,
            gain_auto=camera_config.gain_auto,
            exposure_time=camera_config.exposure_time,
            gain=camera_config.gain,
        )
        return DahengCamera(base_config)
    
    elif camera_config.type == "mock":
        from src.core.camera.mock import MockCamera
        from src.core.camera.base import CameraConfig as BaseCameraConfig
        base_config = BaseCameraConfig()
        return MockCamera(
            base_config,
            mode=camera_config.mode,
            image_dir=camera_config.image_dir,
            width=camera_config.width,
            height=camera_config.height,
        )
    
    else:
        raise ValueError(f"Unknown camera type: {camera_config.type}")


def create_pipeline(config: AppConfig):
    """Create vision pipeline from configuration.
    
    Args:
        config: Application configuration.
        
    Returns:
        VisionPipeline instance.
    """
    from src.core.vision.pipeline import VisionPipeline

    return VisionPipeline.from_vision_config(config.vision)


def create_storage(config: AppConfig):
    """Create storage instances from configuration.
    
    Args:
        config: Application configuration.
        
    Returns:
        Tuple of (image_storage, database).
    """
    storage_config = config.storage
    
    # Image storage
    images_config = storage_config.images
    if images_config.type == "minio":
        from src.storage.minio_storage import MinIOStorage
        image_storage = MinIOStorage(
            endpoint=images_config.endpoint,
            access_key=images_config.access_key,
            secret_key=images_config.secret_key,
            bucket=images_config.bucket,
            secure=images_config.secure,
        )
    elif images_config.type == "local":
        from src.storage.local_storage import LocalStorage
        image_storage = LocalStorage(base_path=images_config.path)
    else:
        raise ValueError(f"Unknown storage type: {images_config.type}")
    
    # Database
    db_config = storage_config.database
    if db_config.type == "postgres":
        from src.storage.postgres_db import PostgresDatabase
        database = PostgresDatabase(
            connection_string=db_config.connection_string,
            echo=db_config.echo,
            pool_size=db_config.pool_size,
        )
    elif db_config.type == "sqlite":
        from src.storage.sqlite_db import SQLiteDatabase
        database = SQLiteDatabase(
            db_path=db_config.path,
            echo=db_config.echo,
        )
    else:
        raise ValueError(f"Unknown database type: {db_config.type}. Supported: postgres, sqlite")
    
    return image_storage, database


def run_detection_loop(config: AppConfig):
    """Run the main detection loop.
    
    Args:
        config: Application configuration.
    """
    logger = logging.getLogger(__name__)
    logger.info("Initializing detection system...")
    
    # Create components with unified config
    camera = create_camera(config)
    pipeline = create_pipeline(config)
    image_storage, database = create_storage(config)

    # Optional Redis publisher for real-time events
    event_publisher = None
    if getattr(config, "redis", None) and config.redis.enabled:
        try:
            from src.events.redis_streams import RedisStreamPublisher
            event_publisher = RedisStreamPublisher(
                url=config.redis.url,
                stream=config.redis.stream,
                maxlen=config.redis.maxlen,
            )
            logger.info("Redis Streams publisher enabled for stream %s", config.redis.stream)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis publisher: {e}")
    
    # Create task manager
    from src.scheduler.task_manager import TaskManager, TaskManagerConfig
    
    scheduler_config = config.scheduler
    task_config = TaskManagerConfig(
        loop_delay_ms=scheduler_config.loop_delay_ms,
        max_errors=scheduler_config.max_errors,
        save_annotated=scheduler_config.save_annotated,
        show_preview=scheduler_config.show_preview,
        async_storage=scheduler_config.async_storage,
        storage_workers=scheduler_config.storage_workers,
        max_pending_saves=scheduler_config.max_pending_saves,
        storage_retry_count=scheduler_config.storage_retry_count,
    )
    
    manager = TaskManager(
        camera=camera,
        pipeline=pipeline,
        image_storage=image_storage,
        database=database,
        config=task_config,
        event_publisher=event_publisher,
    )
    
    logger.info("Starting detection loop...")
    manager.start()


def run_api_server(config: AppConfig):
    """Run the API server with properly injected dependencies.
    
    Args:
        config: Application configuration.
    """
    import uvicorn
    from src.api.app import create_app
    
    logger = logging.getLogger(__name__)
    
    # Create storage instances (same ones used by both API and health checks)
    image_storage, database = create_storage(config)
    logger.info(f"Storage initialized: images={config.storage.images.type}, db={config.storage.database.type}")
    
    # Create app with injected dependencies
    app = create_app(
        config=config,
        database=database,
        image_storage=image_storage,
        title="Pluck Backend API",
    )
    
    logger.info("Starting API server with injected dependencies")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pluck Backend - Bird's Nest Inspection System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "api"],
        default="run",
        help="Run mode: 'run' for detection loop, 'api' for API server",
    )
    args = parser.parse_args()
    
    # Load configuration using unified AppConfig
    try:
        config = AppConfig.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    setup_logging(config.app.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Pluck Backend in {args.mode} mode")
    logger.info(f"Configuration: {args.config}")
    
    # Run selected mode
    try:
        if args.mode == "run":
            run_detection_loop(config)
        elif args.mode == "api":
            run_api_server(config)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
