"""FastAPI application for Pluck Backend.

This module provides the application factory with proper dependency injection.
All dependencies (config, database, storage) are injected at creation time.

Example:
    # Create with dependencies
    config = AppConfig.from_yaml("config/settings.yaml")
    image_storage, database = create_storage(config)
    app = create_app(
        config=config,
        database=database,
        image_storage=image_storage,
    )
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig
from src.storage.interfaces import Database, ImageStorage

from .dependencies import AppState
from .routes import detections, health, images


def create_app(
    config: Optional[AppConfig] = None,
    database: Optional[Database] = None,
    image_storage: Optional[ImageStorage] = None,
    title: str = "Pluck Backend API",
    version: str = "0.1.0",
) -> FastAPI:
    """Create and configure FastAPI application with dependency injection.
    
    Args:
        config: Application configuration (injected).
        database: Database instance (injected).
        image_storage: Image storage instance (injected).
        title: API title.
        version: API version (overridden by config if provided).
        
    Returns:
        Configured FastAPI application with injected dependencies.
    """
    # Use version from config if available
    if config is not None:
        version = config.app.version
    
    app = FastAPI(
        title=title,
        description="REST API for Bird's Nest Inspection System (燕窝挑毛系统)",
        version=version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    # Configure CORS
    cors_config = config.api.cors if config else None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins if cors_config else ["*"],
        allow_credentials=cors_config.allow_credentials if cors_config else True,
        allow_methods=cors_config.allow_methods if cors_config else ["*"],
        allow_headers=cors_config.allow_headers if cors_config else ["*"],
    )
    
    # Inject dependencies via app.state
    app.state.app_state = AppState(
        config=config,
        database=database,
        image_storage=image_storage,
    )
    
    # Include routers
    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(detections.router, prefix="/api", tags=["Detections"])
    app.include_router(images.router, prefix="/api", tags=["Images"])
    
    return app


# Default app instance (for uvicorn direct run without main.py)
# Note: This will have no dependencies injected - use main.py for production
app = create_app()
