# Pluck Backend

Backend service for the Bird's Nest Inspection System (燕窝挑毛系统).

## Overview

This is the main control program for the current development phase, responsible for:

- Camera image acquisition (Daheng industrial camera)
- Foreign object detection (YOLO-based)
- Data storage (MinIO for images, PostgreSQL for records)
- REST API (reserved for future web frontend)

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker & Docker Compose (for MinIO and PostgreSQL)
- Daheng camera SDK (gxipy)
- CUDA-capable GPU (recommended)

### 2. Install Dependencies

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### 3. Start Services

```bash
# Start MinIO, PostgreSQL, Redis
docker-compose up -d

# Wait for services to be ready
docker-compose ps
```

### 4. Run the System

```bash
# Run detection loop (publishes detection events to Redis Streams)
python main.py --config config/settings.yaml --mode run

# Or run API server only (consumes Redis Streams, serves REST + WebSocket)
python main.py --config config/settings.yaml --mode api
```

### Real-time events

- Detection process publishes events to Redis Streams `pluck:detections` (configurable).
- API consumes the stream and broadcasts to WebSocket clients at `/api/ws/events`.
- Settings live under `redis` in `config/settings.yaml` (enable/disable, stream name, maxlen, etc.).

## Project Structure

```
backend/
├── src/
│   ├── core/           # Core algorithms (reusable in ROS)
│   │   ├── camera/     # Camera module
│   │   └── vision/     # Vision pipeline
│   ├── storage/        # Storage module (MinIO + PostgreSQL)
│   ├── scheduler/      # Main loop scheduler
│   └── api/            # REST API (FastAPI)
├── config/             # Configuration files
├── models/             # Model files (not in git)
├── tests/              # Test code
├── main.py             # Entry point
└── docker-compose.yml  # MinIO + PostgreSQL
```

## Configuration

See `config/settings.yaml` for all configuration options.

## Development

```bash
# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type check
mypy src/
```

## Documentation

See [docs/backend-design.md](../docs/backend-design.md) for detailed design documentation.

## License

Proprietary - All rights reserved.

