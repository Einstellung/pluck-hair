# Backend 模块设计文档

## 文档概述

### 目的

本文档定义燕窝挑毛系统 Backend 模块的设计方案，包括需求分析、架构决策和模块接口定义。Backend 作为当前开发阶段的主控程序，负责相机采集、视觉检测、数据存储等核心功能。

### 适用范围

- **开发人员**：了解模块职责和接口约定
- **架构师**：理解设计决策和扩展路径

### 关联文档

| 文档 | 说明 |
|------|------|
| [requirement.md](requirement.md) | 项目整体需求 |
| [system-architecture.md](system-architecture.md) | 系统架构设计（含 ROS2 部分） |

---

## 1. 背景与需求

### 1.1 项目背景

燕窝挑毛系统使用 3 台协作机械臂完成泡发燕窝表面的异物检测和夹取。当前阶段专注于 **视觉检测和数据管理**，暂不涉及机械臂控制。

```
当前阶段：Backend 主控          后期：ROS2 主控 + Backend 数据服务
────────────────────          ─────────────────────────────────

┌─────────────┐                ┌─────────────────┐
│   Backend   │ ◄── 主控        │    pluck_ws     │ ◄── 主控
│             │                │   (ROS2节点)     │
│ • 相机采集  │     演变为       │ • 机械臂控制    │
│ • 视觉检测  │  ─────────►     │ • 调用 Backend  │
│ • 数据存储  │                └────────┬────────┘
└─────────────┘                         │
                                        ▼
                               ┌─────────────────┐
                               │    Backend      │ ◄── 数据服务
                               └─────────────────┘
```

### 1.2 功能需求

| 模块 | 需求 | 优先级 |
|------|------|--------|
| **相机采集** | 支持大恒工业相机 (USB3.1, gxipy SDK) | P0 |
| | 支持配置曝光、增益等参数 | P1 |
| **异物检测** | 检测燕窝表面异物 (debris) | P0 |
| | 输出位置、类别、置信度 | P0 |
| | 支持 Pipeline 多步骤处理 | P1 |
| **数据存储** | 存储原始图像 (MinIO) | P0 |
| | 存储检测结果 (PostgreSQL) | P0 |
| **REST API** | 检测结果查询、图像获取 | P1 |

### 1.3 非功能需求

| 类别 | 需求 | 指标 |
|------|------|------|
| 稳定性 | 长期运行、异常自恢复 | 7×24 小时 |
| 性能 | 单帧处理延迟 | < 500ms |
| 可扩展 | core/ 模块可被 ROS2 复用 | - |
| 可维护 | 配置外置、日志完整 | - |

---

## 2. 架构设计决策

### 2.1 为什么 Backend 独立于 pluck_ws？

**决策**：Backend 作为独立项目，与 ROS2 工作空间分离。

| 考虑因素 | 分离的好处 |
|----------|------------|
| 关注点分离 | 数据管理 vs 机器人控制，职责清晰 |
| 独立开发测试 | 不需要 ROS 环境即可开发和测试 |
| 部署灵活性 | 可独立部署，或与 ROS 节点运行在不同机器 |

### 2.2 为什么用 Python？

**决策**：当前阶段全部使用 Python 开发。

| 考虑因素 | Python 的优势 |
|----------|---------------|
| 开发效率 | 快速迭代，调试方便 |
| 生态支持 | Ultralytics YOLO、SQLAlchemy、minio-py 原生支持 |
| 性能足够 | 系统瓶颈在机械臂动作（秒级），Python 性能可接受 |

**后期优化路径**：如需更高推理性能，使用 TensorRT + pybind11 封装。

### 2.3 为什么选择 MinIO + PostgreSQL？

**决策**：MinIO 存储图像，PostgreSQL 存储结构化数据。

```
┌─────────────────┐         ┌─────────────────┐
│   PostgreSQL    │         │     MinIO       │
│                 │  引用    │                 │
│  • 检测记录     │────────►│  • 原始图像     │
│  • 运行日志     │         │  • 结果图像     │
│  + image_path   │         │  bucket: pluck/ │
└─────────────────┘         └─────────────────┘
```

**理由**：
- MinIO：S3 兼容、工具丰富、易扩展
- PostgreSQL：支持复杂查询、功能强大
- 开发环境可用 SQLite + 本地文件替代

### 2.4 为什么采用 Pipeline 设计？

**决策**：视觉处理采用可配置的 Pipeline 架构。

```
输入图像 ──► Preprocess ──► Detect ──► Postprocess ──► 检测结果
             (resize)      (YOLO)       (NMS)
```

**好处**：
- 灵活配置：通过 YAML 调整流程
- 可扩展：添加新步骤只需实现接口
- 可测试：每个步骤可独立测试

---

## 3. 系统架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Backend 系统架构                        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    main.py (入口)                    │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              scheduler/TaskManager                   │    │
│  │         (采集 ──► 检测 ──► 存储 主循环)              │    │
│  └─────────┬───────────────┬───────────────┬───────────┘    │
│            │               │               │                 │
│            ▼               ▼               ▼                 │
│     ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│     │core/camera│   │core/vision│   │  storage/ │           │
│     │(可复用)   │   │ (可复用)  │   │           │           │
│     └─────┬─────┘   └───────────┘   └─────┬─────┘           │
│           │                               │                  │
│           ▼                               ▼                  │
│       [gxipy]                    [MinIO] [PostgreSQL]        │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    api/ (预留)                       │    │
│  │          FastAPI: /health, /detections, /images      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
backend/
├── src/
│   ├── core/                    # 核心算法（可被 ROS2 复用）
│   │   ├── camera/              # 相机抽象和实现
│   │   │   ├── base.py          # CameraBase 抽象类
│   │   │   ├── daheng.py        # 大恒相机实现
│   │   │   └── mock.py          # Mock 实现（测试用）
│   │   └── vision/              # 视觉处理
│   │       ├── pipeline.py      # Pipeline 框架
│   │       ├── types.py         # 数据类型定义
│   │       └── steps/           # 处理步骤实现
│   │
│   ├── storage/                 # 存储模块
│   │   ├── interfaces.py        # 抽象接口
│   │   ├── minio_storage.py     # MinIO 实现
│   │   ├── postgres_db.py       # PostgreSQL 实现
│   │   └── models.py            # ORM 模型
│   │
│   ├── scheduler/               # 调度模块
│   │   └── task_manager.py      # 主循环调度器
│   │
│   └── api/                     # REST API
│       ├── app.py               # FastAPI 应用
│       └── routes/              # 路由定义
│
├── config/                      # 配置文件
│   ├── settings.yaml            # 主配置
│   └── settings.dev.yaml        # 开发环境配置
│
├── models/                      # YOLO 模型文件
├── tests/                       # 测试代码
├── main.py                      # 主程序入口
├── docker-compose.yml           # MinIO + PostgreSQL
└── pyproject.toml               # 项目依赖
```

**模块职责**：

| 目录 | 职责 | ROS2 复用 |
|------|------|----------|
| `core/camera/` | 相机抽象接口 | ✅ |
| `core/vision/` | Pipeline 和检测步骤 | ✅ |
| `storage/` | 数据存储抽象 | ✅ |
| `scheduler/` | 主循环调度 | ❌ Backend 专用 |
| `api/` | REST API | ❌ Backend 专用 |

---

## 4. 模块接口定义

本节只定义抽象接口，具体实现见代码。

### 4.1 相机模块 (core/camera/)

```
┌─────────────────────────────────┐
│         CameraBase              │  ◄── 抽象基类
├─────────────────────────────────┤
│ + open() -> bool                │
│ + close() -> None               │
│ + capture() -> np.ndarray       │  # BGR, shape (H, W, C)
│ + is_opened() -> bool           │
│ + get_frame_size() -> (W, H)    │
└────────────────┬────────────────┘
                 │ 实现
        ┌────────┴────────┐
        ▼                 ▼
  DahengCamera        MockCamera
```

**配置约定**：

```yaml
camera:
  type: daheng | mock
  device_index: 1
  exposure_auto: false
  exposure_time: 10000  # microseconds
```

### 4.2 视觉模块 (core/vision/)

#### 数据类型

| 类型 | 字段 | 说明 |
|------|------|------|
| `BoundingBox` | x1, y1, x2, y2 | 边界框（像素坐标） |
| `Detection` | bbox, object_type, confidence | 单个检测结果，object_type 为字符串（如 "debris"） |
| `PipelineContext` | original_image, processed_image, detections, metadata | 步骤间传递的上下文 |
| `PipelineResult` | detections, processing_time_ms, metadata | Pipeline 最终输出 |

#### Pipeline 接口

```
┌───────────────────────────────────┐
│         VisionPipeline            │
├───────────────────────────────────┤
│ + add_step(step: ProcessStep)     │
│ + run(image: np.ndarray) -> Result│
│ + from_config(config) -> Pipeline │  # 从 YAML 构建
└───────────────────────────────────┘

┌───────────────────────────────────┐
│           ProcessStep             │  ◄── 抽象基类
├───────────────────────────────────┤
│ + name: str                       │
│ + process(ctx) -> ctx             │
└───────────────────────────────────┘
```

**内置步骤**：

| 类型 | 说明 | 关键参数 |
|------|------|----------|
| `resize` | 图像缩放 | size, keep_aspect |
| `normalize` | 归一化 | mean, std |
| `yolo` | YOLO 检测 | model, conf, device |
| `nms` | 非极大值抑制 | iou_threshold |
| `filter` | 过滤检测结果 | min_confidence, classes |

**配置约定**：

```yaml
vision:
  pipeline:
    steps:
      - name: preprocess
        type: resize
        params:
          size: [640, 640]
      - name: detect
        type: yolo
        params:
          model: models/best.pt
          conf: 0.5
          device: auto  # auto | cuda | cpu
      - name: postprocess
        type: nms
        params:
          iou_threshold: 0.4
```

### 4.3 存储模块 (storage/)

#### 图像存储接口

```
┌─────────────────────────────────┐
│         ImageStorage            │  ◄── 抽象基类
├─────────────────────────────────┤
│ + save(image, path) -> str      │  # 返回完整存储路径
│ + load(path) -> np.ndarray      │
│ + delete(path) -> bool          │
│ + exists(path) -> bool          │
└────────────────┬────────────────┘
                 │ 实现
        ┌────────┴────────┐
        ▼                 ▼
   MinIOStorage      LocalStorage
```

#### 数据库接口

```
┌─────────────────────────────────────────┐
│              Database                    │  ◄── 抽象基类
├─────────────────────────────────────────┤
│ + save_detection(record) -> str          │
│ + save_detections_batch(records) -> [str]│
│ + get_detection(id) -> DetectionRecord   │
│ + query_detections(filters) -> [records] │
└─────────────────┬───────────────────────┘
                  │ 实现
         ┌────────┴────────┐
         ▼                 ▼
  PostgresDatabase    SQLiteDatabase
```

#### 数据模型

| 表 | 字段 | 说明 |
|------|------|------|
| `detections` | id, image_path, bbox_*, object_type, confidence, created_at | 检测记录 |
| `sessions` | id, start_time, end_time, total_frames, total_detections | 运行会话 |

**配置约定**：

```yaml
storage:
  images:
    type: minio | local
    endpoint: "localhost:9000"
    bucket: "pluck-images"
  database:
    type: postgres | sqlite
    connection_string: "postgresql://user:pass@localhost:5432/pluck"
```

### 4.4 调度模块 (scheduler/)

`TaskManager` 负责主循环调度：

```
┌─────────────────────────────────────────────┐
│              TaskManager                     │
├─────────────────────────────────────────────┤
│ - camera: CameraBase                         │
│ - pipeline: VisionPipeline                   │
│ - image_storage: ImageStorage                │
│ - database: Database                         │
├─────────────────────────────────────────────┤
│ + start()                                    │
│ + stop()                                     │
└─────────────────────────────────────────────┘

主循环流程：
  1. camera.capture() -> image
  2. pipeline.run(image) -> detections
  3. image_storage.save(image, path)
  4. database.save_detections_batch(records)
  5. 异常计数，超过阈值停止
```

**配置约定**：

```yaml
scheduler:
  loop_delay_ms: 100
  max_errors: 10
  save_annotated: true
```

### 4.5 API 模块 (api/)

**预留接口**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/ready` | GET | 就绪检查（检测依赖服务） |
| `/api/detections` | GET | 查询检测结果 |
| `/api/detections/{id}` | GET | 获取单条检测 |
| `/api/images/{path}` | GET | 获取图像 |

---

## 5. 配置与部署

### 5.1 完整配置示例

```yaml
# config/settings.yaml

app:
  name: "pluck-backend"
  log_level: "INFO"

camera:
  type: daheng
  device_index: 1
  exposure_auto: false
  exposure_time: 10000

vision:
  pipeline:
    steps:
      - { name: preprocess, type: resize, params: { size: [640, 640] } }
      - { name: detect, type: yolo, params: { model: models/best.pt, conf: 0.5 } }
      - { name: postprocess, type: nms, params: { iou_threshold: 0.4 } }

storage:
  images:
    type: minio
    endpoint: "localhost:9000"
    access_key: "${MINIO_ACCESS_KEY}"
    secret_key: "${MINIO_SECRET_KEY}"
    bucket: "pluck-images"
  database:
    type: postgres
    connection_string: "${DATABASE_URL}"

scheduler:
  loop_delay_ms: 100
  max_errors: 10
  save_annotated: true

api:
  host: "0.0.0.0"
  port: 8000
```

### 5.2 Docker Compose（基础设施）

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: pluck
      POSTGRES_PASSWORD: pluck123
      POSTGRES_DB: pluck
    ports: ["5432:5432"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports: ["9000:9000", "9001:9001"]
```

### 5.3 依赖项

```
# 核心依赖
numpy, opencv-python, ultralytics, torch

# 数据库
sqlalchemy, psycopg2-binary

# 对象存储
minio

# API
fastapi, uvicorn, pydantic

# 配置
pyyaml

# 相机（需单独安装 SDK）
gxipy
```

---

## 6. 后续演进

### 6.1 与 ROS2 集成

```
backend/src/core/  ────────►  pluck_ws/src/applications/
     │                              │
     │  import 或 pip install       ▼
     └───────────────────►  detector_node.py
                            (调用 core.vision)
```

### 6.2 性能优化路径

1. **TensorRT 加速**：YOLO 转 TensorRT 引擎
2. **异步存储**：图像存储使用线程池
3. **C++ 关键模块**：必要时用 pybind11 封装

### 6.3 待细化内容

- [ ] REST API 详细接口定义
- [ ] 数据库表扩展（托盘管理、统计报表）
- [ ] 日志聚合和监控告警

---

## 附录

### 术语表

| 术语 | 说明 |
|------|------|
| Pipeline | 多步骤处理流程 |
| Detection | 单个异物检测结果 |
| Session | 一次运行会话 |
| NMS | Non-Maximum Suppression，非极大值抑制 |

### 相关文档

- [大恒图像 Galaxy SDK](https://www.daheng-imaging.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [SQLAlchemy](https://docs.sqlalchemy.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**文档状态**：✅ 设计完成  
**代码实现**：见 `backend/` 目录
