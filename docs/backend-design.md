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
| | 支持 Tracking 跨帧跟踪过滤 | P1 |
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

### 2.4 为什么采用三层架构设计？

**决策**：采用 Workflow → Task → Pipeline 三层架构。

**问题背景**：
- 原有设计中 `TaskManager` 混合了任务编排、图像处理、存储推送多种职责
- 引入 Tracking（跨帧状态）后，状态归属不清晰
- 未来需要支持 A/B/C 多任务组合（如：挑大块 → 挑小块 → 质检）

**解决方案**：

```
┌─────────────────────────────────────────────────────────────┐
│  层级 3：Workflow（工作流）                                  │
│  编排多个任务的执行顺序                                       │
│  "先挑大块 → 再挑小块 → 最后质检"                             │
│  可配置顺序，可能有条件分支                                   │
├─────────────────────────────────────────────────────────────┤
│  层级 2：Task（任务）                                        │
│  有状态的任务单元                                            │
│  "挑大块异物" = 拍照 + 检测 + 跟踪 + 统计                     │
│  持有：Tracker 状态、统计器、完成条件                         │
├─────────────────────────────────────────────────────────────┤
│  层级 1：Pipeline（管道）                                    │
│  无状态的单帧图像处理                                         │
│  tile → detect → merge → filter                             │
│  纯函数式：进来一张图，出去检测结果                           │
└─────────────────────────────────────────────────────────────┘
```

**好处**：
- Pipeline 保持无状态，纯粹的图像处理，可测试、可复用
- Task 持有跨帧状态（Tracker、Stats），职责清晰
- Workflow 编排多任务，支持未来扩展
- 存储/事件等横切关注点集中在 TaskManager

---

## 3. 系统架构

### 3.1 三层架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Backend 系统架构                                 │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                       main.py (入口)                               │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      TaskManager（调度器）                         │  │
│  │                                                                    │  │
│  │   职责：                                                           │  │
│  │   • 主循环控制           • 会话管理                                │  │
│  │   • 相机采集             • 存储/事件推送                           │  │
│  │   • 调用 Task            • 异常处理                                │  │
│  │                                                                    │  │
│  │   横切关注点：存储、事件、预览 统一由 TaskManager 处理              │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         Task（任务层）                             │  │
│  │                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────┐ │  │
│  │   │                    DetectionTask                             │ │  │
│  │   │                                                              │ │  │
│  │   │    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │  │
│  │   │    │   Pipeline   │  │   Tracker    │  │    Stats     │    │ │  │
│  │   │    │   (注入)     │  │   (状态)     │  │   (统计)     │    │ │  │
│  │   │    └──────────────┘  └──────────────┘  └──────────────┘    │ │  │
│  │   │                                                              │ │  │
│  │   │    ┌──────────────┐                                         │ │  │
│  │   │    │DoneCondition │  完成条件：连续N帧无检测/超时/计数阈值   │ │  │
│  │   │    └──────────────┘                                         │ │  │
│  │   └─────────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────┬───────────────────────────────────┘  │
│                                  │                                       │
│                                  ▼                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     VisionPipeline（管道层）                       │  │
│  │                                                                    │  │
│  │      ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐           │  │
│  │      │  Tile  │──►│  YOLO  │──►│ Merge  │──►│ Filter │           │  │
│  │      └────────┘   └────────┘   └────────┘   └────────┘           │  │
│  │                                                                    │  │
│  │      无状态：image in → detections out                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────┬──────────────────────────┐  │
│  │           core/camera                  │        storage/          │  │
│  │         (相机抽象)                     │      (存储抽象)          │  │
│  └────────────────────────────────────────┴──────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          api/ (预留)                               │  │
│  │             FastAPI: /health, /detections, /images                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 未来多任务架构（Workflow）

未来支持 A → B → C 多任务组合时的架构：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WorkflowEngine（未来）                           │
│                                                                          │
│   配置：[TaskA, TaskB, TaskC]                                            │
│   职责：按顺序执行任务，处理任务间依赖，汇总统计                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│       TaskA         │  │       TaskB         │  │       TaskC         │
│    "挑大块异物"      │  │     "挑小异物"       │  │    "质检分析"       │
│                     │  │                     │  │                     │
│  • pipeline (注入)  │  │  • pipeline (注入)  │  │  • pipeline (注入)  │
│  • tracker 状态     │  │  • tracker 状态     │  │  • 判定规则         │
│  • 统计器           │  │  • 统计器           │  │  • 统计器           │
│  • 完成条件         │  │  • 完成条件         │  │  • 完成条件         │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           │     Pipeline 可共享（同模型）或独立（不同模型）    │
           ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       VisionPipeline（无状态）                           │
│                                                                          │
│         可配置的 Step 组合：tile → detect → merge → filter              │
│                                                                          │
│         • Pipeline 通过 TaskManager 注入给 Task，避免重复加载模型        │
│         • 未来多模型场景：A/B/C 各用不同 Pipeline，串行执行释放 GPU      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 职责边界

| 层级 | 职责 | 状态 | 生命周期 |
|------|------|------|----------|
| **Pipeline** | 单帧图像处理 | 无 | 长期复用 |
| **Task** | 算法逻辑 + 跨帧状态 | Tracker, Stats | 区域/任务切换时 reset |
| **TaskManager** | 采集循环、存储、事件、会话 | Session | 整个运行期间 |
| **Workflow** | 多任务编排（未来）| 任务序列 | 单次流程执行 |

**关键设计原则**：

1. **Pipeline 通过注入**：由 TaskManager 创建，注入给 Task，避免重复加载模型
2. **状态归属清晰**：Tracker 和 Stats 属于 Task，不属于 Pipeline
3. **横切关注点集中**：存储、事件推送仍由 TaskManager 统一处理
4. **重拍主导权**：每次 Task 迭代都由 TaskManager 重新采集图像

### 3.4 目录结构

```
backend/
├── src/
│   ├── core/                       # 核心算法（可被 ROS2 复用）
│   │   ├── camera/                 # 相机抽象和实现
│   │   │   ├── base.py             # CameraBase 抽象类
│   │   │   ├── daheng.py           # 大恒相机实现
│   │   │   └── mock.py             # Mock 实现（测试用）
│   │   └── vision/                 # 视觉处理
│   │       ├── pipeline.py         # Pipeline 框架
│   │       ├── types.py            # 数据类型定义
│   │       └── steps/              # 处理步骤实现
│   │
│   ├── storage/                    # 存储模块
│   │   ├── interfaces.py           # 抽象接口
│   │   ├── minio_storage.py        # MinIO 实现
│   │   ├── local_storage.py        # 本地存储实现
│   │   ├── postgres_db.py          # PostgreSQL 实现
│   │   ├── sqlite_db.py            # SQLite 实现
│   │   └── models.py               # ORM 模型
│   │
│   ├── scheduler/                  # 调度模块
│   │   ├── task_manager.py         # 主循环调度器
│   │   ├── storage_saver.py        # 异步存储处理
│   │   └── tasks/                  # 任务层（新增）
│   │       ├── base.py             # Task 基类
│   │       ├── detection.py        # DetectionTask 实现
│   │       ├── conditions.py       # DoneCondition 接口和实现
│   │       └── stats.py            # TaskStats 统计
│   │
│   └── api/                        # REST API
│       ├── app.py                  # FastAPI 应用
│       └── routes/                 # 路由定义
│
├── config/                         # 配置文件
│   ├── settings.yaml               # 主配置
│   └── settings.dev.yaml           # 开发环境配置
│
├── assets/                         # 模型和资源文件
├── tests/                          # 测试代码
├── main.py                         # 主程序入口
├── docker-compose.yml              # MinIO + PostgreSQL
└── pyproject.toml                  # 项目依赖
```

**模块职责**：

| 目录 | 职责 | ROS2 复用 |
|------|------|----------|
| `core/camera/` | 相机抽象接口 | ✅ |
| `core/vision/` | Pipeline 和检测步骤（无状态） | ✅ |
| `storage/` | 数据存储抽象 | ✅ |
| `scheduler/tasks/` | Task 层（有状态） | ✅ |
| `scheduler/task_manager.py` | 主循环调度 | ❌ Backend 专用 |
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
| `Detection` | bbox, object_type, confidence, detection_id | 单个检测结果 |
| `PipelineContext` | original_image, processed_image, detections, metadata | 步骤间传递的上下文 |
| `PipelineResult` | detections, processing_time_ms, metadata | Pipeline 最终输出 |

#### Pipeline 接口

```
┌───────────────────────────────────────────────────────────────────────┐
│                         VisionPipeline                                │
│                                                                       │
│   特性：无状态，纯函数式，可配置的步骤组合                             │
├───────────────────────────────────────────────────────────────────────┤
│ + add_step(step: ProcessStep) -> VisionPipeline                       │
│ + run(image: np.ndarray) -> PipelineResult                            │
│ + from_config(config: dict) -> VisionPipeline                         │
│ + clear() -> None                                                     │
└───────────────────────────────────────────────────────────────────────┘

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
| `tile` | 图像切片 | tile_size, overlap |
| `yolo` | YOLO 检测 | model, conf, device |
| `merge_tiles` | 合并切片结果 | iou_threshold |
| `nms` | 非极大值抑制 | iou_threshold |
| `filter` | 过滤检测结果 | min_confidence, classes |
| `sort` | 结果排序 | by, ascending |

**配置约定**：

```yaml
vision:
  pipeline:
    steps:
      - name: tile
        type: tile
        params:
          tile_size: 640
          overlap: 0.2
      - name: detect
        type: yolo
        params:
          model: assets/best.pt
          conf: 0.25
      - name: merge
        type: merge_tiles
        params:
          iou_threshold: 0.5
```

### 4.3 任务模块 (scheduler/tasks/)

#### Task 输出契约

```
┌───────────────────────────────────────────────────────────────────────┐
│                      TaskIterationResult                              │
│                                                                       │
│   Task 单次迭代的输出，供 TaskManager 处理存储/事件                    │
├───────────────────────────────────────────────────────────────────────┤
│ + detections: List[Detection]      # 检测/跟踪结果                    │
│ + is_done: bool                    # 任务是否完成                     │
│ + metadata: Dict[str, Any]         # 元数据（帧计数、确认数等）        │
└───────────────────────────────────────────────────────────────────────┘
```

#### Task 接口

```
┌───────────────────────────────────────────────────────────────────────┐
│                             Task                                      │
│                                                                       │
│   有状态的任务单元，持有 Tracker/Stats/DoneCondition                   │
│   Pipeline 通过构造函数注入，不自己创建                                │
├───────────────────────────────────────────────────────────────────────┤
│ + name: str                                                           │
│ + run_iteration(image: np.ndarray) -> TaskIterationResult            │
│ + reset() -> None                    # 区域/任务切换时调用             │
│ + get_stats_summary() -> Dict       # 获取统计摘要                    │
└───────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ 实现
                                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        DetectionTask                                  │
│                                                                       │
│   默认实现：包装 Pipeline + 可选 ByteTrack 跟踪                        │
├───────────────────────────────────────────────────────────────────────┤
│ - pipeline: VisionPipeline         # 注入                             │
│ - tracker: ByteTracker (可选)      # 跨帧状态                         │
│ - stats: TaskStats                 # 统计                             │
│ - done_condition: DoneCondition    # 完成判定                         │
└───────────────────────────────────────────────────────────────────────┘
```

#### Pipeline 与 Tracker 解耦设计

**核心原则**：Pipeline 保持无状态，Tracker 状态由 Task 层管理。

**数据流详解**：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              第 N 帧                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Pipeline（无状态）                                                      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ TileStep: 把大图切成 N 个 640×640 的小块                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ YOLODetectStep: 对每个 tile 做检测                                  │ │
│  │ tile1 → 5个框, tile2 → 3个框, tile3 → 2个框 ...                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ MergeTilesStep: 合并所有 tile 的框，坐标转回原图，NMS 去重           │ │
│  │ 输出: [Detection, Detection, ...]  (比如 8 个原始检测框)            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Pipeline 完成，输出 8 个检测框（原图坐标）                              │
│  Pipeline 不知道这是第几帧，不关心上一帧，纯粹处理当前图                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  8 个检测框（只是数据，List[Detection]）
                                  │
                            ══════╪══════  解耦边界
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Task（有状态）                                                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ ByteTracker.update(8个框)                                           │ │
│  │                                                                     │ │
│  │ Tracker 内部状态（内存中维护）：                                     │ │
│  │ • 记得上一帧有 track_1, track_2, track_3 等轨迹                     │ │
│  │ • 用 IoU/卡尔曼滤波 匹配这 8 个框和历史轨迹                         │ │
│  │ • 匹配上的框 → 更新轨迹位置，hits + 1                               │ │
│  │ • 没匹配的新框 → 创建新轨迹，hits = 1                               │ │
│  │ • 没匹配到框的旧轨迹 → age + 1，超过 track_buffer 则删除            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                  │                                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ 过滤 min_hits：只保留连续出现 >= N 帧的轨迹（confirmed tracks）      │ │
│  │                                                                     │ │
│  │ 8个原始框 → 过滤后剩 5 个稳定框（3个闪烁的假阳性被过滤）            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  输出: 5 个稳定的检测框 + track_id                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**为什么能共存**：

| 层 | 关心什么 | 不关心什么 |
|---|---------|-----------|
| **Pipeline** | 当前这张图怎么处理 | 上一帧是什么、下一帧是什么 |
| **Tracker** | 这堆框和历史轨迹怎么匹配 | 框是怎么检测出来的（tile/全图/哪个模型） |

**解耦点**：Pipeline 输出 `List[Detection]` 是纯数据（x1, y1, x2, y2, conf, class），Tracker 只需要这个列表，不需要知道检测框是 tile 模式还是全图模式产生的。

**Tracker 状态生命周期**：

| 事件 | 处理方式 |
|------|---------|
| 每帧调用 | `tracker.update(detections)` 更新内部状态 |
| 区域切换 | `task.reset()` → 重建 tracker，清空历史轨迹 |
| 新 Session | `task.reset()` → 重建 tracker |
| 程序退出 | 状态自动释放（内存中，不持久化） |

#### DoneCondition 接口

```
┌───────────────────────────────────────────────────────────────────────┐
│                        DoneCondition                                  │
│                                                                       │
│   任务完成条件判定接口                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ + check(stats, result) -> bool     # 检查是否满足完成条件              │
│ + reset() -> None                  # 重置条件状态                     │
└───────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ 内置实现
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
   ConsecutiveEmpty        MaxIterations          Timeout
   (连续N帧无检测)          (最大迭代数)           (超时)
```

**内置完成条件**：

| 条件 | 说明 | 参数 |
|------|------|------|
| `ConsecutiveEmptyFrames` | 连续 N 帧无检测则完成 | n |
| `MaxIterations` | 达到最大迭代次数 | max_iter |
| `Timeout` | 超时 | seconds |
| `Composite` | 组合条件（任一满足即完成） | conditions |

#### TaskStats

```
┌───────────────────────────────────────────────────────────────────────┐
│                          TaskStats                                    │
│                                                                       │
│   任务运行统计                                                         │
├───────────────────────────────────────────────────────────────────────┤
│ + total_frames: int                # 总帧数                           │
│ + total_detections: int            # 总检测数                         │
│ + confirmed_detections: int        # 经 Tracking 确认的检测数         │
│ + start_time: datetime             # 开始时间                         │
│ + end_time: datetime               # 结束时间                         │
├───────────────────────────────────────────────────────────────────────┤
│ + record(result) -> None           # 记录一次迭代结果                  │
│ + summary() -> Dict                # 获取统计摘要                     │
│ + reset() -> None                  # 重置统计                         │
└───────────────────────────────────────────────────────────────────────┘
```

**Task 配置约定**：

```yaml
task:
  name: "detection"
  # pipeline_id: "default"   # 预留：未来多模型时使用
  
  tracker:
    enabled: true
    track_thresh: 0.5
    track_buffer: 30
    match_thresh: 0.8
    min_hits: 3
  
  done_condition:
    consecutive_empty: 3     # 连续 N 帧无检测
    max_iterations: 1000     # 最大迭代数
    # timeout_seconds: 60    # 超时（可选）
```

### 4.4 调度模块 (scheduler/)

#### TaskManager 接口

```
┌───────────────────────────────────────────────────────────────────────┐
│                          TaskManager                                  │
│                                                                       │
│   主循环调度器                                                         │
│   职责：采集控制、调用 Task、存储/事件推送、会话管理                    │
├───────────────────────────────────────────────────────────────────────┤
│ - camera: CameraBase                                                  │
│ - pipeline: VisionPipeline          # 创建并注入给 Task               │
│ - task: Task                        # 当前执行的任务                  │
│ - image_storage: ImageStorage                                         │
│ - database: Database                                                  │
│ - storage_saver: StorageSaver                                         │
├───────────────────────────────────────────────────────────────────────┤
│ + start() -> None                                                     │
│ + stop() -> None                                                      │
│ + is_running: bool                                                    │
│ + session_id: str                                                     │
│ + frame_count: int                                                    │
└───────────────────────────────────────────────────────────────────────┘
```

**主循环流程**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         主循环 _process_frame()                      │
│                                                                      │
│   1. camera.capture() ──────────────────► image                     │
│                                              │                       │
│   2. task.run_iteration(image) ◄─────────────┘                      │
│              │                                                       │
│              ▼                                                       │
│        TaskIterationResult                                           │
│        • detections                                                  │
│        • is_done                                                     │
│        • metadata                                                    │
│              │                                                       │
│   3. storage_saver.save() ◄──────────────────┘                      │
│                                                                      │
│   4. publish_event()                                                 │
│                                                                      │
│   5. if result.is_done: handle_task_done()                          │
│                                                                      │
│   横切关注点（存储、事件、预览）统一由 TaskManager 处理               │
└─────────────────────────────────────────────────────────────────────┘
```

**配置约定**：

```yaml
scheduler:
  loop_delay_ms: 100
  max_errors: 10
  save_images: true
  save_annotated: true
  show_preview: true
```

### 4.5 存储模块 (storage/)

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
│ + create_session(session) -> None        │
│ + update_session(session) -> None        │
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
| `sessions` | id, start_time, end_time, total_frames, total_detections, status | 运行会话 |

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

### 4.6 API 模块 (api/)

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
      - { name: tile, type: tile, params: { tile_size: 640, overlap: 0.2 } }
      - { name: detect, type: yolo, params: { model: assets/best.pt, conf: 0.25 } }
      - { name: merge, type: merge_tiles, params: { iou_threshold: 0.5 } }

task:
  name: "detection"
  tracker:
    enabled: true
    track_thresh: 0.5
    track_buffer: 30
    min_hits: 3
  done_condition:
    consecutive_empty: 3
    max_iterations: 1000

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
  save_images: true
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
  
  redis:
    image: redis:7
    ports: ["6379:6379"]
```

### 5.3 依赖项

```
# 核心依赖
numpy, opencv-python, ultralytics, torch

# Tracking（可选）
boxmot  # or bytetrack

# 数据库
sqlalchemy, psycopg2-binary

# 对象存储
minio

# 事件流
redis

# API
fastapi, uvicorn, pydantic

# 配置
pyyaml

# 相机（需单独安装 SDK）
gxipy
```

---

## 6. 演进路线

### 6.1 阶段 1：单任务 + Tracking（当前）

```
┌─────────────────────────────────────────────────────────────────────┐
│                            当前目标                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • 定义 Task 基类、TaskIterationResult、DoneCondition 接口          │
│  • 实现 DetectionTask（包装 Pipeline + 可选 ByteTrack）             │
│  • 修改 TaskManager 调用 task.run_iteration()                       │
│  • 确保存储/事件流程不变，功能不回退                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 阶段 2：多任务支持（未来）

```
┌─────────────────────────────────────────────────────────────────────┐
│                            未来目标                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • 实现 WorkflowEngine 编排多任务                                    │
│  • 支持 A → B → C 任务序列配置                                       │
│  • 任务级统计汇总和事件推送                                          │
│  • 多模型支持：Pipeline 缓存/释放机制                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 与 ROS2 集成

```
backend/src/core/  ────────►  pluck_ws/src/applications/
backend/src/scheduler/tasks/       │
     │                             │
     │  import 或 pip install      ▼
     └───────────────────►  detector_node.py
                           (调用 core.vision, scheduler.tasks)
```

### 6.4 性能优化路径

1. **TensorRT 加速**：YOLO 转 TensorRT 引擎
2. **异步存储**：图像存储使用线程池（已实现）
3. **C++ 关键模块**：必要时用 pybind11 封装

---

## 附录

### 术语表

| 术语 | 说明 |
|------|------|
| Pipeline | 无状态的多步骤图像处理流程 |
| Task | 有状态的任务单元，持有 Tracker/Stats |
| Workflow | 多任务编排（未来） |
| Detection | 单个异物检测结果 |
| Tracking | 跨帧目标跟踪，过滤闪烁检测、稳定边界框 |
| Session | 一次运行会话 |
| DoneCondition | 任务完成条件 |
| NMS | Non-Maximum Suppression，非极大值抑制 |

### 相关文档

- [大恒图像 Galaxy SDK](https://www.daheng-imaging.com/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [MinIO Python SDK](https://min.io/docs/minio/linux/developers/python/API.html)
- [SQLAlchemy](https://docs.sqlalchemy.org/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**文档状态**：✅ 设计完成  
**代码实现**：见 `backend/` 目录
