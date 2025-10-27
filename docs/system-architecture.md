# 燕窝挑毛系统 - 系统架构设计

## 文档说明
本文档定义燕窝挑毛系统的整体架构设计，包括代码组织结构、模块划分、技术框架集成方案等。

**最后更新：** 2025-10-26
**文档版本：** v1.0
**关联文档：**
- [architecture-decisions.md](architecture-decisions.md) - 架构决策记录
- [requirement.md](requirement.md) - 需求文档
- [workflow-diagrams-noMoveIt.md](workflow-diagrams-noMoveIt.md) - 工作流程图

---

## 1. 文档概述

### 1.1 目的
本文档旨在提供燕窝挑毛系统PC端ROS 2系统的完整架构设计，包括：
- 系统整体架构和边界
- ROS 2 workspace和package组织结构
- 代码目录设计和编程语言分工
- 各技术框架的集成方案
- 配置、日志、数据管理设计

### 1.2 适用范围
- 系统架构师：理解整体设计思路
- 开发人员：了解代码组织和模块职责
- 测试人员：理解系统边界和模块依赖

### 1.3 设计原则
- **模块化**：各模块职责清晰，低耦合高内聚
- **可替换性**：Python原型可无缝替换为C++实现
- **可扩展性**：预留未来功能扩展接口
- **实时性**：关键路径优化，满足伺服控制要求
- **可维护性**：清晰的代码组织和配置管理

---

## 2. 系统整体架构

### 2.1 系统边界

```
┌─────────────────────────────────────────────────────────────────┐
│                        完整系统边界                              │
│                                                                  │
│  ┌────────────────────────────────┐   ┌──────────────────────┐ │
│  │   PC工控机 (ROS 2 系统)        │   │  PLC + 三台机械臂    │ │
│  │                                 │   │                      │ │
│  │  • 视觉处理                     │   │  • 压板臂            │ │
│  │  • 深度学习推理                 │   │  • 视觉臂            │ │
│  │  • IBVS伺服计算                 │◄─►│  • 夹取臂            │ │
│  │  • 任务调度                     │   │  • 运动控制          │ │
│  │  • 数据记录                     │   │  • 协同状态机        │ │
│  │                                 │   │                      │ │
│  └────────────────────────────────┘   └──────────────────────┘ │
│           │                                                      │
│           │ USB 3.0                                              │
│           ▼                                                      │
│  ┌────────────────────────────────┐                             │
│  │      工业相机                   │                             │
│  │  • 图像采集                     │                             │
│  │  • 参数可调                     │                             │
│  └────────────────────────────────┘                             │
│                                                                  │
│  外部存储：数据库（图像、日志、统计数据）                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

系统边界说明：
• ROS 2系统运行在PC工控机上（本文档重点）
• PLC系统独立运行（通信接口待细化）
• 相机通过USB直连PC（驱动在ROS 2系统内）
```

### 2.2 ROS 2系统总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│          燕窝挑毛系统 - PC端（基于ROS 2 Humble）                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │        业务逻辑层 (applications包 - ROS适配层)              ││
│  │                                                               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ││
│  │  │  托盘定位    │  │  异物检测    │  │  视觉伺服    │      ││
│  │  │   业务       │  │   业务       │  │   业务       │      ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘      ││
│  │  ┌──────────────┐  ┌──────────────┐                        ││
│  │  │  任务调度    │  │  坐标管理    │  ROS节点：负责通信    ││
│  │  │   业务       │  │   业务       │  和消息转换            ││
│  │  └──────────────┘  └──────────────┘                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                       ↕ 直接调用（函数调用）                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │        核心能力层 (core包 - 框架无关的纯库)                 ││
│  │                                                               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ││
│  │  │  vision能力  │  │  motion能力  │  │  state能力   │      ││
│  │  │              │  │              │  │              │      ││
│  │  │ •异物检测器  │  │ •坐标转换    │  │ •状态机      │      ││
│  │  │ •托盘定位器  │  │ •运动学计算  │  │ •异常处理    │      ││
│  │  │ •IBVS控制器  │  │ •轨迹规划    │  │              │      ││
│  │  │ •图像处理    │  │              │  │              │      ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘      ││
│  │                                                               ││
│  │  特点：不依赖ROS，可独立测试和复用                           ││
│  └─────────────────────────────────────────────────────────────┘│
│                       ↕ ROS Topic/Service                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            硬件接口层 (hardware包)                           ││
│  │                                                               ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      ││
│  │  │  相机驱动    │  │  PLC通信     │  │  数据存储    │      ││
│  │  │ • USB接口    │  │ • PROFINET   │  │ • 数据库     │      ││
│  │  │ • 图像发布   │  │  (待细化)    │  │ • 文件系统   │      ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘      ││
│  └─────────────────────────────────────────────────────────────┘│
│                       ↕                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              物理硬件                                         ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  ││
│  │  │   相机   │  │   PLC    │  │  数据库  │                  ││
│  │  └──────────┘  └──────────┘  └──────────┘                  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  横向支撑（utils包）：                                            │
│  • 配置管理  • 日志管理  • 工具脚本  • 性能监控                  │
│                                                                  │
│  关键设计原则：                                                   │
│  ✓ 依赖倒置：applications依赖core，core不依赖applications        │
│  ✓ 框架隔离：core完全不知道ROS的存在                             │
│  ✓ 按能力分层：core按技术能力（vision/motion/state）             │
│  ✓ 按业务分层：applications按业务功能（托盘定位/异物检测等）     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 关键技术栈汇总

| 层次 | 组件 | 技术栈 | 说明 |
|------|------|--------|------|
| **操作系统** | OS | Ubuntu 22.04 LTS | ROS 2 Humble官方支持 |
| **机器人框架** | Framework | ROS 2 Humble | 长期支持版本 |
| **深度学习** | 推理引擎 | TensorRT | GPU加速推理 |
| **深度学习** | 训练框架 | PyTorch/ONNX | 模型训练和转换 |
| **视觉伺服** | IBVS库 | ViSP | 成熟的视觉伺服库 |
| **图像处理** | 算法库 | OpenCV | 托盘定位、边缘检测 |
| **坐标管理** | TF库 | tf2_ros | ROS 2标准TF库 |
| **相机接口** | 驱动 | usb_cam / pylon_ros2 | 根据相机型号选择 |
| **PLC通信** | 协议 | PROFINET | 待细化实现方案 |
| **数据存储** | 数据库 | 待定 (SQLite/PostgreSQL) | 图像和日志存储 |
| **编程语言** | 核心 | C++ 14/17 | 性能关键模块 |
| **编程语言** | 原型/工具 | Python 3.10+ | 快速原型和脚本 |

---

## 3. ROS 2 Workspace和Package组织

### 3.1 Workspace结构

```
pluck_ws/                          # ROS 2 工作空间根目录
├── src/                           # 源代码目录
│   ├── core/                      # 核心能力包（框架无关的纯库）
│   ├── applications/              # 业务逻辑包（ROS节点，组装core能力）
│   ├── hardware/                  # 硬件接口包
│   ├── msgs/                      # 自定义消息定义包
│   └── utils/                     # 工具和配置包
├── install/                       # 安装目录（colcon build生成）
├── build/                         # 构建目录
├── log/                           # 构建日志
└── README.md                      # Workspace说明文档
```

### 3.2 Package划分方案

#### 3.2.1 Package总览

| Package名称 | 类型 | 主要职责 | 依赖 |
|------------|------|----------|------|
| **msgs** | 消息定义 | 定义所有自定义msg/srv/action | 无ROS包依赖 |
| **core** | 核心能力库 | 视觉处理、运动计算、状态管理（框架无关的纯库） | OpenCV, TensorRT, ViSP等基础库 |
| **applications** | 业务逻辑 | 托盘定位、异物检测、视觉伺服等业务（ROS节点） | core, msgs, ROS 2相关包 |
| **hardware** | 硬件接口 | 相机驱动、PLC通信 | msgs, usb_cam/pylon_ros2 |
| **utils** | 工具支撑 | 配置管理、日志工具、测试脚本 | msgs |

#### 3.2.2 Package依赖关系图

```
                    ┌─────────────┐
                    │    msgs     │ (消息定义，无依赖)
                    └──────┬──────┘
                           │
                           │
                           ▼
                    ┌─────────────┐
                    │    core     │ (核心能力，框架无关)
                    │  • vision   │ 只依赖基础库：
                    │  • motion   │ OpenCV, TensorRT,
                    │  • state    │ ViSP, numpy等
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────┐
      │ applications │ │ hardware │ │  utils   │
      │ (ROS节点)    │ │          │ │          │
      └──────────────┘ └──────────┘ └──────────┘
       依赖: core,      依赖: msgs   依赖: msgs
             msgs,
             ROS 2

依赖方向：向上依赖，core不依赖任何ROS包
```

---

## 4. 代码目录结构详细设计

### 4.1 msgs（消息定义包）

```
msgs/
├── msg/                           # 消息类型定义
│   ├── ForeignObject.msg          # 异物信息
│   ├── TrayPose.msg               # 托盘位姿
│   ├── ArmStatus.msg              # 机械臂状态
│   ├── RegionInfo.msg             # 区域信息
│   └── SystemState.msg            # 系统状态
├── srv/                           # 服务接口定义
│   ├── DetectObjects.srv          # 异物检测服务
│   ├── LocateTray.srv             # 托盘定位服务
│   ├── IbvsServo.srv              # IBVS伺服服务
│   └── GetRegionStatus.srv        # 获取区域状态
├── action/                        # 动作接口定义（可选）
│   ├── PickObject.action          # 夹取异物动作
│   └── ProcessRegion.action       # 处理区域动作
├── CMakeLists.txt
└── package.xml
```

**设计说明：**
- 纯消息定义包，无代码实现
- 被所有其他包依赖
- 接口一旦确定，尽量保持稳定

### 4.2 core（核心能力包 - 框架无关）

```
core/
├── core/                          # Python包目录（纯库，不依赖ROS）
│   ├── __init__.py
│   │
│   ├── vision/                    # 视觉处理能力
│   │   ├── __init__.py
│   │   ├── object_detector.py     # 异物检测器（纯类）
│   │   ├── tray_locator.py        # 托盘定位器（纯类）
│   │   ├── ibvs_controller.py     # IBVS控制器（纯类）
│   │   └── image_processor.py     # 图像处理工具
│   │
│   ├── motion/                    # 运动计算能力
│   │   ├── __init__.py
│   │   ├── coordinate_transformer.py  # 坐标转换
│   │   ├── kinematics.py          # 运动学计算
│   │   └── trajectory_planner.py  # 轨迹规划（预留）
│   │
│   └── state/                     # 状态管理能力
│       ├── __init__.py
│       ├── state_machine.py       # 状态机（纯类）
│       └── error_handler.py       # 异常处理器
│
├── src/                           # C++源码（后期性能优化）
│   ├── vision/
│   │   ├── object_detector.cpp
│   │   ├── tensorrt_engine.cpp    # TensorRT推理引擎
│   │   └── ibvs_controller.cpp
│   ├── motion/
│   │   └── coordinate_transformer.cpp
│   └── state/
│       └── state_machine.cpp
│
├── include/core/                  # C++头文件
│   ├── vision/
│   │   ├── object_detector.hpp
│   │   └── tensorrt_engine.hpp
│   ├── motion/
│   │   └── coordinate_transformer.hpp
│   └── state/
│       └── state_machine.hpp
│
├── models/                        # 深度学习模型文件
│   ├── foreign_object_detector.onnx
│   ├── foreign_object_detector.trt
│   └── model_config.yaml
│
├── test/                          # 单元测试（不需要ROS环境）
│   ├── test_object_detector.py
│   ├── test_ibvs_controller.py
│   └── test_state_machine.py
│
├── scripts/                       # 工具脚本
│   ├── convert_onnx_to_trt.py     # 模型转换
│   └── benchmark_inference.py     # 性能测试
│
├── CMakeLists.txt
├── package.xml                    # 只依赖：OpenCV, numpy, ViSP等
└── README.md
```

**设计说明：**
- ✅ **完全框架无关**：不依赖ROS，可在任何项目中使用
- ✅ **按能力模块组织**：vision/motion/state清晰分离
- ✅ **可独立测试**：test/中的测试不需要ROS环境
- ✅ **Python→C++替换**：接口一致，实现可替换

### 4.3 applications（业务逻辑包 - ROS适配层）

```
applications/
├── applications/                  # Python包目录（ROS节点）
│   ├── __init__.py
│   │
│   ├── tray_localization/         # 托盘定位业务
│   │   ├── __init__.py
│   │   └── tray_locator_node.py   # ROS节点，调用core.vision.tray_locator
│   │
│   ├── object_detection/          # 异物检测业务
│   │   ├── __init__.py
│   │   └── object_detector_node.py  # ROS节点，调用core.vision.object_detector
│   │
│   ├── visual_servoing/           # 视觉伺服业务
│   │   ├── __init__.py
│   │   └── ibvs_servo_node.py     # ROS节点，调用core.vision.ibvs_controller
│   │
│   ├── task_scheduling/           # 任务调度业务
│   │   ├── __init__.py
│   │   └── task_scheduler_node.py # ROS节点，编排整体工作流
│   │
│   ├── coordinate_management/     # 坐标管理业务
│   │   ├── __init__.py
│   │   └── tf_manager_node.py     # ROS节点，管理TF树
│   │
│   └── utils/                     # 业务层辅助工具
│       ├── __init__.py
│       └── ros_converters.py      # ROS消息转换工具
│
├── src/                           # C++版本（后期）
│   ├── object_detection/
│   │   └── object_detector_node.cpp
│   └── visual_servoing/
│       └── ibvs_servo_node.cpp
│
├── launch/                        # Launch文件
│   ├── full_system.launch.py      # 完整系统启动
│   ├── tray_localization.launch.py
│   ├── object_detection.launch.py
│   └── visual_servoing.launch.py
│
├── config/                        # 配置文件
│   ├── tray_localization.yaml
│   ├── object_detection.yaml
│   ├── ibvs.yaml
│   ├── workspace_poses.yaml       # 预定义位姿
│   └── hand_eye_calibration.yaml  # 手眼标定
│
├── test/                          # 集成测试（需要ROS环境）
│   ├── test_tray_localization_node.py
│   ├── test_object_detection_node.py
│   └── test_task_scheduler.py
│
├── CMakeLists.txt
├── package.xml                    # 依赖：core, msgs, rclpy/rclcpp
└── README.md
```

**设计说明：**
- ✅ **按业务功能组织**：托盘定位、异物检测、视觉伺服等
- ✅ **ROS适配器模式**：节点负责ROS通信，业务逻辑在core
- ✅ **职责单一**：只做消息转换和服务提供
- ✅ **便于扩展**：新增业务功能，新建文件夹即可

### 4.4 hardware（硬件接口包）

```
hardware/
├── hardware/                      # Python包目录
│   ├── __init__.py
│   ├── camera_driver.py           # 相机驱动节点（如需自定义）
│   └── plc_interface.py           # PLC通信接口（待实现）
├── src/                           # C++源码
│   ├── camera_driver_node.cpp     # 相机驱动（C++）
│   └── plc_interface_node.cpp     # PLC通信（C++，待实现）
├── include/hardware/
│   ├── camera_driver.hpp
│   └── plc_interface.hpp
├── launch/
│   ├── camera.launch.py           # 相机启动
│   └── plc.launch.py              # PLC通信启动
├── config/
│   ├── camera.yaml                # 相机参数（分辨率、曝光等）
│   └── plc.yaml                   # PLC通信配置（待定）
├── test/
│   └── test_camera.py
├── CMakeLists.txt
├── package.xml
└── README.md
```

**设计说明：**
- 相机驱动：优先使用现成ROS 2包（如usb_cam、pylon_ros2_camera）
- 如需自定义功能，在此包中封装
- PLC通信：接口预留，后续细化

### 4.5 utils（工具包）

```
utils/
├── utils/                         # Python工具
│   ├── __init__.py
│   ├── config_loader.py           # 配置加载工具
│   ├── logger.py                  # 日志管理
│   └── database_manager.py        # 数据库接口
├── scripts/                       # 独立脚本
│   ├── setup_environment.sh       # 环境配置脚本
│   ├── calibrate_camera.py        # 相机标定工具
│   ├── calibrate_hand_eye.py      # 手眼标定工具
│   └── visualize_results.py       # 结果可视化
├── launch/
│   └── logging.launch.py          # 日志系统启动
├── config/
│   └── logging.yaml               # 日志配置
├── test/
│   └── test_config_loader.py
├── CMakeLists.txt
├── package.xml
└── README.md
```

**设计说明：**
- 提供通用工具函数
- 配置和日志管理
- 标定和调试脚本

---

## 5. 模块架构设计

### 5.1 视觉处理模块

#### 5.1.1 模块组成

```
┌─────────────────────────────────────────────────────────┐
│              视觉处理模块架构                            │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │          异物检测子模块                           │  │
│  │                                                    │  │
│  │  输入: sensor_msgs/Image                          │  │
│  │  ┌─────────────┐      ┌──────────────┐           │  │
│  │  │ 图像预处理  │ ───► │ 深度学习推理 │           │  │
│  │  │ • 去噪      │      │ • TensorRT   │           │  │
│  │  │ • 增强      │      │ • ONNX Runtime│          │  │
│  │  └─────────────┘      └──────┬───────┘           │  │
│  │                              │                    │  │
│  │                              ▼                    │  │
│  │                     ┌──────────────┐              │  │
│  │                     │ 后处理       │              │  │
│  │                     │ • NMS去重    │              │  │
│  │                     │ • 坐标转换   │              │  │
│  │                     └──────┬───────┘              │  │
│  │  输出: ForeignObject[]     │                      │  │
│  └────────────────────────────┼──────────────────────┘  │
│                                │                         │
│  ┌────────────────────────────┼──────────────────────┐  │
│  │          托盘定位子模块     │                      │  │
│  │                            │                      │  │
│  │  输入: sensor_msgs/Image   │                      │  │
│  │  ┌─────────────┐      ┌───▼──────────┐           │  │
│  │  │ 边缘检测    │ ───► │ 边缘拟合     │           │  │
│  │  │ • Canny     │      │ • RANSAC     │           │  │
│  │  │ • 形态学    │      │ • 最小二乘   │           │  │
│  │  └─────────────┘      └──────┬───────┘           │  │
│  │                              │                    │  │
│  │  输出: TrayPose              │                    │  │
│  └────────────────────────────┼──────────────────────┘  │
│                                │                         │
│  ┌────────────────────────────┼──────────────────────┐  │
│  │         IBVS伺服子模块      │                      │  │
│  │                            │                      │  │
│  │  输入: 连续图像流           │                      │  │
│  │  ┌─────────────┐      ┌───▼──────────┐           │  │
│  │  │ 特征检测    │ ───► │ ViSP伺服     │           │  │
│  │  │ • 异物位置  │      │ • 图像雅可比 │           │  │
│  │  │ • 夹爪标记  │      │ • 速度计算   │           │  │
│  │  └─────────────┘      └──────┬───────┘           │  │
│  │                              │                    │  │
│  │  输出: 增量位姿指令          │                    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 5.2 运动计算模块

#### 5.2.1 TF树设计

```
world (全局坐标系，固定)
  │
  ├─ tray_frame (托盘坐标系，定位后发布)
  │    │
  │    └─ region_N_frame (各区域坐标系，按需发布)
  │
  └─ camera_optical_frame (相机坐标系，动态发布)
       │
       └─ detected_object_frame (检测到的异物，临时)
```

**TF发布策略：**
- `world` → `tray_frame`：托盘定位后发布（静态）
- `world` → `camera_optical_frame`：基于预定义位姿动态发布
- 手眼标定参数：离线标定，启动时加载

### 5.4 硬件接口模块

#### 5.4.1 相机驱动集成

**方案选择：**
- **通用USB相机**：使用 `usb_cam` (ROS 2包)
- **Basler工业相机**：使用 `pylon_ros2_camera`
- **自定义需求**：在hardware包中封装SDK

**配置示例：**
```yaml
# hardware/config/camera.yaml
camera:
  device: "/dev/video0"           # USB设备路径
  frame_id: "camera_optical_frame"
  image_width: 2448
  image_height: 2048
  pixel_format: "bgr8"
  framerate: 60
  exposure: 10000                  # 微秒
  gain: 1.0
```

**Launch集成：**
```python
# hardware/launch/camera.launch.py
def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='camera',
            parameters=[camera_config],
            remappings=[('/image_raw', '/pluck/camera/image')]
        )
    ])
```

#### 5.4.2 PLC通信接口（预留）

```python
# hardware/hardware/plc_interface.py
class PLCInterface(Node):
    """PLC通信接口（待实现）"""

    def __init__(self):
        super().__init__('plc_interface')
        # TODO: 初始化PROFINET连接

    def send_command(self, cmd):
        """发送指令到PLC"""
        # TODO: 实现通信协议
        pass

    def receive_event(self):
        """接收PLC事件"""
        # TODO: 实现事件监听
        pass
```

---

## 6. 技术框架集成方案

### 6.1 TensorRT深度学习推理

#### 6.1.1 开发流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  模型训练   │ ──► │ 导出ONNX    │ ──► │ 转换TRT     │
│  PyTorch    │     │             │     │ FP16/INT8   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ ROS 2节点   │
                                        │ 推理引擎    │
                                        └─────────────┘
```

#### 6.1.2 部署方案

**目录组织：**
```
core/models/                           # 所有模型统一放在core包
├── foreign_object_detector.onnx       # ONNX模型（Python用）
├── foreign_object_detector.trt        # TensorRT引擎（C++用）
├── labels.txt                         # 类别标签
└── model_config.yaml                  # 模型元数据
```

**配置文件：**
```yaml
# core/models/model_config.yaml
model:
  name: "foreign_object_detector"
  version: "v1.0"
  input_shape: [1, 3, 640, 640]       # NCHW
  input_type: "float32"
  classes: ["hair", "black_spot", "yellow_spot"]
  confidence_threshold: 0.5
  nms_threshold: 0.4
```

### 6.2 ViSP视觉伺服集成

#### 6.2.1 Python绑定使用

```bash
# 安装ViSP Python绑定
pip install visp
```

```python
# 示例代码
from visp.core import vpImage, vpCameraParameters
from visp.vs import vpServo
from visp.visual_features import vpFeaturePoint

servo = vpServo()
servo.setServo(vpServo.EYEINHAND_CAMERA)
servo.setLambda(0.5)  # 控制增益
```

#### 6.2.2 C++ API使用

```cmake
# CMakeLists.txt
find_package(VISP REQUIRED)
target_link_libraries(${PROJECT_NAME} ${VISP_LIBRARIES})
```

```cpp
#include <visp3/vs/vpServo.h>
vpServo servo;
servo.setServo(vpServo::EYEINHAND_CAMERA);
```

### 6.3 OpenCV图像处理

**用途：**
- 托盘边缘检测（Canny + Hough变换）
- 图像预处理（去噪、增强）
- 夹爪标记检测（颜色分割、轮廓提取）

**集成：**
```python
# Python
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
```

```cpp
// C++
#include <cv_bridge/cv_bridge.h>
cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
```

### 6.4 tf2坐标管理

#### 6.4.1 静态TF发布

```python
# motion/motion/tf_manager.py
from tf2_ros import StaticTransformBroadcaster

class TFManager(Node):
    def __init__(self):
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_hand_eye_calibration()

    def publish_hand_eye_calibration(self):
        """发布手眼标定结果"""
        t = TransformStamped()
        t.header.frame_id = "camera_link"
        t.child_frame_id = "camera_optical_frame"
        # 从配置文件加载标定参数
        t.transform.translation.x = ...
        self.static_broadcaster.sendTransform(t)
```

#### 6.4.2 动态TF发布

```python
def publish_tray_tf(self, tray_pose):
    """发布托盘坐标系"""
    t = TransformStamped()
    t.header.stamp = self.get_clock().now().to_msg()
    t.header.frame_id = "world"
    t.child_frame_id = "tray_frame"
    t.transform.translation.x = tray_pose.x
    # ...
    self.dynamic_broadcaster.sendTransform(t)
```

---

## 7. 编程语言分工

### 7.1 Python模块（初期原型）

| 模块 | 原因 |
|------|------|
| 深度学习推理 | PyTorch/ONNX Runtime生态成熟 |
| 图像处理 | OpenCV Python API友好，快速调试 |
| IBVS算法 | ViSP Python绑定可用 |
| 任务调度 | 快速实现状态机逻辑 |
| 工具脚本 | 标定、可视化、测试 |

### 7.2 C++模块（后期性能优化）

| 模块 | 原因 |
|------|------|
| 深度学习推理 | TensorRT C++ API性能最优 |
| IBVS伺服 | 实时性要求高（30-60Hz） |
| PLC通信 | 底层协议栈通常是C/C++ |
| TF管理 | 高频发布需要低延迟 |

### 7.3 Python→C++替换策略

#### 7.3.1 接口一致性保证

**关键原则：**
1. **Core层接口保持一致**：Python和C++实现相同的类和方法
2. **Applications层ROS接口保持一致**：服务名、话题名、消息类型不变

**示例1：Core层接口一致**

Python版本：
```python
# core/core/vision/object_detector.py
class ObjectDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        pass

    def detect(self, image: np.ndarray) -> List[ForeignObject]:
        """检测异物"""
        pass
```

C++版本（接口完全一致）：
```cpp
// core/src/vision/object_detector.cpp
class ObjectDetector {
public:
    ObjectDetector(const std::string& model_path, float confidence_threshold = 0.5);

    std::vector<ForeignObject> detect(const cv::Mat& image);
};
```

**示例2：Applications层ROS接口一致**

Python版本：
```python
# applications/applications/object_detection/object_detector_node.py
from core.vision.object_detector import ObjectDetector  # 调用core

class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.detector = ObjectDetector(model_path)  # 使用core能力
        self.srv = self.create_service(
            DetectObjects, 'detect_objects', self.callback)
```

C++版本（ROS接口相同，只是调用C++ core）：
```cpp
// applications/src/object_detection/object_detector_node.cpp
#include "core/vision/object_detector.hpp"  // 调用core

class ObjectDetectorNode : public rclcpp::Node {
public:
    ObjectDetectorNode() : Node("object_detector") {
        detector_ = std::make_unique<ObjectDetector>(model_path);  // 使用core能力
        srv_ = this->create_service<msgs::srv::DetectObjects>(
            "detect_objects", ...);  // 服务名相同
    }
private:
    std::unique_ptr<ObjectDetector> detector_;
};
```

**关键优势：**
- ✅ Applications层代码几乎不需要改动
- ✅ 只需要将`from core.vision import ...`改为`#include "core/vision/..."`
- ✅ ROS通信接口完全不变

#### 7.3.2 Launch文件切换

**方式1：通过参数切换（推荐初期）**

```python
# applications/launch/object_detection.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    use_cpp = LaunchConfiguration('use_cpp', default='false')

    return LaunchDescription([
        DeclareLaunchArgument('use_cpp', default_value='false'),

        # Python版本（调用core的Python实现）
        Node(
            package='applications',
            executable='object_detector_node_py',
            name='object_detector',
            parameters=[{'model_path': 'core/models/detector.onnx'}],
            condition=UnlessCondition(use_cpp)
        ),

        # C++版本（调用core的C++实现）
        Node(
            package='applications',
            executable='object_detector_node_cpp',
            name='object_detector',
            parameters=[{'model_path': 'core/models/detector.trt'}],
            condition=IfCondition(use_cpp)
        )
    ])
```

启动：
```bash
# 开发阶段：使用Python
ros2 launch applications object_detection.launch.py use_cpp:=false

# 生产阶段：使用C++
ros2 launch applications object_detection.launch.py use_cpp:=true
```

**方式2：直接切换（后期稳定后）**

```python
# 直接修改launch文件，删除条件判断
Node(
    package='applications',
    executable='object_detector_node_cpp',  # 直接用C++
    name='object_detector',
)
```

#### 7.3.3 Core层Python和C++混用

**通过pybind11实现混用（可选）**

如果applications层用Python，但希望调用C++ core：

```cpp
// core/src/python_bindings.cpp
#include <pybind11/pybind11.h>
#include "core/vision/object_detector.hpp"

namespace py = pybind11;

PYBIND11_MODULE(core_cpp, m) {
    py::class_<ObjectDetector>(m, "ObjectDetector")
        .def(py::init<const std::string&, float>())
        .def("detect", &ObjectDetector::detect);
}
```

Python调用C++ core：
```python
# applications节点可以调用C++实现的core
import core_cpp  # C++编译的Python模块

detector = core_cpp.ObjectDetector("model.trt", 0.5)
objects = detector.detect(image)
```

#### 7.3.4 配置文件共享

**Python和C++都使用YAML配置：**

```yaml
# applications/config/object_detection.yaml
object_detector:
  ros__parameters:
    model_path: "core/models/foreign_object_detector"
    confidence_threshold: 0.5
    nms_threshold: 0.4
```

**Python节点加载：**
```python
# applications/applications/object_detection/object_detector_node.py
self.declare_parameter('model_path', '')
self.declare_parameter('confidence_threshold', 0.5)

model_path = self.get_parameter('model_path').value
threshold = self.get_parameter('confidence_threshold').value

# 传递给core
from core.vision.object_detector import ObjectDetector
self.detector = ObjectDetector(model_path, threshold)
```

**C++节点加载：**
```cpp
// applications/src/object_detection/object_detector_node.cpp
this->declare_parameter("model_path", "");
this->declare_parameter("confidence_threshold", 0.5);

std::string model_path = this->get_parameter("model_path").as_string();
float threshold = this->get_parameter("confidence_threshold").as_double();

// 传递给core
#include "core/vision/object_detector.hpp"
detector_ = std::make_unique<ObjectDetector>(model_path, threshold);
```

**关键点：**
- ✅ 配置文件完全相同
- ✅ 参数加载代码几乎一致
- ✅ 都通过构造函数传递给core层

---

## 8. 配置管理设计

### 8.1 配置文件组织

```
pluck_ws/src/utils/config/              # 全局配置
├── system.yaml                          # 系统级参数
├── workspace_poses.yaml                 # 预定义位姿
├── hand_eye_calibration.yaml            # 手眼标定
└── logging.yaml                         # 日志配置

各package的config/目录：                 # 模块级配置
├── applications/config/
│   ├── object_detection.yaml          # 异物检测业务配置
│   ├── visual_servoing.yaml           # 视觉伺服业务配置
│   ├── tray_localization.yaml         # 托盘定位业务配置
│   └── task_scheduling.yaml           # 任务调度配置
├── hardware/config/
│   └── camera.yaml
└── core/config/
    └── models.yaml                    # 模型配置（可选）
```

### 8.2 预定义位姿配置

```yaml
# utils/config/workspace_poses.yaml
predefined_poses:
  vision_arm:
    tray_scan_start:
      position: [0.5, 0.0, 0.3]      # xyz (米)
      orientation: [0.0, 0.0, 0.0]   # roll, pitch, yaw (弧度)

    region_zones:
      zone_A:
        position: [0.35, 0.10, 0.18]
        orientation: [0.0, 1.57, 0.0]
      zone_B:
        position: [0.35, 0.13, 0.18]
        orientation: [0.0, 1.57, 0.0]
      # ... zone_C ~ zone_Z (20-30个区域)

  gripper_arm:
    home:
      position: [0.0, 0.5, 0.4]
      orientation: [0.0, 0.0, 0.0]
    wash_station:
      position: [-0.2, 0.6, 0.3]
      orientation: [0.0, 0.0, 1.57]

  pressure_arm:
    home:
      position: [0.0, -0.5, 0.4]
      orientation: [0.0, 0.0, 0.0]
    # 各区域压板位姿...
```

### 8.3 手眼标定参数

```yaml
# utils/config/hand_eye_calibration.yaml
hand_eye:
  # 相机光心到机械臂末端法兰的变换
  translation:
    x: 0.05   # 米
    y: 0.02
    z: 0.08
  rotation:
    r: 0.0    # 弧度
    p: 0.0
    y: 1.57

  # 相机内参
  camera_matrix:
    fx: 1500.0
    fy: 1500.0
    cx: 1224.0
    cy: 1024.0

  distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
```

---

## 9. 日志和数据管理设计

### 9.1 日志分级和存储

**日志级别：**
- **DEBUG**：详细调试信息（开发阶段）
- **INFO**：正常运行信息（托盘开始处理、区域完成等）
- **WARN**：警告信息（夹取失败、IBVS不收敛等）
- **ERROR**：错误信息（相机断开、PLC通信中断等）
- **FATAL**：致命错误（系统崩溃）

**日志存储：**
```
pluck_ws/logs/
├── system_YYYYMMDD_HHMMSS.log         # 系统主日志
├── vision_YYYYMMDD_HHMMSS.log         # 视觉模块日志
├── errors/                            # 错误日志单独存储
│   └── error_YYYYMMDD_HHMMSS.log
└── archived/                          # 归档日志（定期清理）
```

**日志配置：**
```yaml
# utils/config/logging.yaml
logging:
  level: INFO                          # 默认级别
  max_file_size: 100MB                 # 单文件最大大小
  max_files: 30                        # 保留文件数
  modules:
    pluck_vision: DEBUG                # 模块级别覆盖
    pluck_hardware: WARN
```

### 9.2 数据库设计（抽象）

**存储内容：**
- 托盘处理记录（开始时间、完成时间、异物总数）
- 区域处理记录（区域ID、异物数量、成功失败统计）
- 异物记录（位置、类型、是否成功夹取）
- 异常记录（时间、类型、处理结果）
- 关键图像（原始检测图、失败案例图）

**数据组织（抽象结构）：**
```
数据库/
├── 托盘记录表
│   ├── 托盘ID
│   ├── 开始时间
│   ├── 完成时间
│   ├── 总异物数
│   └── 成功率
├── 区域记录表
│   ├── 区域ID
│   ├── 所属托盘ID
│   ├── 异物数
│   └── 处理状态
├── 异物记录表
│   ├── 异物ID
│   ├── 位置坐标
│   ├── 类型
│   └── 夹取结果
└── 图像存储
    ├── 原始图像路径
    └── 缩略图（可选）
```

**文件系统存储（图像）：**
```
pluck_ws/data/
├── images/
│   ├── YYYYMMDD/
│   │   ├── tray_001/
│   │   │   ├── zone_A_detection.jpg
│   │   │   ├── zone_A_failed_object.jpg
│   │   │   └── ...
│   │   └── tray_002/
│   └── archived/                     # 定期归档
└── statistics/
    └── daily_report_YYYYMMDD.json
```

---

## 10. 实时性和性能设计

### 10.1 性能目标

| 环节 | 目标延迟 | 说明 |
|------|----------|------|
| 图像采集 | <20ms | 60fps相机 |
| 深度学习推理 | <100ms | TensorRT优化后 |
| IBVS伺服周期 | <50ms (20Hz) | ViSP计算 + PLC通信 |
| 托盘定位 | <500ms | 一次性操作 |
| 单区域处理 | <30s | 包含检测、夹取、清洗 |
| 整托盘处理 | <15min | 20-30个区域 |

### 10.2 实时性保证策略

#### 10.2.1 深度学习推理优化

- **TensorRT优化**：FP16精度，减少推理时间
- **批处理**：单张图像推理（批大小=1）
- **GPU独占**：推理引擎独占GPU，避免其他任务干扰
- **预加载模型**：启动时加载模型到GPU内存

#### 10.2.2 IBVS伺服实时性

- **固定周期**：使用ROS 2 Timer，20-30Hz
- **优先级调度**：Linux实时补丁（可选）
- **减少通信延迟**：使用DDS共享内存（同机节点）

#### 10.2.3 ROS 2 DDS配置

```xml
<!-- utils/config/dds_profile.xml -->
<profiles>
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>SharedMemTransport</transport_id>
            <type>SHM</type>  <!-- 共享内存，降低延迟 -->
        </transport_descriptor>
    </transport_descriptors>
</profiles>
```

### 10.3 性能监控

**关键指标：**
- 推理帧率（FPS）
- IBVS伺服周期实际值
- PLC通信延迟
- 单区域处理时间

**监控工具：**
```python
# utils/utils/performance_monitor.py
class PerformanceMonitor(Node):
    def __init__(self):
        self.create_subscription(Image, '/camera/image', self.image_callback)
        self.create_timer(1.0, self.report_fps)

    def image_callback(self, msg):
        self.frame_count += 1

    def report_fps(self):
        fps = self.frame_count / 1.0
        self.get_logger().info(f'Camera FPS: {fps}')
        self.frame_count = 0
```

---

## 11. 可扩展性设计

### 11.1 模块化设计原则

- **松耦合**：模块间通过ROS 2标准接口通信，降低依赖
- **接口稳定**：消息定义早期确定，避免频繁修改
- **配置驱动**：行为通过配置文件控制，无需修改代码

### 11.2 未来扩展预留

#### 11.2.1 MoveIt集成预留

```python
# motion/motion/motion_planner.py
class MotionPlanner(Node):
    """运动规划接口（预留MoveIt）"""

    def __init__(self):
        self.planning_mode = self.get_parameter('planning_mode').value
        # planning_mode: "simple" 或 "moveit"

    def plan_trajectory(self, target_pose):
        if self.planning_mode == "simple":
            return self.simple_planning(target_pose)
        elif self.planning_mode == "moveit":
            return self.moveit_planning(target_pose)  # 后续实现
```

#### 11.2.2 多模型支持

```yaml
# vision/config/object_detection.yaml
models:
  - name: "hair_detector"
    path: "models/hair_detector.trt"
    classes: ["hair"]
  - name: "spot_detector"
    path: "models/spot_detector.trt"
    classes: ["black_spot", "yellow_spot"]

active_model: "hair_detector"  # 可切换
```

#### 11.2.3 HMI界面接口预留

```python
# utils/utils/hmi_interface.py
class HMIInterface(Node):
    """人机交互接口（后期开发）"""

    def publish_system_status(self):
        """发布系统状态到HMI"""
        # Topic: /pluck/hmi/status
        pass

    def handle_user_command(self):
        """接收用户指令"""
        # Topic: /pluck/hmi/command
        pass
```

---

## 12. 待细化部分

### 12.1 PLC通信接口（留白）

**待确定内容：**
- PROFINET通信库选择（开源库 vs 厂商SDK）
- 通信协议详细规范（数据包格式、握手机制）
- 事件驱动还是轮询模式
- 容错和重连策略

**预留接口：**
```python
# hardware/hardware/plc_interface.py
class PLCInterface(Node):
    def send_motion_command(self, arm_id, target_pose):
        """发送运动指令"""
        raise NotImplementedError("待实现")

    def receive_event(self):
        """接收PLC事件（到位、异常等）"""
        raise NotImplementedError("待实现")
```

### 12.2 其他待定项

- **深度学习模型训练**：数据标注、模型选择、训练流程
- **手眼标定工具**：标定流程、精度验证
- **系统测试策略**：单元测试、集成测试、硬件在环测试
- **部署方案**：Docker容器化、自动启动脚本

---

## 13. 总结

### 13.1 核心设计要点

1. **清晰的分层架构**：业务逻辑、核心功能、硬件接口三层分离
2. **模块化Package组织**：6个ROS 2包，职责明确，低耦合
3. **Python→C++渐进式开发**：初期快速原型，后期性能优化
4. **TensorRT深度学习推理**：高性能GPU加速
5. **ViSP视觉伺服**：成熟IBVS库
6. **配置驱动设计**：YAML配置文件管理所有参数
7. **实时性保证**：TensorRT、固定周期、DDS优化
8. **可扩展性预留**：MoveIt、多模型、HMI接口

### 13.2 下一步工作

基于本系统架构设计，后续需要完成：

1. **接口定义** - 详细定义所有msg/srv/action（参考[Section 4.1](#41-pluck_msgs消息定义包)）
2. **状态机实现** - 基于[Section 5.3](#53-状态管理模块)详细设计
3. **深度学习模型** - 模型训练和TensorRT转换
4. **IBVS详细设计** - 夹爪标记设计、收敛判据
5. **PLC通信协议** - 细化通信接口（当前留白）
6. **开发环境搭建** - Docker容器、依赖安装
7. **测试策略** - 单元测试、集成测试方案

---

**文档状态：** ✅ 系统架构设计已完成
**下一步：** 接口定义文档（interface-definitions.md）

