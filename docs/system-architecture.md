# 燕窝挑毛系统技术架构设计文档

> **版本**: v1.0 详细技术版  
> **日期**: 2025-10-22  
> **受众**: 开发团队  
> **技术栈**: ROS 2 Humble + MoveIt 2 + ViSP + 深度学习

---

## 文档说明

本文档是燕窝挑毛自动化系统的**详细技术架构设计**，面向开发团队。包含：
- 系统分层架构
- ROS 2节点设计
- 接口定义（话题/服务/动作）
- 核心模块实现方案
- 技术选型与待确定事项

---

## 目录

1. [系统整体架构](#1-系统整体架构)
2. [ROS 2节点架构](#2-ros-2节点架构)
3. [核心模块详细设计](#3-核心模块详细设计)
4. [接口定义](#4-接口定义)
5. [核心工作流程](#5-核心工作流程)
6. [技术选型](#6-技术选型)
7. [开发计划](#7-开发计划)

---

## 1. 系统整体架构

### 1.1 系统分层

系统采用**六层架构**，从底层硬件到顶层交互逐层解耦：

| 层级 | 名称 | 职责 | 核心技术 | 优先级 |
|------|------|------|---------|--------|
| **L6** | 交互层 | HMI界面、监控、手动干预、数据可视化 | Qt/Web + ROS Bridge | P4 |
| **L5** | 数据层 | 日志记录、数据库存储、图像归档、统计分析 | 数据库(待定) + 文件系统 | P3 |
| **L4** | 执行层 | IBVS伺服、夹爪控制、压板控制、状态机 | ViSP + ROS 2 Control | P2 |
| **L3** | 规划层 | 任务调度、运动规划、碰撞检测、区域规划 | MoveIt 2 + 自研调度器 | P1 |
| **L2** | 感知层 | 图像采集、深度学习推理、边缘检测、特征提取 | OpenCV + 深度学习 + TensorRT | P1 |
| **L1** | 硬件接口层 | ProfiNet通信、相机驱动、机械臂驱动 | ProfiNet SDK + 相机SDK | P0 |

### 1.2 层间交互原则

- **单向依赖**: 上层依赖下层，下层不依赖上层
- **接口抽象**: 层间通过ROS 2话题/服务/动作通信
- **松耦合**: 各层可独立开发、测试、替换

### 1.3 数据流向

```
硬件设备(机械臂、相机)
    ↓
硬件接口层(驱动封装)
    ↓
感知层(图像处理、目标识别)
    ↓
规划层(任务调度、运动规划)
    ↓
执行层(精确控制、状态管理)
    ↓
数据层(记录、存储)
    ↓
交互层(展示、交互)
```

---

## 2. ROS 2节点架构

### 2.1 节点总览

系统包含**15个核心ROS 2节点**，按层级划分：

#### L1 硬件接口层(4个节点)

| 节点名称 | 功能 | 发布话题 | 订阅话题 | 提供服务 |
|---------|------|---------|---------|---------|
| press_arm_driver | 压板臂驱动 | /press_arm/joint_states | /press_arm/joint_command | - |
| vision_arm_driver | 视觉臂驱动 | /vision_arm/joint_states | /vision_arm/joint_command | - |
| gripper_arm_driver | 夹爪臂驱动 | /gripper_arm/joint_states | /gripper_arm/joint_command | - |
| camera_driver | 相机驱动 | /camera/image_raw, /camera/camera_info | - | /camera/trigger |

#### L2 感知层(4个节点)

| 节点名称 | 功能 | 发布话题 | 订阅话题 | 提供服务 |
|---------|------|---------|---------|---------|
| image_processor | 图像预处理 | /image/processed | /camera/image_raw | - |
| dl_inference | 深度学习推理 | /detections | /image/processed | /detect_objects |
| tray_localizer | 托盘定位 | /tray_pose | /camera/image_raw | /localize_tray |
| gripper_detector | 夹爪标志检测 | /gripper_marker_pose | /camera/image_raw | - |

#### L3 规划层(3个节点)

| 节点名称 | 功能 | 发布话题 | 订阅话题 | 提供服务/动作 |
|---------|------|---------|---------|---------|
| task_coordinator | 任务协调器 | /task/status, /task/current_region | 三臂status | /task/start, /task/stop |
| region_planner | 区域规划器 | /region_list | /tray_pose | /plan_regions |
| moveit_planner | MoveIt规划器 | - | - | action: /execute_trajectory |

#### L4 执行层(3个节点)

| 节点名称 | 功能 | 发布话题 | 订阅话题 | 提供服务/动作 |
|---------|------|---------|---------|---------|
| press_arm_controller | 压板臂控制 | /press_arm/status | /task/press_command | /press_arm/move_to |
| vision_arm_controller | 视觉臂控制 | /vision_arm/status | /task/vision_command | /vision_arm/move_to |
| gripper_arm_controller | 夹爪臂控制+IBVS | /gripper_arm/status | /detections, /gripper_marker_pose | /gripper_arm/pick, action: /ibvs_servo |

#### L5 数据层(1个节点)

| 节点名称 | 功能 | 订阅话题 | 提供服务 |
|---------|------|---------|---------|
| data_recorder | 数据记录器 | /task/status, /detections, /camera/image_raw | /data/query |

### 2.2 节点通信拓扑

核心通信路径：

```
1. 托盘定位流程:
   camera_driver → tray_localizer → task_coordinator → region_planner

2. 异物检测流程:
   camera_driver → image_processor → dl_inference → task_coordinator

3. 夹取执行流程:
   task_coordinator → gripper_arm_controller → ibvs_servo(动作)
   camera_driver → gripper_detector → gripper_arm_controller(IBVS反馈)

4. 运动规划流程:
   task_coordinator → moveit_planner → xxx_arm_driver

5. 数据记录流程:
   task_coordinator/detections/camera → data_recorder
```

### 2.3 关键节点状态机

#### task_coordinator 状态机

| 状态 | 说明 | 转移条件 |
|------|------|---------|
| IDLE | 空闲，等待启动 | 接收启动信号 → LOCALIZING |
| LOCALIZING | 托盘定位中 | 定位成功 → PLANNING; 失败 → ERROR |
| PLANNING | 区域规划中 | 规划完成 → PROCESSING |
| PROCESSING | 处理区域中 | 全部完成 → COMPLETED; 异常 → ERROR |
| COMPLETED | 托盘处理完成 | 发信号，返回 → IDLE |
| ERROR | 异常状态 | 人工干预/重试 → IDLE |

#### gripper_arm_controller 状态机

| 状态 | 说明 | 转移条件 |
|------|------|---------|
| IDLE | 空闲 | 接收夹取指令 → MOVING |
| MOVING | 粗定位移动中 | 到达附近 → SERVOING |
| SERVOING | IBVS伺服中 | 对准成功 → GRIPPING; 超时/失败 → RETRY |
| GRIPPING | 夹取动作 | 成功 → CLEANING; 失败 → RETRY |
| CLEANING | 清洗/处理 | 完成 → IDLE |
| RETRY | 重试 | <3次 → MOVING; ≥3次 → FAILED |
| FAILED | 失败 | 记录失败，返回 → IDLE |

---

## 3. 核心模块详细设计

### 3.1 硬件接口层

#### 3.1.1 ProfiNet驱动节点

**功能**: 封装ProfiNet通信，提供统一的机械臂控制接口

**实现方案**(待定):

- **方案A**: 直接用C++封装ProfiNet SDK
  - 优点: 性能好，控制精度高
  - 缺点: 需要深入理解ProfiNet协议

- **方案B**: 使用OPC UA作为中间层
  - 优点: 标准化，易于集成
  - 缺点: 增加一层延迟

- **方案C**: 使用机械臂厂商提供的ROS驱动
  - 优点: 开箱即用
  - 缺点: 依赖厂商支持

**接口设计**:
- 订阅: /xxx_arm/joint_command (sensor_msgs/JointState)
- 发布: /xxx_arm/joint_states (sensor_msgs/JointState)
- 频率: 100Hz(控制) + 100Hz(反馈)

#### 3.1.2 相机驱动节点

**功能**: 采集图像，发布ROS图像话题

**实现方案**:
- 使用相机厂商SDK(GigE Vision / USB3 Vision)
- 封装为ROS 2 image_transport节点

**接口设计**:
- 发布: /camera/image_raw (sensor_msgs/Image) - 30Hz
- 发布: /camera/camera_info (sensor_msgs/CameraInfo) - 手眼标定参数
- 服务: /camera/trigger (std_srvs/Trigger) - 单次拍照触发

---

### 3.2 感知层

#### 3.2.1 深度学习推理节点(dl_inference)

**功能**: 检测燕窝表面异物(毛发、黑点、黄点)

**实现方案**:
- 输入: 预处理后的图像(3x3cm区域，约1000x1000像素)
- 模型: 抽象接口，支持YOLO/U-Net等(待实验选择)
- 加速: TensorRT GPU推理
- 输出: 异物列表(位置、类别、置信度)

**性能要求**:
- 推理时间: <10ms/帧(TensorRT加速)
- 漏检率: <1%
- 误检率: <5%(可接受)

#### 3.2.2 托盘定位节点(tray_localizer)

**功能**: 检测托盘边缘，计算托盘在世界坐标系中的位姿

**实现方案**:
1. 边缘检测: Canny算法
2. 直线拟合: Hough变换
3. 位姿计算: 根据已知托盘尺寸，反解位姿
4. 坐标变换: 发布TF变换(world → tray)

**精度要求**: ±1mm(平移)，±0.5°(旋转)

#### 3.2.3 夹爪标志检测节点(gripper_detector)

**功能**: 实时检测夹爪上的醒目标志，反推夹爪头位置

**为什么需要**:
- 夹爪头太小，直接识别计算量大
- 在夹爪靠后位置贴醒目标志(如红色圆点)，易于检测
- 通过标志位置 + 已知偏移量，反推夹爪头位置

**实现方案**:
- 颜色空间分割(HSV阈值)
- 形态学滤波
- 连通域分析
- 中心点计算

---

### 3.3 规划层

#### 3.3.1 任务协调器(task_coordinator)

**功能**: 三臂协同调度的核心，管理整个工作流程

**职责**:
1. 接收人工启动信号
2. 协调三臂的状态和任务序列
3. 监控各节点状态，处理异常
4. 记录处理进度

**实现架构**: 分布式协同
- 各臂有独立的控制节点
- 任务协调器通过话题/服务协调
- 不直接控制机械臂，而是发指令给各臂控制节点

#### 3.3.2 区域规划器(region_planner)

**功能**: 根据托盘位姿，划分20-30个3x3cm区域

**算法**:
```
1. 获取托盘位姿(TF: world → tray)
2. 在托盘坐标系中，以3x3cm网格划分
3. 考虑5mm重叠，调整网格间距为2.5cm
4. 将各区域中心点转换到世界坐标系
5. 按固定顺序排列(从左到右，从上到下)
6. 发布区域列表
```

#### 3.3.3 MoveIt规划器(moveit_planner)

**功能**: 机械臂粗定位时的运动规划和碰撞检测

**使用场景**:
- ✅ 大范围移动(如夹爪从工作区到水槽)
- ✅ 需要避障的运动
- ❌ IBVS伺服时不使用(微小移动，无碰撞风险)

---

### 3.4 执行层

#### 3.4.1 IBVS视觉伺服(gripper_arm_controller内集成)

**功能**: 精确对准异物，实现±0.1mm精度

**为什么需要IBVS**:
- 手眼标定误差(±1-2mm)无法满足夹取精度
- 机械臂重复定位误差累积
- IBVS在图像空间闭环，直接补偿误差

**算法原理**:
```
目标: 让图像中异物中心与夹爪中心重合

控制律:
  ΔV = -λ * L^+ * (s - s*)

其中:
  s   = 当前图像特征(异物中心像素坐标)
  s*  = 目标图像特征(图像中心)
  L^+ = 图像雅可比矩阵伪逆
  λ   = 增益系数
  ΔV  = 末端速度增量
```

**实现方案**:
- 使用ViSP库(提供IBVS实现)
- 控制频率: 30-60Hz
- 特征: 异物中心点(2D点特征)

**工作流程**:
```
1. 粗定位: MoveIt规划移动到异物附近(±1-2mm)
2. 启动IBVS:
   loop:
     2.1 相机拍照
     2.2 检测异物中心像素坐标
     2.3 计算图像误差(与图像中心的距离)
     2.4 ViSP计算末端速度
     2.5 发送速度指令给机械臂
     2.6 if 误差 < 0.1mm: break
3. 夹取动作
```

---

## 4. 接口定义

### 4.1 ROS 2话题(Topics)

#### 图像相关

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|---------|------|------|
| /camera/image_raw | sensor_msgs/Image | 30-60Hz | 原始RGB图像 |
| /camera/camera_info | sensor_msgs/CameraInfo | 1Hz | 相机标定参数 |
| /image/processed | sensor_msgs/Image | 30Hz | 预处理后图像 |

#### 感知结果

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|---------|------|------|
| /detections | vision_msgs/Detection2DArray | 按需 | 异物检测结果列表 |
| /tray_pose | geometry_msgs/PoseStamped | 1Hz | 托盘位姿 |
| /gripper_marker_pose | geometry_msgs/PointStamped | 30Hz | 夹爪标志位置 |

#### 机械臂状态

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|---------|------|------|
| /press_arm/joint_states | sensor_msgs/JointState | 100Hz | 压板臂关节状态 |
| /vision_arm/joint_states | sensor_msgs/JointState | 100Hz | 视觉臂关节状态 |
| /gripper_arm/joint_states | sensor_msgs/JointState | 100Hz | 夹爪臂关节状态 |
| /press_arm/status | custom_msgs/ArmStatus | 10Hz | 压板臂任务状态 |
| /vision_arm/status | custom_msgs/ArmStatus | 10Hz | 视觉臂任务状态 |
| /gripper_arm/status | custom_msgs/ArmStatus | 10Hz | 夹爪臂任务状态 |

#### 任务协调

| 话题名 | 消息类型 | 频率 | 说明 |
|--------|---------|------|------|
| /task/status | custom_msgs/TaskStatus | 10Hz | 任务状态 |
| /task/current_region | std_msgs/Int32 | 1Hz | 当前处理区域ID |
| /region_list | custom_msgs/RegionArray | 按需 | 区域列表 |

### 4.2 ROS 2服务(Services)

| 服务名 | 类型 | 功能 |
|--------|------|------|
| /camera/trigger | std_srvs/Trigger | 触发单次拍照 |
| /localize_tray | std_srvs/Trigger | 托盘定位 |
| /plan_regions | std_srvs/Trigger | 区域规划 |
| /detect_objects | custom_srvs/DetectObjects | 异物检测 |
| /press_arm/move_to | custom_srvs/MoveTo | 移动压板臂到指定位置 |
| /vision_arm/move_to | custom_srvs/MoveTo | 移动视觉臂到指定位置 |
| /gripper_arm/pick | custom_srvs/PickObject | 夹取指定位置异物 |
| /task/start | std_srvs/Trigger | 启动任务 |
| /task/stop | std_srvs/Trigger | 停止任务 |
| /data/query | custom_srvs/QueryData | 查询历史数据 |

### 4.3 ROS 2动作(Actions)

| 动作名 | 类型 | 功能 | 反馈 |
|--------|------|------|------|
| /execute_trajectory | moveit_msgs/ExecuteTrajectory | 执行MoveIt轨迹 | 执行进度 |
| /ibvs_servo | custom_actions/IBVSServo | IBVS伺服到目标 | 当前误差 |
| /pick_and_clean | custom_actions/PickAndClean | 夹取+清洗组合 | 当前阶段 |

### 4.4 自定义消息类型

#### ArmStatus.msg

```
uint8 IDLE = 0
uint8 MOVING = 1
uint8 WORKING = 2
uint8 ERROR = 3

uint8 status
string message
geometry_msgs/Pose current_pose
```

#### TaskStatus.msg

```
uint8 IDLE = 0
uint8 LOCALIZING = 1
uint8 PLANNING = 2
uint8 PROCESSING = 3
uint8 COMPLETED = 4
uint8 ERROR = 5

uint8 status
string message
int32 total_regions
int32 completed_regions
```

#### Region.msg

```
int32 id
geometry_msgs/Pose center
bool processed
int32 defect_count
```

---

## 5. 核心工作流程

### 5.1 系统启动流程

```
1. 启动所有ROS 2节点
2. 加载配置参数(机械臂IP、相机参数等)
3. 加载手眼标定参数(从YAML文件)
4. 初始化MoveIt(加载URDF、SRDF、Planning Scene)
5. 三臂回到Home位置
6. 系统进入IDLE状态，等待启动信号
```

### 5.2 单托盘完整流程

```
阶段1: 托盘定位(5-10秒)
------------------------
1. task_coordinator 调用服务 /localize_tray
2. tray_localizer 控制视觉臂移动到托盘上方
3. 拍照，边缘检测，计算托盘位姿
4. 发布TF: world → tray
5. 返回成功

阶段2: 区域规划(<1秒)
------------------------
1. task_coordinator 调用服务 /plan_regions
2. region_planner 根据托盘位姿，划分20-30个区域
3. 发布 /region_list
4. 返回成功

阶段3: 循环处理各区域(主循环)
------------------------
for region in region_list:

  步骤3.1: 压板压住(2秒)
  --------------------
  1. task_coordinator 发送指令给 press_arm_controller
  2. press_arm_controller 调用MoveIt规划路径
  3. 移动到区域上方，下压到固定高度
  4. 发布状态: PRESSED

  步骤3.2: 视觉检测(2-3秒)
  --------------------
  1. task_coordinator 发送指令给 vision_arm_controller
  2. vision_arm_controller 移动到区域上方(15-20cm)
  3. 触发拍照
  4. task_coordinator 调用 /detect_objects
  5. dl_inference 推理，返回异物列表(<10个)

  步骤3.3: 循环夹取(内循环)
  --------------------
  for obj in detections:

    3.3.1 粗定位(1-2秒)
    ------------------
    1. gripper_arm_controller 调用MoveIt规划路径
    2. 移动到异物附近(±1-2mm)

    3.3.2 IBVS精定位(3-5秒)
    ------------------
    1. gripper_arm_controller 启动IBVS动作
    2. 循环(30-60Hz):
       - gripper_detector 发布夹爪标志位置
       - 检测异物位置
       - IBVS计算速度指令
       - 发送给机械臂驱动
       - 检查误差是否 < 0.1mm
    3. 收敛，进入夹取

    3.3.3 夹取动作(1秒)
    ------------------
    1. 夹爪闭合
    2. 检测夹取力
    3. if 成功: 提升，前往清洗
       else: 重试(最多3次)

    3.3.4 清洗/处理(1-2秒)
    ------------------
    1. 移动到水槽
    2. 清洗(或吸走)
    3. 返回工作区
    4. 其他臂等待

  步骤3.4: 质量验证(2秒)
  --------------------
  1. vision_arm_controller 重新拍照
  2. task_coordinator 调用 /detect_objects
  3. if 无异物检测: region.completed = true
     else: 继续步骤3.3

阶段4: 完成(<1秒)
------------------------
1. task_coordinator 记录数据
2. 发布完成信号 TRAY_COMPLETED
3. 返回IDLE状态
4. 等待工人取走托盘，放入新托盘
```

### 5.3 异常处理流程

| 异常类型 | 检测方式 | 处理策略 |
|---------|---------|---------|
| 托盘定位失败 | 超时/边缘检测失败 | 重试3次，失败后报警等待人工干预 |
| 深度学习推理超时 | 超时检测 | 跳过该区域，记录日志 |
| IBVS不收敛 | 超时/最大迭代次数 | 重试3次，失败后记录该异物位置，继续下一个 |
| 夹取失败 | 力传感器/视觉检测 | 重试3次，失败后记录数据库 |
| 机械臂通信断开 | 心跳检测 | 急停，报警 |
| 碰撞检测 | MoveIt碰撞检测 | 停止运动，报警 |

---

## 6. 技术选型

### 6.1 已确定的技术栈

| 模块 | 技术 | 版本 | 说明 |
|------|------|------|------|
| 操作系统 | Ubuntu | 22.04 LTS | ROS 2 Humble官方支持 |
| 机器人框架 | ROS 2 | Humble | 工业级稳定版本 |
| 运动规划 | MoveIt 2 | Humble | 碰撞检测、路径规划 |
| 视觉伺服 | ViSP | 3.5+ | IBVS算法库 |
| 图像处理 | OpenCV | 4.x | 边缘检测、预处理 |
| 深度学习推理 | TensorRT | 8.x | NVIDIA GPU加速 |
| 坐标变换 | TF2 | - | ROS 2坐标树 |
| 编程语言 | C++/Python | 17/3.10 | C++主力，Python辅助 |
| 构建工具 | colcon | - | ROS 2标准构建工具 |
| 版本控制 | Git | - | 代码管理 |

### 6.2 待确定的技术方案

| 模块 | 备选方案 | 决策依据 | 优先级 |
|------|---------|----------|--------|
| ProfiNet通信 | A.C++封装SDK  B.OPC UA中间层  C.厂商ROS驱动 | 机械臂厂商技术支持情况 | P0 |
| 深度学习模型 | A.YOLOv8  B.U-Net  C.混合方案 | 实验精度对比 | P1 |
| 数据库 | A.SQLite  B.PostgreSQL  C.MongoDB | 数据量、查询复杂度 | P3 |
| HMI技术 | A.Qt  B.Web(React+ROS Bridge) | 用户习惯、部署环境 | P4 |
| 手眼标定工具 | A.easy_handeye2  B.ViSP标定 | 标定精度、易用性 | P1 |

---

## 7. 开发计划

### 7.1 阶段划分(总计13-19周)

| 阶段 | 时间 | 优先级 | 目标 |
|------|------|--------|------|
| 阶段1 | 2-3周 | P0 | 基础验证: ProfiNet通信、相机驱动、手眼标定 |
| 阶段2 | 4-6周 | P1 | 核心功能: 深度学习模型、MoveIt配置、IBVS伺服 |
| 阶段3 | 3-4周 | P2 | 系统集成: 三臂协调、完整流程、异常处理 |
| 阶段4 | 2-3周 | P3 | 优化测试: 性能优化、稳定性测试 |
| 阶段5 | 2-3周 | P4 | 交互界面: HMI开发(可选) |

### 7.2 开发任务清单

#### 阶段1: 基础验证(P0)
- [ ] ProfiNet通信开发
- [ ] 相机驱动开发
- [ ] 手眼标定
- [ ] 托盘定位实验

#### 阶段2: 核心功能(P1)
- [ ] 数据采集与标注
- [ ] 深度学习模型训练
- [ ] 模型部署(TensorRT)
- [ ] MoveIt配置
- [ ] IBVS伺服实现
- [ ] 单区域集成测试

#### 阶段3: 系统集成(P2)
- [ ] 任务协调器开发
- [ ] 区域规划器开发
- [ ] 完整流程集成
- [ ] 异常处理实现
- [ ] 数据管理实现

#### 阶段4: 优化测试(P3)
- [ ] 性能分析与优化
- [ ] 稳定性测试(8小时运行)
- [ ] 文档编写

#### 阶段5: 交互界面(P4)
- [ ] HMI设计与开发
- [ ] 用户培训

---

## 附录

### A. 代码仓库结构

```
pluck_ws/
├── src/
│   ├── pluck_bringup/          # 启动文件
│   ├── pluck_drivers/          # 硬件驱动
│   ├── pluck_perception/       # 感知层
│   ├── pluck_planning/         # 规划层
│   ├── pluck_control/          # 执行层
│   ├── pluck_data/             # 数据层
│   ├── pluck_msgs/             # 自定义消息
│   └── pluck_hmi/              # 交互层(后期)
├── models/                     # 深度学习模型
├── calibration/                # 标定文件
└── docs/                       # 文档
```

### B. 术语表

| 术语 | 说明 |
|------|------|
| IBVS | Image-Based Visual Servoing，基于图像的视觉伺服 |
| MoveIt | ROS的运动规划框架 |
| ViSP | Visual Servoing Platform，视觉伺服平台 |
| TensorRT | NVIDIA的深度学习推理加速引擎 |
| ProfiNet | 工业以太网通信协议 |
| TF2 | Transform 2，ROS 2的坐标变换库 |
| URDF | Unified Robot Description Format，机器人描述格式 |
| 手眼标定 | 计算相机与机械臂末端的坐标变换关系 |

---

**文档结束**

如有疑问或需要补充，请联系开发团队。
