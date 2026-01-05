# Taks_T1 IK控制系统

## 坐标系说明

### MuJoCo世界坐标系
- **X轴（红色）**：向前
- **Y轴（绿色）**：向左
- **Z轴（蓝色）**：向上

### 末端执行器局部坐标系
- 末端执行器的局部坐标系与其link的body frame对齐
- 通过 `mink.move_mocap_to_frame(model, data, mocap, link, "body")` 实现对齐
- IK目标通过 `mink.SE3.from_mocap_id(data, mid)` 在世界坐标系下设置

### XML中的坐标轴可视化
每个mocap目标都包含三个圆柱体来可视化坐标轴：
- **第一个圆柱体** (`pos="0.04 0 0"`, `rgba="1 0 0 1"`)：X轴-红色-向前
- **第二个圆柱体** (`pos="0 0.04 0"`, `rgba="0 1 0 1"`)：Y轴-绿色-向左
- **第三个圆柱体** (`pos="0 0 0.04"`, `rgba="0 0 1 1"`)：Z轴-蓝色-向上

## 项目结构

```
Taks_T1_IK/
├── halfbody_ik.py              # 半身IK控制（双手+颈部）
├── fullbody_ik.py              # 全身IK控制（双手+双脚+颈部）
├── fullbody_ik_independent.py  # 全身独立IK API
├── assets/
│   ├── Semi_Taks_T1/           # 半身模型资源
│   │   ├── scene_Semi_Taks_T1.xml  # 半身场景（4个mocap目标）
│   │   └── Semi_Taks_T1.xml        # 半身机器人模型
│   └── Taks_T1/                # 全身模型资源
│       ├── scene_Taks_T1.xml       # 全身场景（5个mocap目标）
│       └── Taks_T1.xml             # 全身机器人模型
└── README.md
```

## 功能说明

### 半身IK控制 (`halfbody_ik.py`)
- **末端执行器**：左手、右手、颈部yaw、颈部pitch
- **控制关节组**：左臂、右臂、腰部、颈部
- **初始姿态**：keyframe "home"
- **Mocap目标数量**：4个

### 全身IK控制 (`fullbody_ik.py`)
- **末端执行器**：左手、右手、左脚、右脚、颈部
- **控制关节组**：左臂、右臂、左腿、右腿、腰部、颈部
- **初始姿态**：keyframe "stand"
- **Mocap目标数量**：5个

## 核心特性

### 1. 独立肢体控制
- 每个肢体独立响应其对应的mocap目标移动
- 非活动肢体保持当前姿态不动
- 通过动态调整posture cost实现选择性控制

### 2. 零空间约束
- 非活动肢体的shoulder/hip关节冻结
- 使用 `DofFreezingTask` 实现零空间保持
- 防止非活动肢体漂移

### 3. 重力补偿
- 使用 `data.qfrc_bias` 计算重力补偿力矩
- 通过 `data.qfrc_applied` 应用前馈控制
- 减少IK求解负担，提高控制精度

### 4. 碰撞避免
- 使用 `CollisionAvoidanceLimit` 避免自碰撞
- 预定义碰撞对列表 `COLLISION_PAIRS`
- 安全距离：0.01m，影响距离：0.1m

## 使用方法

### 运行半身IK
```bash
cd /home/xhz/mink/Taks_T1_IK
python halfbody_ik.py
```

### 运行全身IK
```bash
cd /home/xhz/mink/Taks_T1_IK
python fullbody_ik.py
```

### 交互控制
1. 启动程序后，MuJoCo viewer会打开
2. 使用鼠标拖动彩色球体（mocap目标）
3. 对应的肢体会自动跟随目标移动
4. 其他肢体保持静止

## 技术细节

### IK求解器
- **求解器**：DAQP（Dual Active-set QP solver）
- **阻尼系数**：0.1
- **控制频率**：200Hz
- **时间步长**：0.005s

### 任务优先级
1. **Base/Pelvis固定任务**：cost=5.0（位置+姿态）
2. **末端执行器任务**：cost=5.0（位置+姿态）
3. **姿态保持任务**：cost=1e-2（活动肢体）/ 1e4（非活动肢体）

### 关节限制
- 使用 `ConfigurationLimit` 强制关节限位
- 自动从URDF/XML中读取关节限位

## 依赖项
- `mujoco` >= 3.0
- `mink` - IK求解库
- `numpy`
- `loop_rate_limiters` - 频率控制

## 注意事项
1. 确保mocap目标的初始位置与keyframe姿态一致
2. 移动mocap目标时避免过大的位置跳变
3. 观察终端输出的重力补偿力矩，确认系统正常工作
4. 如遇到碰撞，调整 `COLLISION_PAIRS` 或安全距离参数
