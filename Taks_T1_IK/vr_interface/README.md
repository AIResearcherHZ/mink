# VR接口模块使用说明

## 概述

本模块提供VR位姿数据的UDP接收、EMA平滑处理和按键事件检测功能，用于MuJoCo机器人IK控制。

## 使用流程（重要）

### 完整操作步骤

```
1. 双击左摇杆     → 设置本机IP地址
2. 点击Connect   → 连接到Python端
3. 双击右摇杆     → 启用追踪（开始发送数据）
4. 双击B按钮      → 软校准（记录VR与机器人位置偏移）
5. 使用侧键       → 控制双手末端位置
6. 双击右摇杆     → 暂停（停止发送数据）
7. 双击A按钮      → 复位（机器人回到初始姿态，可换人操作）
```

### 操作说明

| 操作 | VR手柄 | 键盘 | 功能 |
|------|--------|------|------|
| 设置IP | 双击左摇杆 | - | 设置Python端IP地址 |
| 连接 | Connect按钮 | - | 建立UDP连接 |
| 启用/暂停追踪 | 双击右摇杆 | - | 开始/停止发送数据 |
| 软校准 | 双击B | C | 记录VR与机器人位置偏移 |
| 控制手部 | 侧键 | - | 控制末端执行器位置 |
| 控制夹爪 | 扳机 | - | 控制夹爪开合位置 |
| 复位 | 双击A | Backspace | 机器人回初始姿态，可重新校准 |

### 换人操作流程

复位后会自动清除校准状态，新操作者可直接进行软校准：

```
1. 当前操作者双击A复位
2. 新操作者戴上VR设备
3. 双击右摇杆启用追踪
4. 双击B进行软校准
5. 开始控制
```

## 运行方法

```bash
conda activate gmr
python halfbody_ik_vr.py
```

## API参考

### VRReceiver

```python
VRReceiver(
    ip: str = "0.0.0.0",      # 监听IP
    port: int = 7000,          # UDP端口
    convert_quat: bool = True, # 是否转换四元数(Unity->MuJoCo)
    ema_alpha: float = 0.3,    # EMA平滑系数(0-1, 越小越平滑)
    target_fps: float = 100.0  # 目标帧率
)
```

### VRData

```python
@dataclass
class VRData:
    head: VRPose              # 头部位姿
    left_hand: VRPose         # 左手位姿
    right_hand: VRPose        # 右手位姿
    tracking_enabled: bool    # 追踪是否启用
    timestamp: float          # 时间戳
    button_events: VRButtonEvents  # 按键事件
```

### VRPose

```python
@dataclass
class VRPose:
    position: np.ndarray      # 位置 [x, y, z]
    quaternion: np.ndarray    # 四元数 [w, x, y, z]
    gripper: float            # 夹爪值 (0-100)
```

### VRButtonEvents

```python
@dataclass
class VRButtonEvents:
    left_x: bool   # 左手X按钮双击
    left_y: bool   # 左手Y按钮双击
    right_a: bool  # 右手A按钮双击
    right_b: bool  # 右手B按钮双击
```

## 校准原理

软校准时计算偏移量:
- **左手偏移**: MuJoCo左手位置 - VR左手位置
- **右手偏移**: MuJoCo右手位置 - VR右手位置

控制时:
- 机器人手部位置 = VR手部位置 + 手部偏移

## UDP数据格式

Unity端发送的JSON格式:

```json
{
    "head": {"position": [x, y, z], "quaternion": [w, x, y, z]},
    "leftHand": {"position": [x, y, z], "quaternion": [w, x, y, z], "gripper": 0-100},
    "rightHand": {"position": [x, y, z], "quaternion": [w, x, y, z], "gripper": 0-100},
    "trackingEnabled": true/false,
    "timestamp": 0.0,
    "buttonEvents": {"leftX": false, "leftY": false, "rightA": false, "rightB": false}
}
```

## 注意事项

1. 确保Unity VR端和Python端使用相同的UDP端口(默认7000)
2. 校准前必须先启用追踪（双击右摇杆）
3. 如果机器人抖动，可以降低`ema_alpha`值
4. 复位后校准状态会清除，需要重新校准
5. 按键事件是一次性的，读取后自动清除
