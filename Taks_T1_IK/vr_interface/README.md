# VR接口模块使用说明

## 概述

本模块提供VR位姿数据的UDP接收、EMA平滑处理和按键事件检测功能，用于MuJoCo机器人IK控制。

## 功能特性

- **UDP数据接收**: 接收Unity VR发送的头部和双手位姿数据
- **EMA数据平滑**: 指数移动平均平滑，减少抖动
- **帧率稳定**: 网络波动时保持数据输出稳定
- **按键双击事件**: 检测ABXY四个按键的双击事件
- **坐标转换**: Unity到MuJoCo四元数自动转换

## 快速开始

### 基本使用

```python
from vr_interface import VRReceiver

# 创建接收器
vr = VRReceiver(port=7000, ema_alpha=0.3)
vr.start()

# 获取数据
data = vr.data
print(f"左手位置: {data.left_hand.position}")
print(f"右手位置: {data.right_hand.position}")
print(f"头部位置: {data.head.position}")
print(f"追踪状态: {data.tracking_enabled}")

# 检测按键事件
if data.button_events.right_b:
    print("右手B按钮双击!")
if data.button_events.right_a:
    print("右手A按钮双击!")

vr.stop()
```

### 上下文管理器

```python
from vr_interface import VRReceiver

with VRReceiver() as vr:
    while True:
        data = vr.data
        # 处理数据...
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

## VR IK控制使用方法

### 运行halfbody_ik_vr.py

```bash
conda activate gmr
python halfbody_ik_vr.py
```

### 操作说明

| 操作 | 键盘 | VR手柄 | 功能 |
|------|------|--------|------|
| 校准 | C | B双击 | 校准VR位置偏移 |
| 复位 | Backspace | A双击 | 机器人回到初始姿态 |

### 使用流程

1. **启动程序**: 运行`halfbody_ik_vr.py`
2. **启用追踪**: 在Unity VR端启用追踪(trackingEnabled=true)
3. **校准偏移**: 
   - 将机器人手臂调整到合适位置
   - 按键盘`C`或VR手柄`B双击`进行校准
   - 校准会记录VR位置与机器人位置的偏移
4. **开始控制**: 校准后移动VR手柄即可控制机器人
5. **复位**: 按`Backspace`或`A双击`让机器人回到初始姿态

### 校准原理

校准时计算三个偏移量:
- **左手偏移**: MuJoCo左手位置 - VR左手位置
- **右手偏移**: MuJoCo右手位置 - VR右手位置  
- **头到腰偏移**: MuJoCo腰部位置 - VR头部位置

控制时:
- 机器人手部位置 = VR手部位置 + 手部偏移
- 机器人腰部位置 = VR头部位置 + 头到腰偏移

## UDP数据格式

Unity端发送的JSON格式:

```json
{
    "head": {
        "position": [x, y, z],
        "quaternion": [w, x, y, z]
    },
    "leftHand": {
        "position": [x, y, z],
        "quaternion": [w, x, y, z],
        "gripper": 0-100
    },
    "rightHand": {
        "position": [x, y, z],
        "quaternion": [w, x, y, z],
        "gripper": 0-100
    },
    "trackingEnabled": true/false,
    "timestamp": 0.0,
    "buttonEvents": {
        "leftX": false,
        "leftY": false,
        "rightA": false,
        "rightB": false
    }
}
```

## EMA平滑说明

EMA(指数移动平均)公式:
```
smoothed = alpha * new_value + (1 - alpha) * smoothed
```

- `alpha = 0.3`: 默认值，平衡响应速度和平滑度
- `alpha = 0.1`: 更平滑，但响应较慢
- `alpha = 0.5`: 响应更快，但平滑度降低

四元数使用球面线性插值(slerp)进行平滑。

## 注意事项

1. 确保Unity VR端和Python端使用相同的UDP端口(默认7000)
2. 校准前确保VR追踪已启用
3. 如果机器人抖动，可以降低`ema_alpha`值
4. 按键事件是一次性的，读取后自动清除
