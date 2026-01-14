"""电机控制模块

提供统一的电机控制接口，包括：
- DM_Motor: DM电机控制类
- Feetech_Motor: Feetech舵机控制类
- UnifiedMotorControl: 统一电机控制接口
- USB2CAN: USB转CAN通信模块

使用约定与差异：
- DM电机状态（位置/速度/力矩）为缓存值，只在发送控制指令或显式调用
  `refresh_motor_status(motor)` 后更新。调用 `getPosition/getVelocity/getTorque`
  不会主动轮询设备。
- Feetech电机的 `getPosition/getVelocity/getTorque` 为即时查询，会直接从控制器
  读取当前值；也可通过统一接口的 `refresh_motor_status(motor)` 主动拉取并更新缓存。
"""

__all__ = [
    "DM_Motor",
    "Feetech_Motor",
    "UnifiedMotorControl",
    "USB2CAN",
]