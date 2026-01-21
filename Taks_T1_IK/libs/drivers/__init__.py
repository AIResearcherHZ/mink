"""电机控制驱动模块

提供DM电机和IMU的控制接口：
- DM_Motor: 串口版DM电机控制
- DM_CAN_FD: CAN FD版DM电机控制
- DM_IMU: IMU传感器接口
- ankle_kinematics: 踝关节运动学解算

使用说明：
- DM电机状态（位置/速度/力矩）为缓存值，只在发送控制指令或显式调用
  `refresh_motor_status(motor)` 后更新
- 所有模块支持相对导入，可通过 `from libs.drivers import ...` 使用
"""

__all__ = [
    "DM_Motor",
    "DM_CAN_FD", 
    "DM_IMU",
    "ankle_kinematics",
]