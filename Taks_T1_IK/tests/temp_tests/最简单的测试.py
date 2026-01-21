#!/usr/bin/env python3
"""极简版电机位置读取测试"""

import time
from DM_CAN_FD import MotorControlFD, Motor, DM_Motor_Type

# 初始化
motor_ctrl = MotorControlFD(can_interface='can0')
motor = Motor(DM_Motor_Type.DM4340, SlaveID=0xb, MasterID=0x8b)
motor_ctrl.addMotor(motor)

# 使能电机
motor_ctrl.enable(motor)

# 持续读取位置
try:
    while True:
        motor_ctrl.refresh_motor_status(motor)
        print(f"位置: {motor.getPosition():.4f} rad, 速度: {motor.getVelocity():.4f} rad/s, 力矩: {motor.getTorque():.4f} Nm")
        time.sleep(0.0001)
except KeyboardInterrupt:
    motor_ctrl.disable(motor)
    motor_ctrl.close()
