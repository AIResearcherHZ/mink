#!/usr/bin/env python3
"""使用DM_Motor库(串口版)设置左右手腕6个电机的减速比为10"""

from time import sleep
import serial
from DM_Motor import MotorControl, Motor, DM_Motor_Type, DM_variable

# 手腕电机ID: 右手腕5,6,7  左手腕13,14,15
WRIST_MOTOR_IDS = [5, 6, 7, 13, 14, 15]
GEAR_RATIO = 10.0

# 串口配置
SERIAL_PORT_RIGHT = '/dev/ttyUSB0'  # 右臂串口
SERIAL_PORT_LEFT = '/dev/ttyUSB1'   # 左臂串口
BAUDRATE = 921600

def main():
    # 初始化串口
    print("初始化串口...")
    ser_right = serial.Serial(port=SERIAL_PORT_RIGHT, baudrate=BAUDRATE, timeout=0.5)
    ser_left = serial.Serial(port=SERIAL_PORT_LEFT, baudrate=BAUDRATE, timeout=0.5)
    
    mc_right = MotorControl(ser_right)  # 右臂
    mc_left = MotorControl(ser_left)    # 左臂
    
    # 创建电机对象
    motors = {}
    for mid in [5, 6, 7]:  # 右手腕
        motors[mid] = Motor(DM_Motor_Type.DM4310, mid, mid)
        mc_right.addMotor(motors[mid])
    for mid in [13, 14, 15]:  # 左手腕
        motors[mid] = Motor(DM_Motor_Type.DM4310, mid, mid)
        mc_left.addMotor(motors[mid])
    
    print(f"设置减速比为 {GEAR_RATIO}...")
    
    # 设置右手腕电机减速比
    for mid in [5, 6, 7]:
        motor = motors[mid]
        print(f"  电机 {mid}: ", end="")
        result = mc_right.change_motor_param(motor, DM_variable.Gr, GEAR_RATIO)
        if result:
            print("设置成功")
        else:
            print("设置失败")
    
    # 设置左手腕电机减速比
    for mid in [13, 14, 15]:
        motor = motors[mid]
        print(f"  电机 {mid}: ", end="")
        result = mc_left.change_motor_param(motor, DM_variable.Gr, GEAR_RATIO)
        if result:
            print("设置成功")
        else:
            print("设置失败")
    
    # 保存参数到flash
    print("保存参数到flash...")
    for mid in [5, 6, 7]:
        mc_right.save_motor_param(motors[mid])
        print(f"  电机 {mid}: 已保存")
        sleep(0.1)
    for mid in [13, 14, 15]:
        mc_left.save_motor_param(motors[mid])
        print(f"  电机 {mid}: 已保存")
        sleep(0.1)
    
    # 关闭串口
    ser_right.close()
    ser_left.close()
    print("完成!")

if __name__ == "__main__":
    main()
