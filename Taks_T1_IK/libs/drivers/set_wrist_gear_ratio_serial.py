#!/usr/bin/env python3
"""
读取手腕电机的减速比参数（串口版）

注意：经测试发现，DM4310电机的Gr（减速比）参数是只读的，无法通过软件修改。
减速比由电机的机械结构决定。如果需要不同的减速比，需要更换不同型号的电机或机械减速器。
"""

from time import sleep
import serial
from DM_Motor import MotorControl, Motor, DM_Motor_Type, DM_variable

# 手腕电机ID: 右手腕5,6,7  左手腕13,14,15
WRIST_MOTOR_IDS = [5, 6, 7, 13, 14, 15]

# 串口配置
SERIAL_PORT_RIGHT = '/dev/ttyUSB0'  # 右臂串口
SERIAL_PORT_LEFT = '/dev/ttyUSB1'   # 左臂串口
BAUDRATE = 921600

def main():
    print("初始化串口...")
    ser_right = serial.Serial(port=SERIAL_PORT_RIGHT, baudrate=BAUDRATE, timeout=0.5)
    ser_left = serial.Serial(port=SERIAL_PORT_LEFT, baudrate=BAUDRATE, timeout=0.5)

    mc_right = MotorControl(ser_right)
    mc_left = MotorControl(ser_left)

    motors = {}
    for mid in [5, 6, 7]:  # 右手腕
        motors[mid] = Motor(DM_Motor_Type.DM4310, mid, mid)
        mc_right.addMotor(motors[mid])
    for mid in [13, 14, 15]:  # 左手腕
        motors[mid] = Motor(DM_Motor_Type.DM4310, mid, mid)
        mc_left.addMotor(motors[mid])

    print("\n读取手腕电机减速比参数:")
    print("-" * 40)

    for mid in [5, 6, 7]:
        gr = mc_right.read_motor_param(motors[mid], DM_variable.Gr)
        if gr is not None:
            print(f"  电机 {mid:2d}: Gr = {gr}")
        else:
            print(f"  电机 {mid:2d}: 无响应（可能未连接）")

    for mid in [13, 14, 15]:
        gr = mc_left.read_motor_param(motors[mid], DM_variable.Gr)
        if gr is not None:
            print(f"  电机 {mid:2d}: Gr = {gr}")
        else:
            print(f"  电机 {mid:2d}: 无响应（可能未连接）")

    print("-" * 40)
    print("\n说明：Gr参数是只读的，由电机机械结构决定，无法软件修改。")

    ser_right.close()
    ser_left.close()
    print("\n完成!")

if __name__ == "__main__":
    main()
