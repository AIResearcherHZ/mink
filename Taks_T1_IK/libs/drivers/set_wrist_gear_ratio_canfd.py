#!/usr/bin/env python3
"""
读取手腕电机的减速比参数

注意：经测试发现，DM4310电机的Gr（减速比）参数是只读的，
无法通过软件修改。减速比由电机的机械结构决定。
如果需要不同的减速比，需要更换不同型号的电机或机械减速器。
"""

from time import sleep
from DM_CAN_FD import MotorControlFD, Motor, DM_Motor_Type, DM_variable

# 手腕电机ID: 右手腕5,6,7  左手腕13,14,15
WRIST_MOTOR_IDS = [5, 6, 7, 13, 14, 15]

def main():
    print("初始化CAN FD接口...")
    mc = MotorControlFD(can_interface='can1')

    print("\n读取手腕电机减速比参数:")
    print("-" * 40)
    
    motors = {}
    for mid in WRIST_MOTOR_IDS:
        motors[mid] = Motor(DM_Motor_Type.DM4310, mid, mid + 0x80)
        mc.addMotor(motors[mid])
        gr = mc.read_motor_param(motors[mid], DM_variable.Gr)
        if gr is not None:
            print(f"  电机 {mid:2d}: Gr = {gr}")
        else:
            print(f"  电机 {mid:2d}: 无响应（可能未连接）")

    print("-" * 40)
    print("\n说明：Gr参数是只读的，由电机机械结构决定，无法软件修改。")
    
    mc.close()
    print("\n完成!")

if __name__ == "__main__":
    main()