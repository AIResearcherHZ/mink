#!/usr/bin/env python3
"""使用DM_CAN_FD库设置左右手腕6个电机的减速比为10"""

from time import sleep
from DM_CAN_FD import MotorControlFD, Motor, DM_Motor_Type, DM_variable

# 手腕电机ID: 右手腕5,6,7  左手腕13,14,15
WRIST_MOTOR_IDS = [5, 6, 7, 13, 14, 15]
GEAR_RATIO = 10.0

def main():
    # 初始化CAN FD控制器 (右臂用can0, 左臂用can1)
    print("初始化CAN FD接口...")
    mc_right = MotorControlFD(can_interface='can0')  # 右臂
    mc_left = MotorControlFD(can_interface='can1')   # 左臂
    
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
    
    # 关闭CAN接口
    mc_right.close()
    mc_left.close()
    print("完成!")

if __name__ == "__main__":
    main()
