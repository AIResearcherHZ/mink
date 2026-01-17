#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
踝关节电机测试 - DM_CAN_FD版本
测试踝关节运动学解算与CAN FD电机驱动的集成
"""

import math
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from DM_CAN_FD import Motor, MotorControlFD, DM_Motor_Type, Control_Type
from ankle_kinematics import ankle_ik, ankle_fk, ankle_vel_to_motor_vel, ankle_tau_to_motor_tau


# 踝关节电机配置 (根据实际硬件修改)
ANKLE_MOTOR1_ID = 27      # 上电机ID
ANKLE_MOTOR1_SLAVE = 0xA7 # 上电机从机ID
ANKLE_MOTOR2_ID = 28      # 下电机ID
ANKLE_MOTOR2_SLAVE = 0xA8 # 下电机从机ID
CAN_INTERFACE = 'can0'    # CAN接口
LEFT_LEG = True           # True=左脚, False=右脚


def test_ankle_control():
    """踝关节控制测试"""
    print("=" * 60)
    print("踝关节电机测试 - DM_CAN_FD")
    print("=" * 60)
    
    # 创建电机对象
    motor1 = Motor(DM_Motor_Type.DM4340, ANKLE_MOTOR1_ID, ANKLE_MOTOR1_SLAVE)
    motor2 = Motor(DM_Motor_Type.DM4340, ANKLE_MOTOR2_ID, ANKLE_MOTOR2_SLAVE)
    
    # 创建控制器
    mc = MotorControlFD(can_interface=CAN_INTERFACE)
    mc.addMotor(motor1)
    mc.addMotor(motor2)
    
    print(f"✓ 电机初始化完成")
    print(f"  Motor1: ID={ANKLE_MOTOR1_ID}")
    print(f"  Motor2: ID={ANKLE_MOTOR2_ID}")
    
    # 切换MIT模式
    if mc.switchControlMode(motor1, Control_Type.MIT):
        print("✓ Motor1 切换MIT模式成功")
    if mc.switchControlMode(motor2, Control_Type.MIT):
        print("✓ Motor2 切换MIT模式成功")
    
    # 使能电机
    mc.enable(motor1)
    mc.enable(motor2)
    print("✓ 电机已使能")
    
    try:
        # 读取当前电机位置
        theta1_cur = motor1.getPosition()
        theta2_cur = motor2.getPosition()
        print(f"\n当前电机角度: theta1={np.rad2deg(theta1_cur):.2f}° theta2={np.rad2deg(theta2_cur):.2f}°")
        
        # 正运动学: 电机角度 -> 踝关节角度
        pitch_cur, roll_cur = ankle_fk(theta1_cur, theta2_cur, LEFT_LEG)
        print(f"当前踝关节角度: pitch={np.rad2deg(pitch_cur):.2f}° roll={np.rad2deg(roll_cur):.2f}°")
        
        # 目标踝关节角度
        target_pitch = np.deg2rad(5.0)
        target_roll = np.deg2rad(0.0)
        print(f"\n目标踝关节角度: pitch={np.rad2deg(target_pitch):.2f}° roll={np.rad2deg(target_roll):.2f}°")
        
        # 逆运动学: 踝关节角度 -> 电机角度
        theta1_target, theta2_target = ankle_ik(target_pitch, target_roll, LEFT_LEG)
        print(f"目标电机角度: theta1={np.rad2deg(theta1_target):.2f}° theta2={np.rad2deg(theta2_target):.2f}°")
        
        # 控制参数
        kp = 20.0
        kd = 2.0
        
        print(f"\n开始控制 (kp={kp}, kd={kd})...")
        print("按 Ctrl+C 停止")
        
        start_time = time.time()
        duration = 5.0  # 运行5秒
        
        while time.time() - start_time < duration:
            # 读取当前状态
            theta1 = motor1.getPosition()
            theta2 = motor2.getPosition()
            vel1 = motor1.getVelocity()
            vel2 = motor2.getVelocity()
            
            # 正运动学
            pitch, roll = ankle_fk(theta1, theta2, LEFT_LEG)
            
            # 发送MIT控制命令
            mc.controlMIT(motor1, kp, kd, theta1_target, 0, 0)
            mc.controlMIT(motor2, kp, kd, theta2_target, 0, 0)
            
            # 打印状态
            print(f"\r踝关节: pitch={np.rad2deg(pitch):6.2f}° roll={np.rad2deg(roll):6.2f}° | "
                  f"电机: θ1={np.rad2deg(theta1):6.2f}° θ2={np.rad2deg(theta2):6.2f}°", end="")
            
            time.sleep(0.002)  # 500Hz
        
        print("\n\n✓ 控制完成")
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    finally:
        # 失能电机
        mc.disable(motor1)
        mc.disable(motor2)
        mc.close()
        print("✓ 电机已失能并关闭")


def test_kinematics_only():
    """仅测试运动学解算 (无需硬件)"""
    print("=" * 60)
    print("踝关节运动学解算测试 (无硬件)")
    print("=" * 60)
    
    # 测试多个姿态
    test_poses = [
        (0, 0),
        (5, 0),
        (0, 5),
        (10, 5),
        (-5, 3),
    ]
    
    for pitch_deg, roll_deg in test_poses:
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        
        # IK
        theta1, theta2 = ankle_ik(pitch, roll, LEFT_LEG)
        
        # FK验证
        pitch_fk, roll_fk = ankle_fk(theta1, theta2, LEFT_LEG)
        
        err = max(abs(pitch - pitch_fk), abs(roll - roll_fk))
        status = "✓" if err < 1e-4 else "✗"
        
        print(f"{status} pitch={pitch_deg:5.1f}° roll={roll_deg:5.1f}° -> "
              f"θ1={np.rad2deg(theta1):7.3f}° θ2={np.rad2deg(theta2):7.3f}° -> "
              f"pitch={np.rad2deg(pitch_fk):7.3f}° roll={np.rad2deg(roll_fk):7.3f}°")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--no-hw':
        test_kinematics_only()
    else:
        print("使用 --no-hw 参数可仅测试运动学解算 (无需硬件)")
        print()
        test_ankle_control()
