#!/usr/bin/env python3
"""
电机扫描程序 (带部位标注与彩色输出)
扫描 0x01 ~ 0x16 (1~22) ID 的电机

sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on

"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from libs.drivers.DM_CAN_FD import Motor, MotorControlFD, DM_Motor_Type

# ANSI 颜色定义
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def get_body_part(motor_id):
    """根据 ID 获取对应的身体部位"""
    if 0x01 <= motor_id <= 0x07:
        return f"{Color.GREEN}右手 (Arm_R){Color.END}"
    elif motor_id == 0x08:
        return f"{Color.GREEN}{Color.BOLD}右手夹爪 (Gripper_R){Color.END}"
    elif 0x09 <= motor_id <= 0x0F:
        return f"{Color.CYAN}左手 (Arm_L){Color.END}"
    elif motor_id == 0x10:
        return f"{Color.CYAN}{Color.BOLD}左手夹爪 (Gripper_L){Color.END}"
    elif 0x11 <= motor_id <= 0x13:
        return f"{Color.YELLOW}腰部 (Waist){Color.END}"
    elif 0x14 <= motor_id <= 0x16:
        return f"{Color.PURPLE}脖子 (Neck){Color.END}"
    return f"{Color.RED}未知部位{Color.END}"

def scan_motors(interface='can1'):
    print(f"\n🚀 {Color.BOLD}开始扫描 CAN 接口: {interface}{Color.END}")
    try:
        motor_ctrl = MotorControlFD(can_interface=interface)
    except Exception as e:
        print(f"❌ {Color.RED}无法初始化 CAN 接口: {e}{Color.END}")
        return

    found_motors = []
    
    # 先清空接收缓冲区
    time.sleep(0.2)
    motor_ctrl.recv()
    
    # 扫描配置 - 电机多时需要更长间隔
    SCAN_INTERVAL = 0.02  # 每个电机扫描间隔20ms
    RESPONSE_WAIT = 0.05  # 等待响应50ms
    MAX_RETRIES = 5  # 最大重试次数
    
    motors = {}
    
    # 逐个扫描 ID 0x01 到 0x16 (1 到 22)
    for slave_id in range(0x01, 0x17):
        part_name = get_body_part(slave_id)
        print(f"🔍 {Color.DARKCYAN}正在探测 ID: {hex(slave_id).ljust(4)} {part_name.ljust(30)}...{Color.END}", end='\r')
        
        # 为当前电机创建对象
        master_id = slave_id + 0x80
        test_motor = Motor(DM_Motor_Type.DM4340, SlaveID=slave_id, MasterID=master_id)
        test_motor.state_q = float('nan')
        motor_ctrl.addMotor(test_motor)
        motors[slave_id] = test_motor
        
        # 先尝试enable
        try:
            motor_ctrl.enable(test_motor)
            motor_ctrl.controlMIT(test_motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            time.sleep(SCAN_INTERVAL)
        except Exception:
            pass
        
        detected = False
        
        for retry in range(MAX_RETRIES):
            # 清空缓冲区
            motor_ctrl.recv()
            
            # 重置状态
            test_motor.state_q = float('nan')
            
            # 发送刷新状态指令
            motor_ctrl.refresh_motor_status(test_motor)
            
            # 等待响应
            time.sleep(RESPONSE_WAIT)
            motor_ctrl.recv()
            
            # 检查是否收到有效响应
            if not math.isnan(test_motor.state_q):
                detected = True
                break
            
            # 重试前额外等待
            time.sleep(SCAN_INTERVAL)
        
        # 检查是否检测到电机
        if not math.isnan(test_motor.state_q):
            print(f"✅ {Color.BOLD}{Color.GREEN}发现电机! [ID: {hex(slave_id).ljust(4)}] 部位: {part_name.ljust(25)} Pos: {test_motor.getPosition():.3f} rad{Color.END}")
            found_motors.append(slave_id)
        else:
            print(f"   {Color.RED}未响应    [ID: {hex(slave_id).ljust(4)}] 部位: {part_name}{Color.END}                    ")

    print("\n" + "="*50)
    if found_motors:
        print(f"🎉 {Color.BOLD}{Color.GREEN}扫描完成! 共发现 {len(found_motors)} 个电机{Color.END}")
        for mid in found_motors:
            print(f"  - {Color.BLUE}ID: {hex(mid).ljust(4)}{Color.END} -> {get_body_part(mid)}")
    else:
        print(f"❌ {Color.RED}{Color.BOLD}未发现任何在线电机。请检查接线和电源。{Color.END}")
    print("="*50 + "\n")

    motor_ctrl.close()

if __name__ == "__main__":
    try:
        scan_motors('can1')
    except KeyboardInterrupt:
        print(f"\n🛑 {Color.YELLOW}用户停止扫描{Color.END}")
    except Exception as e:
        print(f"\n⚠️ {Color.RED}发生错误: {e}{Color.END}")