#!/usr/bin/env python3
"""
夹爪控制测试代码 (SSH友好版)
- F键：切换夹爪开/关状态
- O键：打开夹爪
- C键：闭合夹爪
- Q键：退出程序

MIT控制模式: kp=1.5, kd=0.1, 速度=0, 前馈力矩=0

夹爪打开进行零位校准
"""

import sys
import tty
import termios
import threading
from time import sleep
from DM_CAN_FD import MotorControlFD, Motor, DM_Motor_Type

# 夹爪位置定义
GRIPPER_OPEN = 0.0      # 打开位置（电机弧度位置）
GRIPPER_CLOSE = 1.05    # 闭合位置（电机弧度位置）

# MIT控制参数
KP = 1.5
KD = 0.1
DQ = 0.0    # 速度
TAU = 0.0   # 前馈力矩


class GripperController:
    """夹爪控制器"""
    
    def __init__(self, motor_ctrl, motor):
        self.motor_ctrl = motor_ctrl
        self.motor = motor
        self.is_closed = False
        self.current_position = GRIPPER_OPEN
        self.running = True
        
    def toggle(self):
        """切换夹爪状态"""
        self.is_closed = not self.is_closed
        self.current_position = GRIPPER_CLOSE if self.is_closed else GRIPPER_OPEN
        
    def open(self):
        """打开夹爪"""
        self.is_closed = False
        self.current_position = GRIPPER_OPEN
        
    def close(self):
        """闭合夹爪"""
        self.is_closed = True
        self.current_position = GRIPPER_CLOSE
    
    def get_status(self):
        """获取状态字符串"""
        return "闭合" if self.is_closed else "打开"
    
    def control_loop(self):
        """控制循环，持续发送命令"""
        while self.running:
            self.motor_ctrl.controlMIT(
                self.motor, KP, KD, 
                self.current_position, DQ, TAU
            )
            sleep(0.01)  # 100Hz


def getch():
    """获取单个字符输入（非阻塞风格）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_status(gripper):
    """打印当前状态"""
    status = gripper.get_status()
    pos = gripper.current_position
    bar_len = 20
    fill = int((pos - GRIPPER_CLOSE) / (GRIPPER_OPEN - GRIPPER_CLOSE) * bar_len)
    bar = "█" * fill + "░" * (bar_len - fill)
    print(f"\r夹爪状态: [{bar}] {status} (位置: {pos:.2f})  ", end='', flush=True)


def main():
    print("=" * 50)
    print("夹爪控制测试程序 (SSH友好版)")
    print("=" * 50)
    print("操作说明:")
    print("  F键 -> 切换夹爪开/关")
    print("  O键 -> 打开夹爪")
    print("  C键 -> 闭合夹爪")
    print("  Q键 -> 退出程序")
    print("=" * 50)
    
    # 初始化电机控制器
    print("\n正在初始化电机...")
    motor_ctrl = MotorControlFD(can_interface='can0')
    motor = Motor(DM_Motor_Type.DM3507, SlaveID=0x10, MasterID=0x90)
    motor_ctrl.addMotor(motor)
    
    sleep(0.5)
    
    # 使能电机
    print("使能电机...")
    motor_ctrl.enable(motor)
    sleep(0.5)
    
    # 创建夹爪控制器
    gripper = GripperController(motor_ctrl, motor)
    
    # 启动控制循环线程
    control_thread = threading.Thread(target=gripper.control_loop, daemon=True)
    control_thread.start()
    
    print("\n开始监听键盘输入...\n")
    print_status(gripper)
    
    try:
        while True:
            key = getch()
            
            if key.lower() == 'q' or ord(key) == 3:  # Q或Ctrl+C
                print("\n\n退出程序...")
                break
            elif key.lower() == 'f':
                gripper.toggle()
                print_status(gripper)
            elif key.lower() == 'o':
                gripper.open()
                print_status(gripper)
            elif key.lower() == 'c':
                gripper.close()
                print_status(gripper)
                
    except KeyboardInterrupt:
        print("\n\n程序被中断...")
    finally:
        # 停止控制循环
        gripper.running = False
        sleep(0.1)
        
        # 关闭夹爪到打开位置
        print("恢复夹爪到打开位置...")
        motor_ctrl.controlMIT(motor, KP, KD, 0, DQ, TAU)
        sleep(0.5)
        
        # 失能电机
        print("失能电机...")
        motor_ctrl.disable(motor)
        
        # 关闭CAN总线
        motor_ctrl.close()
        print("程序结束")


if __name__ == "__main__":
    main()
