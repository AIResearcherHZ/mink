#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夹爪控制模块 - 基于DM_Motor.py库驱动
支持左右手夹爪的开合控制
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from libs.drivers.DM_Motor import Motor as DM_Motor, MotorControl as DM_MotorControl, DM_Motor_Type, Control_Type
from time import sleep

# 夹爪位置定义 (电机弧度位置)
GRIPPER_OPEN = 0.0       # 打开位置
GRIPPER_CLOSE = 1.05      # 闭合位置

# 左右手夹爪镜像配置 (预留，目前默认都一样)
# 如果左手夹爪需要镜像，将LEFT_GRIPPER_DIRECTION设为-1
RIGHT_GRIPPER_DIRECTION = 1  # 右手夹爪方向系数
LEFT_GRIPPER_DIRECTION = 1   # 左手夹爪方向系数 (如需镜像改为-1)

# 夹爪MIT控制参数
GRIPPER_KP = 1.5
GRIPPER_KD = 0.1
GRIPPER_DQ = 0.0    # 速度
GRIPPER_TAU = 0.0   # 前馈力矩

# 夹爪关节ID
RIGHT_GRIPPER_ID = 8   # 右手夹爪
LEFT_GRIPPER_ID = 16   # 左手夹爪


class GripperController:
    """夹爪控制器"""
    
    def __init__(self, motor_ctrl: DM_MotorControl, motor: DM_Motor, is_left: bool = False):
        """
        初始化夹爪控制器
        :param motor_ctrl: 电机控制器对象
        :param motor: 电机对象
        :param is_left: 是否为左手夹爪
        """
        self.motor_ctrl = motor_ctrl
        self.motor = motor
        self.is_left = is_left
        self.direction = LEFT_GRIPPER_DIRECTION if is_left else RIGHT_GRIPPER_DIRECTION
        self.is_closed = False
        self.current_position = GRIPPER_OPEN
        
    def _apply_direction(self, position: float) -> float:
        """应用方向系数"""
        return position * self.direction
    
    def SetOC(self, close: bool):
        """
        设置夹爪开合状态
        :param close: True=闭合(1), False=打开(0)
        """
        self.is_closed = close
        self.current_position = GRIPPER_CLOSE if close else GRIPPER_OPEN
        actual_pos = self._apply_direction(self.current_position)
        self.motor_ctrl.controlMIT(
            self.motor, GRIPPER_KP, GRIPPER_KD,
            actual_pos, GRIPPER_DQ, GRIPPER_TAU
        )
    
    def SetPosition(self, percent: float):
        """
        设置夹爪位置百分比
        :param percent: 0-100, 0=完全打开, 100=完全闭合
        """
        percent = max(0, min(100, percent))
        # 线性映射: 0% -> GRIPPER_OPEN, 100% -> GRIPPER_CLOSE
        self.current_position = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (percent / 100.0)
        self.is_closed = percent >= 50
        actual_pos = self._apply_direction(self.current_position)
        self.motor_ctrl.controlMIT(
            self.motor, GRIPPER_KP, GRIPPER_KD,
            actual_pos, GRIPPER_DQ, GRIPPER_TAU
        )
    
    def controlMIT(self, percent: float, kp: float = None, kd: float = None):
        """
        MIT控制模式，支持自定义kp/kd
        :param percent: 0-100位置百分比, 0=完全打开, 100=完全闭合
        :param kp: 可选，自定义kp值
        :param kd: 可选，自定义kd值
        """
        percent = max(0, min(100, percent))
        position = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (percent / 100.0)
        actual_pos = self._apply_direction(position)
        use_kp = kp if kp is not None else GRIPPER_KP
        use_kd = kd if kd is not None else GRIPPER_KD
        self.motor_ctrl.controlMIT(
            self.motor, use_kp, use_kd,
            actual_pos, GRIPPER_DQ, GRIPPER_TAU
        )
    
    def open(self):
        """打开夹爪"""
        self.SetOC(False)
    
    def close(self):
        """闭合夹爪"""
        self.SetOC(True)
    
    def toggle(self):
        """切换夹爪状态"""
        self.SetOC(not self.is_closed)
    
    def get_status(self) -> str:
        """获取状态字符串"""
        return "闭合" if self.is_closed else "打开"
    
    def get_position_percent(self) -> float:
        """获取当前位置百分比"""
        return (self.current_position - GRIPPER_OPEN) / (GRIPPER_CLOSE - GRIPPER_OPEN) * 100.0


def create_gripper(motor_ctrl: DM_MotorControl, gripper_id: int, is_left: bool = False) -> GripperController:
    """
    创建夹爪控制器
    :param motor_ctrl: 电机控制器
    :param gripper_id: 夹爪电机ID (8=右手, 16=左手)
    :param is_left: 是否为左手夹爪
    :return: GripperController对象
    """
    motor = DM_Motor(DM_Motor_Type.DM3507, gripper_id, gripper_id + 0x80)
    motor_ctrl.addMotor(motor)
    motor_ctrl.switchControlMode(motor, Control_Type.MIT)
    motor_ctrl.enable(motor)
    return GripperController(motor_ctrl, motor, is_left)