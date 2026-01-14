#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
踝关节运动学解算模块
- FK (Forward Kinematics): 电机角度 -> 脚踝 pitch/roll
- IK (Inverse Kinematics): 脚踝 pitch/roll -> 电机角度
- 雅可比矩阵: 速度/力矩转换
"""

import math
import numpy as np

# ============ 几何参数 ============
D1 = 0.035
D2 = 0.039625
L1 = 0.039625
H1 = 0.19
H2 = 0.13


def ankle_ik(pitch: float, roll: float) -> tuple:
    """
    逆运动学：脚踝角度 -> 电机角度
    
    重要：参数顺序与参考文件不同！
    - 参考文件 ankle_motor_to_real.py: ankle_inverse_kinematics(roll, pitch)
    - 本函数: ankle_ik(pitch, roll)
    
    内部会自动处理参数映射，确保与机械结构一致。
    
    Args:
        pitch: 脚踝俯仰角 (rad)
        roll: 脚踝横滚角 (rad)
    
    Returns:
        (theta_upper, theta_lower): 电机角度 (rad)
    """
    # 参数映射：公式中 theta_p 对应 roll，theta_r 对应 pitch
    theta_p = roll
    theta_r = pitch
    
    d1, d2, L1, h1, h2 = D1, D2, 0.039625, H1, H2
    a1, a2 = h1, h2
    
    sin_p, cos_p = math.sin(theta_p), math.cos(theta_p)
    sin_r, cos_r = math.sin(theta_r), math.cos(theta_r)
    
    A_L = 2 * d2 * L1 * sin_p - 2 * L1 * d1 * sin_r - 2 * h2 * L1
    A_R = -2 * d2 * L1 * sin_p - 2 * L1 * d1 * sin_r + 2 * h1 * L1
    
    B_L = 2 * d2 * L1 * cos_p
    B_R = 2 * d2 * L1 * cos_p
    
    C_L = (2 * d1 * d1 + h2 * h2 - a2 * a2 + L1 * L1 + d2 * d2 - 
           2 * d1 * d1 * cos_r + 2 * d1 * h2 * sin_r - 
           2 * d2 * h2 * sin_p - 2 * d1 * d2 * sin_p * sin_r)
           
    C_R = (2 * d1 * d1 + h1 * h1 - a1 * a1 + L1 * L1 + d2 * d2 - 
           2 * d1 * d1 * cos_r - 2 * d1 * h1 * sin_r - 
           2 * d2 * h1 * sin_p + 2 * d1 * d2 * sin_p * sin_r)

    Len_L = A_L * A_L - C_L * C_L + B_L * B_L
    Len_R = A_R * A_R - C_R * C_R + B_R * B_R

    if Len_L > 0 and Len_R > 0:
        theta_lower = 2 * math.atan((-A_L - math.sqrt(Len_L)) / (B_L + C_L))
        theta_upper = 2 * math.atan((-A_R + math.sqrt(Len_R)) / (B_R + C_R))
        theta_lower = -theta_lower
        return theta_upper, theta_lower
    
    return 0.0, 0.0


def ankle_fk(theta_upper: float, theta_lower: float) -> tuple:
    """
    正运动学：电机角度 -> 脚踝角度
    使用牛顿迭代法，高精度快速收敛
    """
    # 优化的初始猜测（线性回归系数）
    pitch = 0.5504 * theta_upper - 0.5501 * theta_lower
    roll = 0.4862 * theta_upper + 0.4866 * theta_lower
    
    # 牛顿迭代（通常2-3次收敛到高精度）
    for _ in range(4):
        calc_upper, calc_lower = ankle_ik(pitch, roll)
        err_u = calc_upper - theta_upper
        err_l = calc_lower - theta_lower
        
        if err_u * err_u + err_l * err_l < 1e-20:
            break
        
        # 前向差分计算雅可比
        up_p, lo_p = ankle_ik(pitch + 1e-7, roll)
        up_r, lo_r = ankle_ik(pitch, roll + 1e-7)
        
        J00 = (up_p - calc_upper) * 1e7
        J01 = (up_r - calc_upper) * 1e7
        J10 = (lo_p - calc_lower) * 1e7
        J11 = (lo_r - calc_lower) * 1e7
        
        det = J00 * J11 - J01 * J10
        if abs(det) < 1e-12:
            break
        
        inv_det = 1.0 / det
        pitch -= inv_det * (J11 * err_u - J01 * err_l)
        roll -= inv_det * (-J10 * err_u + J00 * err_l)
    
    return pitch, roll


def compute_jacobian_fast(pitch: float, roll: float) -> tuple:
    """
    计算雅可比矩阵 J = d(theta_motor) / d(ankle)
    返回四个元素，避免numpy开销
    
    Returns:
        (J00, J01, J10, J11): 雅可比矩阵元素
           J00 = d(upper)/d(pitch), J01 = d(upper)/d(roll)
           J10 = d(lower)/d(pitch), J11 = d(lower)/d(roll)
    """
    eps = 1e-6
    inv_2eps = 0.5 / eps
    
    # 对 pitch 求偏导
    up_pp, lo_pp = ankle_ik(pitch + eps, roll)
    up_pm, lo_pm = ankle_ik(pitch - eps, roll)
    J00 = (up_pp - up_pm) * inv_2eps
    J10 = (lo_pp - lo_pm) * inv_2eps
    
    # 对 roll 求偏导
    up_rp, lo_rp = ankle_ik(pitch, roll + eps)
    up_rm, lo_rm = ankle_ik(pitch, roll - eps)
    J01 = (up_rp - up_rm) * inv_2eps
    J11 = (lo_rp - lo_rm) * inv_2eps
    
    return J00, J01, J10, J11


def compute_jacobian(pitch: float, roll: float, eps: float = 1e-6) -> np.ndarray:
    """
    计算雅可比矩阵 J = d(theta_motor) / d(ankle)
    
    Returns:
        J: 2x2 矩阵
           J[0,0] = d(upper)/d(pitch), J[0,1] = d(upper)/d(roll)
           J[1,0] = d(lower)/d(pitch), J[1,1] = d(lower)/d(roll)
    """
    J00, J01, J10, J11 = compute_jacobian_fast(pitch, roll)
    return np.array([[J00, J01], [J10, J11]])


def motor_vel_to_ankle_vel(pitch: float, roll: float, 
                           vel_upper: float, vel_lower: float) -> tuple:
    """
    电机速度 -> 脚踝角速度
    ankle_dot = J^(-1) * motor_dot
    """
    J00, J01, J10, J11 = compute_jacobian_fast(pitch, roll)
    
    # 2x2 矩阵求逆: det = J00*J11 - J01*J10
    det = J00 * J11 - J01 * J10
    if abs(det) < 1e-12:
        return 0.0, 0.0
    
    inv_det = 1.0 / det
    # J^(-1) = [[J11, -J01], [-J10, J00]] / det
    pitch_vel = inv_det * (J11 * vel_upper - J01 * vel_lower)
    roll_vel = inv_det * (-J10 * vel_upper + J00 * vel_lower)
    
    return pitch_vel, roll_vel


def motor_tau_to_ankle_tau(pitch: float, roll: float,
                           tau_upper: float, tau_lower: float) -> tuple:
    """
    电机扭矩 -> 脚踝扭矩
    tau_ankle = J^T * tau_motor
    """
    J00, J01, J10, J11 = compute_jacobian_fast(pitch, roll)
    # J^T = [[J00, J10], [J01, J11]]
    tau_pitch = J00 * tau_upper + J10 * tau_lower
    tau_roll = J01 * tau_upper + J11 * tau_lower
    return tau_pitch, tau_roll


def ankle_vel_to_motor_vel(pitch: float, roll: float,
                           pitch_vel: float, roll_vel: float) -> tuple:
    """
    脚踝角速度 -> 电机速度
    motor_dot = J * ankle_dot
    """
    J00, J01, J10, J11 = compute_jacobian_fast(pitch, roll)
    # J * [pitch_vel, roll_vel]^T
    vel_upper = J00 * pitch_vel + J01 * roll_vel
    vel_lower = J10 * pitch_vel + J11 * roll_vel
    return vel_upper, vel_lower


def ankle_tau_to_motor_tau(pitch: float, roll: float,
                           tau_pitch: float, tau_roll: float) -> tuple:
    """
    脚踝扭矩 -> 电机扭矩
    tau_motor = (J^T)^(-1) * tau_ankle
    """
    J00, J01, J10, J11 = compute_jacobian_fast(pitch, roll)
    
    # (J^T)^(-1): J^T = [[J00, J10], [J01, J11]]
    # det(J^T) = J00*J11 - J10*J01 = det(J)
    det = J00 * J11 - J01 * J10
    if abs(det) < 1e-12:
        return 0.0, 0.0
    
    inv_det = 1.0 / det
    # (J^T)^(-1) = [[J11, -J10], [-J01, J00]] / det
    tau_upper = inv_det * (J11 * tau_pitch - J10 * tau_roll)
    tau_lower = inv_det * (-J01 * tau_pitch + J00 * tau_roll)
    
    return tau_upper, tau_lower