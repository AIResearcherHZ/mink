#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
踝关节运动学解算模块 (异向摇臂并联机构)
基于 close_chain_mapping.cpp / verify_kinematics.py 移植
- IK (Inverse Kinematics): 脚踝 pitch/roll -> 电机角度
- FK (Forward Kinematics): 电机角度 -> 脚踝 pitch/roll
- 雅可比矩阵: 速度/力矩转换
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

# ============ 几何参数 (单位: mm) ============
L_BAR = 35.0        # 短连杆(摇臂)长度
L_ROD = np.array([190.0, 130.0])  # 两根作动杆长度 [上/长杆, 下/短杆]
L_SPACING = 20.0    # y向偏置(左脚+, 右脚-)

# 异向摇臂: 两个摇臂初始角度相反
LONG_LINK_ANGLE_0 = 0.0              # 长杆摇臂初始角 (rad)
SHORT_LINK_ANGLE_0 = np.pi           # 短杆摇臂初始角 (rad), 异向

# 基座点A的z坐标(电机轴高度)
A1_Z = 190.0  # 上电机轴高度
A2_Z = 130.0  # 下电机轴高度

# 平台点C的x坐标(足端铰点相对踝中心的前后偏移)
C1_X = -35.0  # 长杆对应的C点x偏移
C2_X = 35.0   # 短杆对应的C点x偏移


def _inverse_kinematics_full(q_roll: float, q_pitch: float, left_leg: bool = True) -> Dict:
    """
    完整逆运动学: 踝关节姿态 -> 电机角度及中间几何量
    
    Args:
        q_roll: 踝关节横滚角 (rad), 绕x轴
        q_pitch: 踝关节俯仰角 (rad), 绕y轴
        left_leg: True=左脚, False=右脚
    
    Returns:
        dict: 包含 THETA, r_A, r_B, r_C, r_bar, r_rod
    """
    l_bar = L_BAR
    l_rod = L_ROD
    l_spacing = L_SPACING if left_leg else -L_SPACING
    
    # B点初始位置(由摇臂角度决定)
    r_B1_0_x = -l_bar * np.cos(LONG_LINK_ANGLE_0)
    r_B1_0_z = A1_Z - l_bar * np.sin(LONG_LINK_ANGLE_0)
    r_B2_0_x = -l_bar * np.cos(SHORT_LINK_ANGLE_0)
    r_B2_0_z = A2_Z - l_bar * np.sin(SHORT_LINK_ANGLE_0)
    
    # 定义初始点
    r_A_0 = [
        np.array([0.0, l_spacing, A1_Z]),
        np.array([0.0, l_spacing, A2_Z])
    ]
    r_B_0 = [
        np.array([r_B1_0_x, l_spacing, r_B1_0_z]),
        np.array([r_B2_0_x, l_spacing, r_B2_0_z])
    ]
    r_C_0 = [
        np.array([C1_X, l_spacing, 0.0]),
        np.array([C2_X, l_spacing, 0.0])
    ]
    
    # 旋转矩阵: R_y(pitch) * R_x(roll)
    cp, sp = np.cos(q_pitch), np.sin(q_pitch)
    cr, sr = np.cos(q_roll), np.sin(q_roll)
    
    R_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])
    x_rot = R_y @ R_x
    
    result = {
        'r_A': [], 'r_B': [], 'r_C': [],
        'r_bar': [], 'r_rod': [], 'THETA': np.zeros(2)
    }
    
    for i in range(2):
        r_A_i = r_A_0[i]
        r_C_i = x_rot @ r_C_0[i]  # 平台点随踝关节旋转
        rBA_bar = r_B_0[i] - r_A_0[i]  # 初始摇臂向量
        
        # 求解摇臂转角theta_i
        a = r_C_i[0] - r_A_i[0]
        b = r_A_i[2] - r_C_i[2]
        c = (l_rod[i]**2 - l_bar**2 - np.sum((r_C_i - r_A_i)**2)) / (2 * l_bar)
        
        a_sq, b_sq, c_sq = a**2, b**2, c**2
        ab_sq_sum = a_sq + b_sq
        
        discriminant = b_sq * c_sq - ab_sq_sum * (c_sq - a_sq)
        if discriminant < 0:
            discriminant = 0.0
        
        asin_arg = (b * c + np.sqrt(discriminant)) / ab_sq_sum
        asin_arg = np.clip(asin_arg, -1.0, 1.0)
        
        theta_i = np.arcsin(asin_arg)
        theta_i = theta_i if a < 0 else -theta_i
        
        # 计算旋转后的B点
        ct, st = np.cos(theta_i), np.sin(theta_i)
        R_y_theta = np.array([
            [ct, 0, st],
            [0, 1, 0],
            [-st, 0, ct]
        ])
        
        r_B_i = r_A_i + R_y_theta @ rBA_bar
        r_bar_i = r_B_i - r_A_i
        r_rod_i = r_C_i - r_B_i
        
        result['r_A'].append(r_A_i)
        result['r_B'].append(r_B_i)
        result['r_C'].append(r_C_i)
        result['r_bar'].append(r_bar_i)
        result['r_rod'].append(r_rod_i)
        result['THETA'][i] = theta_i
    
    return result


def _jacobian_full(r_C: List[np.ndarray], r_bar: List[np.ndarray], 
                   r_rod: List[np.ndarray], q_pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算雅可比矩阵
    
    Returns:
        (J_motor2Joint, J_Joint2motor): 两个2x2矩阵
    """
    s_unit = np.array([0.0, 1.0, 0.0])
    
    # J_x: 2x6, 约束方程对末端位姿的偏导
    J_x = np.zeros((2, 6))
    J_x[0, 0:3] = r_rod[0]
    J_x[0, 3:6] = np.cross(r_C[0], r_rod[0])
    J_x[1, 0:3] = r_rod[1]
    J_x[1, 3:6] = np.cross(r_C[1], r_rod[1])
    
    # J_theta: 2x2, 约束方程对电机角的偏导
    J_theta = np.zeros((2, 2))
    J_theta[0, 0] = np.dot(s_unit, np.cross(r_bar[0], r_rod[0]))
    J_theta[1, 1] = np.dot(s_unit, np.cross(r_bar[1], r_rod[1]))
    
    # J_q: 6x2, 末端位姿对踝关节角的偏导 [roll, pitch]
    cp, sp = np.cos(q_pitch), np.sin(q_pitch)
    J_q = np.zeros((6, 2))
    J_q[3, 1] = cp   # d(omega_x)/d(pitch)
    J_q[4, 0] = 1.0  # d(omega_y)/d(roll)
    J_q[5, 1] = -sp  # d(omega_z)/d(pitch)
    
    J_Temp = J_x @ J_q  # 2x2
    
    # J_motor2Joint = J_Temp^(-1) @ J_theta
    # J_Joint2motor = J_theta^(-1) @ J_Temp
    J_motor2Joint = np.linalg.solve(J_Temp, J_theta)
    J_Joint2motor = np.linalg.solve(J_theta, J_Temp)
    
    return J_motor2Joint, J_Joint2motor


# ============ 公开接口 ============

def ankle_ik(pitch: float, roll: float, left_leg: bool = True) -> Tuple[float, float]:
    """
    逆运动学: 踝关节角度 -> 电机角度
    
    Args:
        pitch: 踝关节俯仰角 (rad)
        roll: 踝关节横滚角 (rad)
        left_leg: True=左脚, False=右脚
    
    Returns:
        (theta1, theta2): 两个电机角度 (rad)
    """
    result = _inverse_kinematics_full(roll, pitch, left_leg)
    return float(result['THETA'][0]), float(result['THETA'][1])


def ankle_fk(theta1: float, theta2: float, left_leg: bool = True,
             initial_guess: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """
    正运动学: 电机角度 -> 踝关节角度 (牛顿迭代)
    
    Args:
        theta1, theta2: 两个电机角度 (rad)
        left_leg: True=左脚, False=右脚
        initial_guess: 初始猜测 (pitch, roll), 可选
    
    Returns:
        (pitch, roll): 踝关节角度 (rad)
    """
    theta_target = np.array([theta1, theta2])
    
    # 初始猜测
    if initial_guess is not None:
        x_c_k = np.array([initial_guess[0], initial_guess[1]])  # [pitch, roll]
    else:
        x_c_k = np.array([0.0, 0.0])
    
    MAX_ITERATIONS = 100
    TOLERANCE = 1e-6
    ALPHA = 0.5
    
    for _ in range(MAX_ITERATIONS):
        current_pitch, current_roll = x_c_k[0], x_c_k[1]
        
        ik_result = _inverse_kinematics_full(current_roll, current_pitch, left_leg)
        J_m2j, _ = _jacobian_full(ik_result['r_C'], ik_result['r_bar'], 
                                   ik_result['r_rod'], current_pitch)
        
        f_error = theta_target - ik_result['THETA']
        
        if np.linalg.norm(f_error) < TOLERANCE:
            break
        
        x_c_k = x_c_k + ALPHA * (J_m2j @ f_error)
    
    return float(x_c_k[0]), float(x_c_k[1])  # pitch, roll


def compute_jacobian(pitch: float, roll: float, left_leg: bool = True) -> np.ndarray:
    """
    计算雅可比矩阵 J = d(theta_motor) / d(ankle)
    
    Args:
        pitch: 踝关节俯仰角 (rad)
        roll: 踝关节横滚角 (rad)
        left_leg: True=左脚, False=右脚
    
    Returns:
        J: 2x2 矩阵, J_Joint2motor
    """
    ik_result = _inverse_kinematics_full(roll, pitch, left_leg)
    _, J_j2m = _jacobian_full(ik_result['r_C'], ik_result['r_bar'],
                              ik_result['r_rod'], pitch)
    return J_j2m


def compute_jacobian_both(pitch: float, roll: float, left_leg: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个雅可比矩阵
    
    Returns:
        (J_motor2Joint, J_Joint2motor)
    """
    ik_result = _inverse_kinematics_full(roll, pitch, left_leg)
    return _jacobian_full(ik_result['r_C'], ik_result['r_bar'],
                          ik_result['r_rod'], pitch)


def motor_vel_to_ankle_vel(pitch: float, roll: float,
                           vel1: float, vel2: float, left_leg: bool = True) -> Tuple[float, float]:
    """
    电机速度 -> 踝关节角速度
    ankle_vel = J_motor2Joint @ motor_vel
    """
    J_m2j, _ = compute_jacobian_both(pitch, roll, left_leg)
    motor_vel = np.array([vel1, vel2])
    ankle_vel = J_m2j @ motor_vel
    return float(ankle_vel[0]), float(ankle_vel[1])  # pitch_vel, roll_vel


def ankle_vel_to_motor_vel(pitch: float, roll: float,
                           pitch_vel: float, roll_vel: float, left_leg: bool = True) -> Tuple[float, float]:
    """
    踝关节角速度 -> 电机速度
    motor_vel = J_Joint2motor @ ankle_vel
    """
    _, J_j2m = compute_jacobian_both(pitch, roll, left_leg)
    ankle_vel = np.array([pitch_vel, roll_vel])
    motor_vel = J_j2m @ ankle_vel
    return float(motor_vel[0]), float(motor_vel[1])


def motor_tau_to_ankle_tau(pitch: float, roll: float,
                           tau1: float, tau2: float, left_leg: bool = True) -> Tuple[float, float]:
    """
    电机扭矩 -> 踝关节扭矩
    ankle_tau = J_Joint2motor^T @ motor_tau
    """
    _, J_j2m = compute_jacobian_both(pitch, roll, left_leg)
    motor_tau = np.array([tau1, tau2])
    ankle_tau = J_j2m.T @ motor_tau
    return float(ankle_tau[0]), float(ankle_tau[1])


def ankle_tau_to_motor_tau(pitch: float, roll: float,
                           tau_pitch: float, tau_roll: float, left_leg: bool = True) -> Tuple[float, float]:
    """
    踝关节扭矩 -> 电机扭矩
    motor_tau = J_motor2Joint^T @ ankle_tau
    """
    J_m2j, _ = compute_jacobian_both(pitch, roll, left_leg)
    ankle_tau = np.array([tau_pitch, tau_roll])
    motor_tau = J_m2j.T @ ankle_tau
    return float(motor_tau[0]), float(motor_tau[1])


if __name__ == '__main__':
    print("=" * 60)
    print("踝关节运动学测试 (异向摇臂)")
    print("=" * 60)
    
    # 测试参数
    test_pitch = np.deg2rad(10.0)
    test_roll = np.deg2rad(5.0)
    
    print(f"\n输入: pitch={np.rad2deg(test_pitch):.2f}°, roll={np.rad2deg(test_roll):.2f}°")
    
    # 逆运动学
    theta1, theta2 = ankle_ik(test_pitch, test_roll, left_leg=True)
    print(f"IK结果: theta1={np.rad2deg(theta1):.4f}°, theta2={np.rad2deg(theta2):.4f}°")
    
    # 正运动学
    pitch_fk, roll_fk = ankle_fk(theta1, theta2, left_leg=True)
    print(f"FK结果: pitch={np.rad2deg(pitch_fk):.4f}°, roll={np.rad2deg(roll_fk):.4f}°")
    
    # 误差
    err_pitch = abs(np.rad2deg(pitch_fk - test_pitch))
    err_roll = abs(np.rad2deg(roll_fk - test_roll))
    print(f"误差: pitch={err_pitch:.6f}°, roll={err_roll:.6f}°")
    
    # 雅可比
    J_m2j, J_j2m = compute_jacobian_both(test_pitch, test_roll, left_leg=True)
    print(f"\nJ_motor2Joint:\n{J_m2j}")
    print(f"\nJ_Joint2motor:\n{J_j2m}")
    
    # 验证雅可比互逆
    identity = J_m2j @ J_j2m
    print(f"\nJ_m2j @ J_j2m (应为单位阵):\n{identity}")
    
    # 速度/力矩变换测试
    test_vel = (0.1, 0.05)  # pitch_vel, roll_vel
    motor_vel = ankle_vel_to_motor_vel(test_pitch, test_roll, *test_vel)
    ankle_vel_back = motor_vel_to_ankle_vel(test_pitch, test_roll, *motor_vel)
    print(f"\n速度变换: {test_vel} -> {motor_vel} -> {ankle_vel_back}")
    
    test_tau = (1.0, 0.5)  # tau_pitch, tau_roll
    motor_tau = ankle_tau_to_motor_tau(test_pitch, test_roll, *test_tau)
    ankle_tau_back = motor_tau_to_ankle_tau(test_pitch, test_roll, *motor_tau)
    print(f"力矩变换: {test_tau} -> {motor_tau} -> {ankle_tau_back}")
    
    print("\n" + "=" * 60)
    if err_pitch < 0.01 and err_roll < 0.01:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
    print("=" * 60)