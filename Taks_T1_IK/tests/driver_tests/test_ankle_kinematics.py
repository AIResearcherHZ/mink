#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
踝关节运动学测试脚本
测试 ankle_kinematics.py 的逆解、正解、雅可比、速度/力矩变换
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from libs.drivers.ankle_kinematics import (
    ankle_ik, ankle_fk, 
    compute_jacobian, compute_jacobian_both,
    motor_vel_to_ankle_vel, ankle_vel_to_motor_vel,
    motor_tau_to_ankle_tau, ankle_tau_to_motor_tau,
    L_BAR, L_ROD, L_SPACING
)


def test_ik_fk_consistency():
    """测试逆解和正解的一致性"""
    print("\n" + "=" * 60)
    print("测试1: IK/FK一致性")
    print("=" * 60)
    
    test_cases = [
        (0.0, 0.0, True, "左脚零位"),
        (0.0, 0.0, False, "右脚零位"),
        (np.deg2rad(10), np.deg2rad(5), True, "左脚 pitch=10° roll=5°"),
        (np.deg2rad(-10), np.deg2rad(5), True, "左脚 pitch=-10° roll=5°"),
        (np.deg2rad(10), np.deg2rad(-5), True, "左脚 pitch=10° roll=-5°"),
        (np.deg2rad(15), np.deg2rad(10), False, "右脚 pitch=15° roll=10°"),
        (np.deg2rad(5), np.deg2rad(3), True, "左脚小角度"),
    ]
    
    all_passed = True
    for pitch, roll, left_leg, desc in test_cases:
        theta1, theta2 = ankle_ik(pitch, roll, left_leg)
        pitch_fk, roll_fk = ankle_fk(theta1, theta2, left_leg)
        
        err_pitch = abs(np.rad2deg(pitch_fk - pitch))
        err_roll = abs(np.rad2deg(roll_fk - roll))
        
        passed = err_pitch < 0.01 and err_roll < 0.01
        status = "✓" if passed else "✗"
        
        print(f"{status} {desc}")
        print(f"   输入: pitch={np.rad2deg(pitch):.2f}° roll={np.rad2deg(roll):.2f}°")
        print(f"   IK: theta1={np.rad2deg(theta1):.4f}° theta2={np.rad2deg(theta2):.4f}°")
        print(f"   FK: pitch={np.rad2deg(pitch_fk):.4f}° roll={np.rad2deg(roll_fk):.4f}°")
        print(f"   误差: pitch={err_pitch:.6f}° roll={err_roll:.6f}°")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_jacobian_inverse():
    """测试雅可比矩阵互逆性"""
    print("\n" + "=" * 60)
    print("测试2: 雅可比矩阵互逆性")
    print("=" * 60)
    
    test_cases = [
        (0.0, 0.0, True),
        (np.deg2rad(10), np.deg2rad(5), True),
        (np.deg2rad(-5), np.deg2rad(8), False),
    ]
    
    all_passed = True
    for pitch, roll, left_leg in test_cases:
        J_m2j, J_j2m = compute_jacobian_both(pitch, roll, left_leg)
        identity = J_m2j @ J_j2m
        
        is_identity = np.allclose(identity, np.eye(2), atol=1e-6)
        status = "✓" if is_identity else "✗"
        leg_str = "左脚" if left_leg else "右脚"
        
        print(f"{status} {leg_str} pitch={np.rad2deg(pitch):.1f}° roll={np.rad2deg(roll):.1f}°")
        print(f"   J_m2j @ J_j2m =\n{identity}")
        
        if not is_identity:
            all_passed = False
    
    return all_passed


def test_velocity_transform():
    """测试速度变换的可逆性"""
    print("\n" + "=" * 60)
    print("测试3: 速度变换可逆性")
    print("=" * 60)
    
    pitch, roll = np.deg2rad(10), np.deg2rad(5)
    test_vels = [(0.1, 0.05), (0.2, -0.1), (-0.15, 0.08)]
    
    all_passed = True
    for pitch_vel, roll_vel in test_vels:
        # ankle -> motor -> ankle
        motor_vel = ankle_vel_to_motor_vel(pitch, roll, pitch_vel, roll_vel)
        ankle_vel_back = motor_vel_to_ankle_vel(pitch, roll, *motor_vel)
        
        err = np.linalg.norm([pitch_vel - ankle_vel_back[0], roll_vel - ankle_vel_back[1]])
        passed = err < 1e-10
        status = "✓" if passed else "✗"
        
        print(f"{status} 踝关节速度 ({pitch_vel:.3f}, {roll_vel:.3f})")
        print(f"   -> 电机速度 {motor_vel}")
        print(f"   -> 踝关节速度 {ankle_vel_back}")
        print(f"   误差: {err:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_torque_transform():
    """测试力矩变换的可逆性"""
    print("\n" + "=" * 60)
    print("测试4: 力矩变换可逆性")
    print("=" * 60)
    
    pitch, roll = np.deg2rad(10), np.deg2rad(5)
    test_taus = [(1.0, 0.5), (2.0, -1.0), (-1.5, 0.8)]
    
    all_passed = True
    for tau_pitch, tau_roll in test_taus:
        # ankle -> motor -> ankle
        motor_tau = ankle_tau_to_motor_tau(pitch, roll, tau_pitch, tau_roll)
        ankle_tau_back = motor_tau_to_ankle_tau(pitch, roll, *motor_tau)
        
        err = np.linalg.norm([tau_pitch - ankle_tau_back[0], tau_roll - ankle_tau_back[1]])
        passed = err < 1e-10
        status = "✓" if passed else "✗"
        
        print(f"{status} 踝关节力矩 ({tau_pitch:.3f}, {tau_roll:.3f})")
        print(f"   -> 电机力矩 {motor_tau}")
        print(f"   -> 踝关节力矩 {ankle_tau_back}")
        print(f"   误差: {err:.2e}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_geometry_params():
    """显示几何参数"""
    print("\n" + "=" * 60)
    print("几何参数 (异向摇臂)")
    print("=" * 60)
    print(f"L_BAR (摇臂长度): {L_BAR} mm")
    print(f"L_ROD (作动杆长度): {L_ROD} mm")
    print(f"L_SPACING (y向偏置): ±{L_SPACING} mm")


def main():
    print("=" * 60)
    print("踝关节运动学测试 (异向摇臂并联机构)")
    print("=" * 60)
    
    test_geometry_params()
    
    results = []
    results.append(("IK/FK一致性", test_ik_fk_consistency()))
    results.append(("雅可比互逆性", test_jacobian_inverse()))
    results.append(("速度变换可逆性", test_velocity_transform()))
    results.append(("力矩变换可逆性", test_torque_transform()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
