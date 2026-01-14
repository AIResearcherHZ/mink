"""MuJoCo数字孪生程序 - 临时调试工具
键盘控制关节，真机同步执行
"""

import sys
import argparse
import signal
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import time

sys.path.insert(0, str(Path(__file__).parent))
from taks_sdk import taks

# 模型路径
MODELS = {
    "full": Path(__file__).parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml",
    "semi": Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml",
}

# 半身关节列表（按索引升序）
JOINT_LIST_SEMI = [
    ("right_shoulder_pitch_joint", 1),
    ("right_shoulder_roll_joint", 2),
    ("right_shoulder_yaw_joint", 3),
    ("right_elbow_joint", 4),
    ("right_wrist_roll_joint", 5),
    ("right_wrist_yaw_joint", 6),
    ("right_wrist_pitch_joint", 7),
    ("left_shoulder_pitch_joint", 9),
    ("left_shoulder_roll_joint", 10),
    ("left_shoulder_yaw_joint", 11),
    ("left_elbow_joint", 12),
    ("left_wrist_roll_joint", 13),
    ("left_wrist_yaw_joint", 14),
    ("left_wrist_pitch_joint", 15),
    ("waist_yaw_joint", 17),
    ("waist_roll_joint", 18),
    ("waist_pitch_joint", 19),
    ("neck_yaw_joint", 20),
    ("neck_roll_joint", 21),
    ("neck_pitch_joint", 22),
]

# 全身关节列表（腿部+半身，按索引升序）
JOINT_LIST_FULL = [
    ("right_shoulder_pitch_joint", 1),
    ("right_shoulder_roll_joint", 2),
    ("right_shoulder_yaw_joint", 3),
    ("right_elbow_joint", 4),
    ("right_wrist_roll_joint", 5),
    ("right_wrist_yaw_joint", 6),
    ("right_wrist_pitch_joint", 7),
    ("left_shoulder_pitch_joint", 9),
    ("left_shoulder_roll_joint", 10),
    ("left_shoulder_yaw_joint", 11),
    ("left_elbow_joint", 12),
    ("left_wrist_roll_joint", 13),
    ("left_wrist_yaw_joint", 14),
    ("left_wrist_pitch_joint", 15),
    ("waist_yaw_joint", 17),
    ("waist_roll_joint", 18),
    ("waist_pitch_joint", 19),
    ("neck_yaw_joint", 20),
    ("neck_roll_joint", 21),
    ("neck_pitch_joint", 22),
    ("right_hip_pitch_joint", 23),
    ("right_hip_roll_joint", 24),
    ("right_hip_yaw_joint", 25),
    ("right_knee_joint", 26),
    ("right_ankle_pitch_joint", 27),
    ("right_ankle_roll_joint", 28),
    ("left_hip_pitch_joint", 29),
    ("left_hip_roll_joint", 30),
    ("left_hip_yaw_joint", 31),
    ("left_knee_joint", 32),
    ("left_ankle_pitch_joint", 33),
    ("left_ankle_roll_joint", 34),
]

# 默认KP/KD
SDK_GAINS = {
    1: (20.0, 2.0), 2: (20.0, 2.0), 3: (20.0, 2.0), 4: (20.0, 2.0),
    5: (10.0, 1.0), 6: (10.0, 1.0), 7: (10.0, 1.0), 8: (2.0, 0.2),
    9: (20.0, 2.0), 10: (20.0, 2.0), 11: (20.0, 2.0), 12: (20.0, 2.0),
    13: (10.0, 1.0), 14: (10.0, 1.0), 15: (10.0, 1.0), 16: (2.0, 0.2),
    17: (250.0, 5.0), 18: (250.0, 5.0), 19: (250.0, 5.0),
    20: (1.0, 0.5), 21: (1.0, 0.5), 22: (1.0, 0.5),
    # 腿部关节
    23: (100.0, 5.0), 24: (100.0, 5.0), 25: (100.0, 5.0),
    26: (100.0, 5.0), 27: (50.0, 2.0), 28: (50.0, 2.0),
    29: (100.0, 5.0), 30: (100.0, 5.0), 31: (100.0, 5.0),
    32: (100.0, 5.0), 33: (50.0, 2.0), 34: (50.0, 2.0),
}

# 关节调整步长
JOINT_STEP = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo数字孪生")
    parser.add_argument("--model", type=str, default="semi", choices=["full", "semi"], help="模型类型")
    parser.add_argument("--host", type=str, default="192.168.5.7", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", help="禁用真机(仅仿真)")
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = MODELS[args.model]
    enable_real = not args.no_real
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 根据模型选择关节列表
    joint_list = JOINT_LIST_FULL if args.model == "full" else JOINT_LIST_SEMI
    
    # 构建关节映射: [(jname, qpos_idx, sdk_id), ...]
    joint_info = []
    for jname, sdk_id in joint_list:
        try:
            jid = model.joint(jname).id
            qpos_idx = model.jnt_qposadr[jid]
            joint_info.append((jname, qpos_idx, sdk_id))
        except:
            pass
    
    # 当前选中的关节索引
    current_joint_idx = [0]
    
    # 连接真机
    robot = None
    if enable_real:
        try:
            taks.connect(args.host, cmd_port=args.port)
            device_type = "Taks-T1" if args.model == "full" else "Taks-T1-semibody"
            robot = taks.register(device_type)
            print(f"[TAKS] 已连接 {args.host}:{args.port}, 设备: {device_type}")
        except Exception as e:
            print(f"[TAKS] 连接失败: {e}")
            enable_real = False
    
    running = True
    shutdown_requested = False
    need_send = [True]  # 初始发送一次
    
    def signal_handler(signum, frame):
        nonlocal running, shutdown_requested
        if shutdown_requested:
            sys.exit(1)
        shutdown_requested = True
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def key_callback(keycode):
        """键盘回调: 上下选关节，左右调值，R复位"""
        idx = current_joint_idx[0]
        jname, qpos_idx, sdk_id = joint_info[idx]
        
        if keycode == 265:  # UP
            current_joint_idx[0] = (idx - 1) % len(joint_info)
            print_current_joint()
        elif keycode == 264:  # DOWN
            current_joint_idx[0] = (idx + 1) % len(joint_info)
            print_current_joint()
        elif keycode == 263:  # LEFT
            data.qpos[qpos_idx] -= JOINT_STEP
            need_send[0] = True
            print_current_joint()
        elif keycode == 262:  # RIGHT
            data.qpos[qpos_idx] += JOINT_STEP
            need_send[0] = True
            print_current_joint()
        elif keycode == 82:  # R = Reset
            mujoco.mj_resetDataKeyframe(model, data, 0)
            need_send[0] = True
            print("[Reset] 已复位到初始位置")
        elif keycode == 48:  # 0 = 当前关节归零
            data.qpos[qpos_idx] = 0.0
            need_send[0] = True
            print_current_joint()
    
    def print_current_joint():
        idx = current_joint_idx[0]
        jname, qpos_idx, sdk_id = joint_info[idx]
        print(f"[{idx+1}/{len(joint_info)}] {jname} (SDK:{sdk_id}) = {data.qpos[qpos_idx]:.4f}")
    
    def send_to_real():
        """发送当前MuJoCo关节位置到真机"""
        if not enable_real or robot is None:
            return
        mit_cmd = {}
        for jname, qpos_idx, sdk_id in joint_info:
            q_val = float(data.qpos[qpos_idx])
            kp, kd = SDK_GAINS.get(sdk_id, (10.0, 1.0))
            mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp, 'kd': kd}
        if mit_cmd:
            robot.controlMIT(mit_cmd)
    
    rate = RateLimiter(frequency=50.0, warn=False)
    print(f"[Info] 模型: {args.model}, 真机: {'ON' if enable_real else 'OFF'}")
    print(f"[Info] 可控关节数: {len(joint_info)}")
    print("[操作说明]")
    print("  ↑/↓ = 切换关节")
    print("  ←/→ = 减少/增加关节值")
    print("  R   = 复位所有关节")
    print("  0   = 当前关节归零")
    print_current_joint()
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False,
                                       key_callback=key_callback) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        while running and viewer.is_running():
            mujoco.mj_forward(model, data)
            
            if need_send[0]:
                send_to_real()
                need_send[0] = False
            
            viewer.sync()
            rate.sleep()
    
    # 清理
    if enable_real:
        mit_cmd = {sdk_id: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for _, _, sdk_id in joint_info}
        if robot and mit_cmd:
            robot.controlMIT(mit_cmd)
        taks.disconnect()
        print("[TAKS] 已断开")


if __name__ == "__main__":
    main()