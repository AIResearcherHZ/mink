"""MuJoCo数字孪生程序 - 临时调试工具
在MuJoCo中调整关节值，真机同步执行
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

sys.path.insert(0, str(Path(__file__).parent / "taks_sdk"))
import taks

# 模型路径
MODELS = {
    "full": Path(__file__).parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml",
    "semi": Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml",
}

# 关节名 -> SDK ID映射
JOINT_TO_SDK = {
    "right_shoulder_pitch_joint": 1, "right_shoulder_roll_joint": 2, "right_shoulder_yaw_joint": 3,
    "right_elbow_joint": 4, "right_wrist_roll_joint": 5, "right_wrist_yaw_joint": 6, "right_wrist_pitch_joint": 7,
    "left_shoulder_pitch_joint": 9, "left_shoulder_roll_joint": 10, "left_shoulder_yaw_joint": 11,
    "left_elbow_joint": 12, "left_wrist_roll_joint": 13, "left_wrist_yaw_joint": 14, "left_wrist_pitch_joint": 15,
    "waist_yaw_joint": 17, "waist_roll_joint": 18, "waist_pitch_joint": 19,
    "neck_yaw_joint": 20, "neck_roll_joint": 21, "neck_pitch_joint": 22,
    # 全身额外关节
    "left_hip_yaw_joint": 23, "left_hip_roll_joint": 24, "left_hip_pitch_joint": 25,
    "left_knee_joint": 26, "left_ankle_pitch_joint": 27, "left_ankle_roll_joint": 28,
    "right_hip_yaw_joint": 29, "right_hip_roll_joint": 30, "right_hip_pitch_joint": 31,
    "right_knee_joint": 32, "right_ankle_pitch_joint": 33, "right_ankle_roll_joint": 34,
}

# 默认KP/KD
SDK_GAINS = {
    1: (20.0, 2.0), 2: (20.0, 2.0), 3: (20.0, 2.0), 4: (20.0, 2.0),
    5: (10.0, 1.0), 6: (10.0, 1.0), 7: (10.0, 1.0), 8: (2.0, 0.2),
    9: (20.0, 2.0), 10: (20.0, 2.0), 11: (20.0, 2.0), 12: (20.0, 2.0),
    13: (10.0, 1.0), 14: (10.0, 1.0), 15: (10.0, 1.0), 16: (2.0, 0.2),
    17: (250.0, 5.0), 18: (250.0, 5.0), 19: (250.0, 5.0),
    20: (1.0, 0.5), 21: (1.0, 0.5), 22: (1.0, 0.5),
    23: (100.0, 5.0), 24: (100.0, 5.0), 25: (100.0, 5.0),
    26: (100.0, 5.0), 27: (50.0, 2.0), 28: (50.0, 2.0),
    29: (100.0, 5.0), 30: (100.0, 5.0), 31: (100.0, 5.0),
    32: (100.0, 5.0), 33: (50.0, 2.0), 34: (50.0, 2.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo数字孪生")
    parser.add_argument("--model", type=str, default="semi", choices=["full", "semi"], help="模型类型")
    parser.add_argument("--host", type=str, default="192.168.5.16", help="taks服务器地址")
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
    
    # 构建关节映射: jname -> (qpos_idx, sdk_id)
    joint_map = {}
    for jname, sdk_id in JOINT_TO_SDK.items():
        try:
            jid = model.joint(jname).id
            qpos_idx = model.jnt_qposadr[jid]
            joint_map[jname] = (qpos_idx, sdk_id)
        except:
            pass  # 模型中不存在该关节
    
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
    
    def signal_handler(signum, frame):
        nonlocal running, shutdown_requested
        if shutdown_requested:
            sys.exit(1)
        shutdown_requested = True
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def send_to_real():
        """发送当前MuJoCo关节位置到真机"""
        if not enable_real or robot is None:
            return
        mit_cmd = {}
        for jname, (qpos_idx, sdk_id) in joint_map.items():
            q_val = float(data.qpos[qpos_idx])
            kp, kd = SDK_GAINS.get(sdk_id, (10.0, 1.0))
            mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp, 'kd': kd}
        if mit_cmd:
            robot.controlMIT(mit_cmd)
    
    rate = RateLimiter(frequency=50.0, warn=False)
    print(f"[Info] 模型: {args.model}, 真机: {'ON' if enable_real else 'OFF'}")
    print("[Info] 在MuJoCo GUI中拖动关节，真机会同步执行")
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        while running and viewer.is_running():
            mujoco.mj_forward(model, data)
            send_to_real()
            viewer.sync()
            rate.sleep()
    
    # 清理
    if enable_real:
        # 失能所有关节
        mit_cmd = {}
        for jname, (_, sdk_id) in joint_map.items():
            mit_cmd[sdk_id] = {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0}
        if robot:
            robot.controlMIT(mit_cmd)
        taks.disconnect()
        print("[TAKS] 已断开")


if __name__ == "__main__":
    main()
