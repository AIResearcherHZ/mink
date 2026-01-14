"""MuJoCo数字孪生程序 - 临时调试工具
键盘控制关节，真机同步执行
"""

import sys
import argparse
import signal
import subprocess
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

# 启停配置
RAMP_UP_TIME = 3.0    # 缓启动时间(秒)
RAMP_DOWN_TIME = 3.0  # 缓停止时间(秒)

# 安全倒向配置: 停止时主动倒向此方向
SAFE_FALL_POSITIONS = {
    17: 0.0,   # waist_yaw: 保持中位
    18: 0.15,  # waist_roll: 微向右倒
    19: -0.20, # waist_pitch: 微向后倒
}


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo数字孪生")
    parser.add_argument("--model", type=str, default="semi", choices=["full", "semi"], help="模型类型")
    parser.add_argument("--headless", action="store_true", default=True, help="无头模式(无GUI)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", help="禁用真机(仅仿真)")
    parser.add_argument("--no-ramp-up", action="store_true", default=False, help="禁用缓启动")
    parser.add_argument("--no-ramp-down", action="store_true", default=False, help="禁用缓停止")
    parser.add_argument("--ramp-up-time", type=float, default=RAMP_UP_TIME, help="缓启动时间(秒)")
    parser.add_argument("--ramp-down-time", type=float, default=RAMP_DOWN_TIME, help="缓停止时间(秒)")
    return parser.parse_args()


def start_local_sdk():
    """启动本机SDK服务端子进程"""
    sdk_path = Path(__file__).parent / "taks_sdk" / "SDK.py"
    if not sdk_path.exists():
        print(f"[SDK] 错误: SDK.py 不存在: {sdk_path}")
        return None
    print(f"[SDK] 本机模式: 启动 SDK 服务端...")
    proc = subprocess.Popen(
        [sys.executable, str(sdk_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(sdk_path.parent)
    )
    time.sleep(3.0)  # 等待SDK初始化
    if proc.poll() is not None:
        print(f"[SDK] 错误: SDK 服务端启动失败")
        return None
    print(f"[SDK] SDK 服务端已启动 (PID: {proc.pid})")
    return proc


def main():
    args = parse_args()
    xml_path = MODELS[args.model]
    headless = args.headless
    enable_real = not args.no_real
    enable_ramp_up = not args.no_ramp_up
    enable_ramp_down = not args.no_ramp_down
    ramp_up_time = args.ramp_up_time
    ramp_down_time = args.ramp_down_time
    
    # 本机模式: 自动启动SDK服务端
    sdk_proc = None
    is_local = args.host in ("127.0.0.1", "localhost") and enable_real
    if is_local:
        sdk_proc = start_local_sdk()
        if sdk_proc is None:
            print("[SDK] 本机SDK启动失败，切换到仅仿真模式")
            enable_real = False
    
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
    
    # 缓启动状态
    ramp_state = {"active": enable_ramp_up and enable_real, "start_time": None, "progress": 0.0 if enable_ramp_up else 1.0, "start_positions": {}}
    
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
    
    def get_real_positions():
        """获取真机当前位置"""
        if not enable_real or robot is None:
            return {sdk_id: 0.0 for _, _, sdk_id in joint_info}
        real_pos = robot.GetPosition()
        if real_pos is None:
            return {sdk_id: 0.0 for _, _, sdk_id in joint_info}
        return {sdk_id: real_pos.get(sdk_id, 0.0) for _, _, sdk_id in joint_info}
    
    def send_to_real(ramp_progress=1.0):
        """发送当前MuJoCo关节位置到真机"""
        if not enable_real or robot is None:
            return
        mit_cmd = {}
        for jname, qpos_idx, sdk_id in joint_info:
            q_target = float(data.qpos[qpos_idx])
            kp, kd = SDK_GAINS.get(sdk_id, (10.0, 1.0))
            # 启动阶段: 从真机起始位置线性插值到目标位置
            if ramp_progress < 1.0:
                start_pos = ramp_state["start_positions"].get(sdk_id, 0.0)
                q_val = start_pos + (q_target - start_pos) * ramp_progress
            else:
                q_val = q_target
            mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp, 'kd': kd}
        if mit_cmd:
            robot.controlMIT(mit_cmd)
    
    def ramp_down():
        """安全停止: 从真机当前位置线性移动到安全倒向位置，然后失能"""
        if not enable_real or robot is None:
            return
        if not enable_ramp_down:
            # 直接失能
            mit_cmd = {sdk_id: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for _, _, sdk_id in joint_info}
            robot.controlMIT(mit_cmd)
            print("[Ramp Down] 缓停止已禁用，直接失能")
            return
        
        print(f"[Ramp Down] 线性移动到安全位置 ({ramp_down_time}s)...")
        start = time.time()
        start_positions = get_real_positions()
        
        while time.time() - start < ramp_down_time:
            elapsed = time.time() - start
            t = elapsed / ramp_down_time  # 线性进度 [0,1]
            mit_cmd = {}
            for jname, qpos_idx, sdk_id in joint_info:
                kp, kd = SDK_GAINS.get(sdk_id, (10.0, 1.0))
                start_pos = start_positions.get(sdk_id, 0.0)
                target_pos = SAFE_FALL_POSITIONS.get(sdk_id, 0.0)
                q_val = start_pos + (target_pos - start_pos) * t
                mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp, 'kd': kd}
            if mit_cmd:
                robot.controlMIT(mit_cmd)
            time.sleep(0.005)  # 200Hz
        # 到达安全位置后失能
        mit_cmd = {sdk_id: {'q': SAFE_FALL_POSITIONS.get(sdk_id, 0.0), 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for _, _, sdk_id in joint_info}
        robot.controlMIT(mit_cmd)
        print("[Ramp Down] 已到达安全位置并失能")
    
    rate = RateLimiter(frequency=50.0, warn=False)
    mode_str = "有头" if not headless else "无头"
    real_str = "SIM2REAL" if enable_real else "仅仿真"
    print(f"[Info] 模型: {args.model}, 模式: {mode_str}, {real_str}")
    print(f"[Info] 可控关节数: {len(joint_info)}")
    if not headless:
        print("[操作说明]")
        print("  ↑/↓ = 切换关节")
        print("  ←/→ = 减少/增加关节值")
        print("  R   = 复位所有关节")
        print("  0   = 当前关节归零")
        print_current_joint()
    
    def control_loop(viewer=None):
        nonlocal running
        
        # 初始化启动: 从真机获取当前位置作为起点
        if ramp_state["active"]:
            ramp_state["start_time"] = time.time()
            ramp_state["start_positions"] = get_real_positions()
            print(f"[Ramp Up] 线性启动 ({ramp_up_time}s)...")
        else:
            ramp_state["progress"] = 1.0
            if enable_real:
                print("[Ramp Up] 缓启动已禁用")
        
        while running:
            if viewer is not None and not viewer.is_running():
                break
            
            # 启动处理: 线性位置插值
            if ramp_state["active"]:
                elapsed = time.time() - ramp_state["start_time"]
                if elapsed >= ramp_up_time:
                    ramp_state["active"] = False
                    ramp_state["progress"] = 1.0
                    print("[Ramp Up] 启动完成")
                else:
                    ramp_state["progress"] = elapsed / ramp_up_time
            
            mujoco.mj_forward(model, data)
            
            if need_send[0]:
                send_to_real(ramp_state["progress"])
                need_send[0] = False
            
            if viewer:
                viewer.sync()
            rate.sleep()
    
    try:
        if headless:
            print("[Info] 无头模式运行中，Ctrl+C退出")
            control_loop(viewer=None)
        else:
            with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False,
                                               key_callback=key_callback) as viewer:
                mujoco.mjv_defaultFreeCamera(model, viewer.cam)
                control_loop(viewer=viewer)
    except KeyboardInterrupt:
        print("\n[Info] 用户中断")
    finally:
        running = False
    
    # 清理
    if enable_real:
        ramp_down()
        taks.disconnect()
        print("[TAKS] 已断开")
    # 清理本机SDK子进程
    if sdk_proc is not None:
        print("[SDK] 关闭本机SDK服务端...")
        sdk_proc.terminate()
        try:
            sdk_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            sdk_proc.kill()
        print("[SDK] SDK服务端已关闭")


if __name__ == "__main__":
    main()