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
    1: (20, 5), 2: (20, 5), 3: (20, 5), 4: (20, 5),
    5: (10, 1), 6: (10, 1), 7: (10, 1), 8: (1.5, 0.1),
    9: (20, 5), 10: (20, 5), 11: (20, 5), 12: (20, 5),
    13: (10, 1), 14: (10, 1), 15: (10, 1), 16: (1.5, 0.1),
    17: (150, 4), 18: (150, 4), 19: (150, 4),
    20: (1.5, 0.1), 21: (1.5, 0.1), 22: (1.5, 0.1),
    23: (50, 50), 24: (150, 50), 25: (150, 50),
    26: (50, 50), 27: (40, 2), 28: (40, 2),
    29: (50, 50), 30: (150, 50), 31: (150, 50),
    32: (50, 50), 33: (40, 2), 34: (40, 2),
}

# 关节调整步长
JOINT_STEP = 0.05

# 启停配置
RAMP_UP_TIME = 5.0    # 缓启动时间(秒)
RAMP_DOWN_TIME = 5.0  # 缓停止时间(秒)

# 安全倒向配置: 停止时主动倒向此方向
SAFE_FALL_POSITIONS = {
    4: 1.2,     # right_elbow_joint
    12: 1.2,    # left_elbow_joint
    17: 0.0,    # waist_yaw_joint
    18: 0.52,    # waist_roll_joint
    19: -0.45,  # waist_pitch_joint
}

# 安全模式 kp/kd (缓停止结束时的目标值)
SAFE_KP_KD = {
    1: (5.0, 1.0), 2: (5.0, 1.0), 3: (5.0, 1.0), 4: (5.0, 1.0),
    5: (5.0, 1.0), 6: (5.0, 1.0), 7: (5.0, 1.0), 8: (2.0, 0.2),
    9: (5.0, 1.0), 10: (5.0, 1.0), 11: (5.0, 1.0), 12: (5.0, 1.0),
    13: (5.0, 1.0), 14: (5.0, 1.0), 15: (5.0, 1.0), 16: (2.0, 0.2),
    17: (25.0, 1.0), 18: (25.0, 1.0), 19: (25.0, 1.0),
    20: (1.0, 0.5), 21: (1.0, 0.5), 22: (1.0, 0.5),
    # 腿部关节
    23: (20.0, 2.0), 24: (20.0, 2.0), 25: (20.0, 2.0),
    26: (20.0, 2.0), 27: (10.0, 1.0), 28: (10.0, 1.0),
    29: (20.0, 2.0), 30: (20.0, 2.0), 31: (20.0, 2.0),
    32: (20.0, 2.0), 33: (10.0, 1.0), 34: (10.0, 1.0),
}

# 非线性缓动参数 (可调)

# 作用: 控制 kp/kd 增益在 ramp up/down 过程中的变化曲线
# - Ramp Up (缓启动): kp/kd 从 0 渐变到目标值，使电机平滑启动
# - Ramp Down (缓停止): kp/kd 从当前值渐变到安全值，使电机平滑停止

# RAMP_EXPONENT 参数说明（形象理解）:
# - 值越大，曲线越弯曲，过渡越平滑
# - = 1.0: 线性变化，像“匀速走路”
# - = 2.0: 默认，开始稍快、尾部渐缓
# - = 3.0: 开始更快、后段更柔，像“先迅速起步再慢慢刹车”

# Ramp Up (ease_out): **开始更快，后面放缓**
#   - 增大 exponent → kp/kd 前半段冲得更快，越接近目标越慢，避免突然“硬”起来
# Ramp Down (ease_in): **开始更慢，后面更快**
#   - 增大 exponent → kp/kd 前半段降得更慢，越接近结束越快归零/安全值，避免“突然塌陷”

RAMP_EXPONENT = 1.05  # 调参: 增大此值使曲线更弯

def ease_out(t: float, exp: float = RAMP_EXPONENT) -> float:
    """缓出函数: 开始快，结束慢 (用于 ramp up)
    t: 进度 [0,1]
    exp: 指数，越大曲线越弯
    返回: 输出 [0,1]
    公式: 1 - (1-t)^exp
    """
    return 1.0 - pow(1.0 - t, exp)

def ease_in(t: float, exp: float = RAMP_EXPONENT) -> float:
    """缓入函数: 开始慢，结束快 (用于 ramp down)
    t: 进度 [0,1]
    exp: 指数，越大曲线越弯
    返回: 输出 [0,1]
    公式: t^exp
    """
    return pow(t, exp)


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
    sdk_path = Path(__file__).parent / "taks_sdk" / "SDK_MF.py"
    if not sdk_path.exists():
        print(f"[SDK] 错误: SDK_MF.py 不存在: {sdk_path}")
        return None
    print(f"[SDK] 本机模式: 启动 SDK 服务端...")
    proc = subprocess.Popen(
        [sys.executable, str(sdk_path)],
        stdout=None,  # 继承父进程的stdout，显示输出
        stderr=None,  # 继承父进程的stderr
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
    left_gripper = None
    right_gripper = None
    if enable_real:
        try:
            taks.connect(args.host, cmd_port=args.port)
            device_type = "Taks-T1" if args.model == "full" else "Taks-T1-semibody"
            robot = taks.register(device_type)
            print(f"[TAKS] 已注册设备: {device_type}")
            time.sleep(4.0)
            print(f"[TAKS] 等待4秒后注册gripper...")
            left_gripper = taks.register("Taks-T1-leftgripper")
            print(f"[TAKS] 已注册左gripper")
            time.sleep(1.0)
            right_gripper = taks.register("Taks-T1-rightgripper")
            print(f"[TAKS] 已注册右gripper")
            time.sleep(1.0)
            print(f"[TAKS] 已连接 {args.host}:{args.port}, 所有设备注册完成")
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
    
    def get_real_positions(timeout: float = 2.0):
        """获取真机当前位置，等待有效数据
        timeout: 等待超时时间(秒)，超时后返回0
        """
        if not enable_real or robot is None:
            return {sdk_id: 0.0 for _, _, sdk_id in joint_info}
        
        # 等待有效数据
        start = time.time()
        while time.time() - start < timeout:
            real_pos = robot.GetPosition()
            if real_pos is not None and len(real_pos) > 0:
                return {sdk_id: real_pos.get(sdk_id, 0.0) for _, _, sdk_id in joint_info}
            time.sleep(0.05)  # 20Hz轮询
        
        print(f"[警告] 获取真机位置超时({timeout}s)，使用0作为起始位置")
        return {sdk_id: 0.0 for _, _, sdk_id in joint_info}
    
    def send_to_real(ramp_progress=1.0):
        """发送当前MuJoCo关节位置到真机
        ramp_progress: 启动进度[0,1]，用于非线性kp/kd渐变
        - ramp up: kp/kd 从0非线性(ease_out:越来越慢)增加到目标值
        - 正常运行: ramp_progress=1.0, 使用完整kp/kd
        """
        if not enable_real or robot is None:
            return
        mit_cmd = {}
        # 非线性kp/kd缩放因子 (ease_out: 开始快，结束慢)
        kp_kd_scale = ease_out(ramp_progress) if ramp_progress < 1.0 else 1.0
        for jname, qpos_idx, sdk_id in joint_info:
            q_target = float(data.qpos[qpos_idx])
            kp_target, kd_target = SDK_GAINS.get(sdk_id, (10.0, 1.0))
            # 启动阶段: kp/kd非线性渐变，位置直接发送目标位置
            kp_val = kp_target * kp_kd_scale
            kd_val = kd_target * kp_kd_scale
            mit_cmd[sdk_id] = {'q': q_target, 'dq': 0.0, 'tau': 0.0, 'kp': kp_val, 'kd': kd_val}
        if mit_cmd:
            robot.controlMIT(mit_cmd)
    
    def ramp_down():
        """安全停止: kp/kd非线性(ease_in:越来越快)从目标值降低到安全值，然后失能
        使用 ease_in 函数: 开始慢，结束快
        """
        if not enable_real or robot is None:
            return
        if not enable_ramp_down:
            # 直接失能
            mit_cmd = {sdk_id: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for _, _, sdk_id in joint_info}
            robot.controlMIT(mit_cmd)
            print("[Ramp Down] 缓停止已禁用，直接失能")
            return
        
        print(f"[Ramp Down] 非线性降低kp/kd到安全值 ({ramp_down_time}s)...")
        start = time.time()
        start_positions = get_real_positions()
        
        while time.time() - start < ramp_down_time:
            elapsed = time.time() - start
            t = elapsed / ramp_down_time  # 线性进度 [0,1]
            # 非线性kp/kd缩放因子 (ease_in: 开始慢，结束快)
            # t=0时 scale=1(目标kp/kd), t=1时 scale=0(安全kp/kd)
            kp_kd_scale = 1.0 - ease_in(t)
            mit_cmd = {}
            for jname, qpos_idx, sdk_id in joint_info:
                kp_target, kd_target = SDK_GAINS.get(sdk_id, (10.0, 1.0))
                kp_safe, kd_safe = SAFE_KP_KD.get(sdk_id, (5.0, 1.0))
                # kp/kd从目标值渐变到安全值
                kp_val = kp_safe + (kp_target - kp_safe) * kp_kd_scale
                kd_val = kd_safe + (kd_target - kd_safe) * kp_kd_scale
                # 位置保持当前位置
                q_val = start_positions.get(sdk_id, 0.0)
                mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp_val, 'kd': kd_val}
            if mit_cmd:
                robot.controlMIT(mit_cmd)
            time.sleep(0.005)  # 200Hz
        # 到达安全kp/kd后失能
        mit_cmd = {sdk_id: {'q': start_positions.get(sdk_id, 0.0), 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for _, _, sdk_id in joint_info}
        robot.controlMIT(mit_cmd)
        print("[Ramp Down] 已降低到安全kp/kd并失能")
    
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