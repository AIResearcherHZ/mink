"""半身VR控制IK - SIM2REAL版

解决多路IK跳变: 使用DofFreezingTask作为equality constraint冻结非活动关节
支持taks SDK实现SIM2REAL控制
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
import mink
import time
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from vr_interface import VRReceiver
from taks_sdk import taks

_XML = Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"

# 关节分组
JOINT_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

# MuJoCo关节名 -> SDK关节ID映射
JOINT_NAME_TO_SDK_ID = {
    "right_shoulder_pitch_joint": 1, "right_shoulder_roll_joint": 2, "right_shoulder_yaw_joint": 3,
    "right_elbow_joint": 4, "right_wrist_roll_joint": 5, "right_wrist_yaw_joint": 6, "right_wrist_pitch_joint": 7,
    "left_shoulder_pitch_joint": 9, "left_shoulder_roll_joint": 10, "left_shoulder_yaw_joint": 11,
    "left_elbow_joint": 12, "left_wrist_roll_joint": 13, "left_wrist_yaw_joint": 14, "left_wrist_pitch_joint": 15,
    "waist_yaw_joint": 17, "waist_roll_joint": 18, "waist_pitch_joint": 19,
    "neck_yaw_joint": 20, "neck_roll_joint": 21, "neck_pitch_joint": 22,
}

# 全局关节默认KP/KD
SDK_JOINT_GAINS = {
    1: (20.0, 2.0), 2: (20.0, 2.0), 3: (20.0, 2.0), 4: (20.0, 2.0),
    5: (10.0, 1.0), 6: (10.0, 1.0), 7: (10.0, 1.0), 8: (2.0, 0.2),
    9: (20.0, 2.0), 10: (20.0, 2.0), 11: (20.0, 2.0), 12: (20.0, 2.0),
    13: (10.0, 1.0), 14: (10.0, 1.0), 15: (10.0, 1.0), 16: (2.0, 0.2),
    17: (250.0, 5.0), 18: (250.0, 5.0), 19: (250.0, 5.0),
    20: (1.0, 0.5), 21: (1.0, 0.5), 22: (1.0, 0.5),
}

# 启停配置
RAMP_UP_TIME = 3.0    # 缓启动时间(秒)
RAMP_DOWN_TIME = 3.0  # 缓停止时间(秒)
FEEDFORWARD_SCALE = 0.8  # 前馈补偿缩放(防止URDF质量偏差导致过补)
DEBUG_TABLE = True  # 是否显示调试表格

# 安全倒向配置: 停止时主动倒向此方向，避免断电后随机倒向四角冲击结构
SAFE_FALL_POSITIONS = {
    17: 0.0,   # waist_yaw: 保持中位
    18: 0.15,  # waist_roll: 微向右倒
    19: -0.20,   # waist_pitch: 微向后倒
}

# 安全模式 kp/kd (缓停止结束时的目标值)
# 与 DM_CAN_FD.py 中 JOINT_KP_KD_LIMITS_SAFE 保持一致
SAFE_KP_KD = {
    1: (5.0, 1.0), 2: (5.0, 1.0), 3: (5.0, 1.0), 4: (5.0, 1.0),
    5: (5.0, 1.0), 6: (5.0, 1.0), 7: (5.0, 1.0), 8: (2.0, 0.2),
    9: (5.0, 1.0), 10: (5.0, 1.0), 11: (5.0, 1.0), 12: (5.0, 1.0),
    13: (5.0, 1.0), 14: (5.0, 1.0), 15: (5.0, 1.0), 16: (2.0, 0.2),
    17: (40.0, 2.0), 18: (40.0, 2.0), 19: (40.0, 2.0),
    20: (1.0, 0.5), 21: (1.0, 0.5), 22: (1.0, 0.5),
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

RAMP_EXPONENT = 2.0

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

# 末端执行器: (link, mocap, limbs)
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target", ["left_arm"]),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target", ["right_arm"]),
    "waist": ("neck_yaw_link", "waist_target", ["waist"]),
}

# 碰撞对
COLLISION_PAIRS = [
    (["left_hand_collision"], ["torso_collision"]),
    (["right_hand_collision"], ["torso_collision"]),
    (["left_elbow_collision"], ["torso_collision"]),
    (["right_elbow_collision"], ["torso_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_elbow_collision"], ["right_elbow_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["head_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_elbow_collision"]),
    (["right_hand_collision"], ["left_elbow_collision"]),
]


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元数球面插值"""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        r = (1 - t) * q0 + t * q1
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        r = (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
    return r / np.linalg.norm(r)


def compute_lookat_quat(head_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """计算look-at四元数(MuJoCo wxyz格式)"""
    direction = target_pos - head_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction /= dist
    forward = np.array([1.0, 0.0, 0.0])
    dot = np.clip(np.dot(forward, direction), -1.0, 1.0)
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = np.cross(forward, direction)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def parse_args():
    parser = argparse.ArgumentParser(description="半身VR控制IK - SIM2REAL")
    parser.add_argument("--headless", action="store_true", default=True, help="无头模式(无GUI)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", default=False, help="禁用真机控制(仅仿真)")
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
    headless = args.headless
    enable_real = not args.no_real
    
    # 本机模式: 自动启动SDK服务端
    sdk_proc = None
    is_local = args.host in ("127.0.0.1", "localhost") and enable_real
    if is_local:
        sdk_proc = start_local_sdk()
        if sdk_proc is None:
            print("[SDK] 本机SDK启动失败，切换到仅仿真模式")
            enable_real = False
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算DOF索引和关节ID映射
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    # 所有可控DOF索引(排除floating base)
    all_dof_indices = []
    for limb in JOINT_GROUPS:
        all_dof_indices.extend(joint_idx[limb])
    all_dof_indices = sorted(set(all_dof_indices))
    
    # 构建关节名 -> (qpos索引, dof索引, SDK ID)映射
    joint_mapping = {}  # jname -> {'qpos': int, 'dof': int, 'sdk_id': int}
    for group, names in JOINT_GROUPS.items():
        for jname in names:
            jid = model.joint(jname).id
            qpos_idx = model.jnt_qposadr[jid]  # qpos索引用于读取关节位置
            dof_idx = model.jnt_dofadr[jid]    # dof索引用于读取扭矩
            sdk_id = JOINT_NAME_TO_SDK_ID.get(jname)
            if sdk_id is not None:
                joint_mapping[jname] = {'qpos': qpos_idx, 'dof': dof_idx, 'sdk_id': sdk_id}
    
    # 连接taks服务器
    robot = None
    left_gripper = None
    right_gripper = None
    if enable_real:
        try:
            taks.connect(args.host, cmd_port=args.port)
            robot = taks.register("Taks-T1-semibody")
            left_gripper = taks.register("Taks-T1-leftgripper")
            right_gripper = taks.register("Taks-T1-rightgripper")
            print(f"[TAKS] 已连接 {args.host}:{args.port}, 且夹爪已注册")
        except Exception as e:
            print(f"[TAKS] 连接失败: {e}, 仅仿真模式")
            enable_real = False
    
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap, _) in END_EFFECTORS.items()}
    ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
    neck_pitch_mid = model.body("neck_pitch_target").mocapid[0]
    left_hand_mid, right_hand_mid, waist_mid = mocap_ids["left_hand"], mocap_ids["right_hand"], mocap_ids["waist"]
    
    # 创建任务(固定cost，不动态调整)
    tasks = [
        mink.FrameTask("base_link", "body", position_cost=1e6, orientation_cost=1e6),
        mink.PostureTask(model, cost=1e-2),
    ]
    for name, (link, _, _) in END_EFFECTORS.items():
        cost = (0.0, 2.0) if name == "waist" else (2.0, 2.0)
        tasks.append(mink.FrameTask(link, "body", position_cost=cost[0], orientation_cost=cost[1]))
    neck_task = mink.FrameTask("neck_pitch_link", "body", position_cost=0.0, orientation_cost=1.0)
    tasks.append(neck_task)
    ee_tasks = {name: tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
    
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model),
        mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, gain=0.5, 
                                     minimum_distance_from_collisions=0.02, 
                                     collision_detection_distance=0.1)
    ]
    
    # VR接收器
    vr = VRReceiver()
    vr.start()
    
    # VR校准状态
    vr_calib = {"done": False, "left": np.zeros(3), "right": np.zeros(3), "head": np.zeros(3)}
    # 复位状态
    reset_state = {"active": False, "alpha": 0.0, "start_q": None, "start_pos": {}, "start_quat": {}}
    # 运行状态
    running = True
    shutdown_requested = False
    
    # 启动状态: 线性位置插值，kp/kd恒定
    enable_ramp_up = not args.no_ramp_up
    enable_ramp_down = not args.no_ramp_down
    ramp_up_time = args.ramp_up_time
    ramp_down_time = args.ramp_down_time
    ramp_state = {"active": enable_ramp_up, "start_time": None, "progress": 0.0 if enable_ramp_up else 1.0, "start_positions": {}}
    # 调试状态
    debug_state = {"pos_errors": {}, "console": Console() if DEBUG_TABLE else None}
    
    def do_calibrate():
        vr_data = vr.data
        if not vr_data.tracking_enabled:
            print("[VR] 未启用追踪，无法校准")
            return
        vr_calib["left"] = data.mocap_pos[left_hand_mid] - vr_data.left_hand.position
        vr_calib["right"] = data.mocap_pos[right_hand_mid] - vr_data.right_hand.position
        vr_calib["head"] = data.mocap_pos[waist_mid] - vr_data.head.position
        vr_calib["done"] = True
        vr.reset_smooth()
        print(f"[VR] 校准完成")
    
    def do_reset():
        reset_state["active"] = True
        reset_state["alpha"] = 0.0
        reset_state["start_q"] = cfg.q.copy()
        for name, mid in mocap_ids.items():
            reset_state["start_pos"][name] = data.mocap_pos[mid].copy()
            reset_state["start_quat"][name] = data.mocap_quat[mid].copy()
        vr.reset_smooth()
        print("[Reset] 复位开始...")
    
    def key_callback(keycode: int) -> None:
        if keycode == 259:  # BACKSPACE
            do_reset()
        elif keycode == 67:  # C
            do_calibrate()
    
    def send_to_real(q_arr, tau_arr, ramp_progress=1.0):
        """发送关节位置和前馈扭矩到真机
        ramp_progress: 启动进度[0,1]，用于非线性kp/kd渐变
        - ramp up: kp/kd 从0非线性(ease_out:越来越慢)增加到目标值
        - 正常运行: ramp_progress=1.0, 使用完整kp/kd
        """
        if not enable_real or robot is None:
            return
        mit_cmd = {}
        # 非线性kp/kd缩放因子 (ease_out: 开始快，结束慢)
        kp_kd_scale = ease_out(ramp_progress) if ramp_progress < 1.0 else 1.0
        for jname, info in joint_mapping.items():
            qpos_idx = info['qpos']
            dof_idx = info['dof']
            sdk_id = info['sdk_id']
            kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
            q_target = float(q_arr[qpos_idx]) if qpos_idx < len(q_arr) else 0.0
            tau_val = float(tau_arr[dof_idx]) if dof_idx < len(tau_arr) else 0.0
            # 启动阶段: kp/kd非线性渐变，位置直接发送目标位置
            kp_val = kp_target * kp_kd_scale
            kd_val = kd_target * kp_kd_scale
            # 前馈扭矩应用缩放(启动阶段也按进度缩放)
            mit_cmd[sdk_id] = {'q': q_target, 'dq': 0.0, 'tau': tau_val * FEEDFORWARD_SCALE * kp_kd_scale, 
                              'kp': kp_val, 'kd': kd_val}
            # 记录位置误差用于调试
            if DEBUG_TABLE:
                debug_state["pos_errors"][sdk_id] = {'target': q_target, 'cmd': q_target, 'kp': kp_val}
        if mit_cmd:
            robot.controlMIT(mit_cmd)
    
    def send_gripper(left_val: float, right_val: float):
        """发送夹爪控制命令(VR gripper值0-1映射到0-100百分比)"""
        if not enable_real:
            return
        left_percent = np.clip(left_val * 100.0, 0.0, 100.0)
        right_percent = np.clip(right_val * 100.0, 0.0, 100.0)
        if left_gripper is not None:
            kp, kd = SDK_JOINT_GAINS.get(16, (0.5, 0.05))
            left_gripper.controlMIT(percent=left_percent, kp=kp, kd=kd)
        if right_gripper is not None:
            kp, kd = SDK_JOINT_GAINS.get(8, (0.5, 0.05))
            right_gripper.controlMIT(percent=right_percent, kp=kp, kd=kd)
    
    # 初始化配置
    cfg.update_from_keyframe("home")
    tasks[0].set_target_from_configuration(cfg)
    tasks[1].set_target_from_configuration(cfg)
    for name, (link, mocap, _) in END_EFFECTORS.items():
        mink.move_mocap_to_frame(model, data, mocap, link, "body")
        ee_tasks[name].set_target_from_configuration(cfg)
    mink.move_mocap_to_frame(model, data, "neck_pitch_target", "neck_pitch_link", "body")
    neck_task.set_target_from_configuration(cfg)
    
    # 保存初始状态
    init_q = cfg.q.copy()
    init_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
    init_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
    prev_pos = {name: pos.copy() for name, pos in init_pos.items()}
    prev_quat = {name: quat.copy() for name, quat in init_quat.items()}
    
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.dt
    reset_duration = 1.5
    print_counter = 0
    
    mode_str = "有头" if not headless else "无头"
    real_str = "SIM2REAL" if enable_real else "仅仿真"
    print(f"[Info] 模式: {mode_str}, {real_str}")
    print("[Info] 键盘: C=校准, Backspace=复位 | VR手柄: B双击=校准, A双击=复位")
    
    def get_real_positions(timeout: float = 2.0):
        """获取真机当前位置，等待有效数据
        timeout: 等待超时时间(秒)，超时后返回0
        """
        if not enable_real or robot is None:
            return {info['sdk_id']: 0.0 for info in joint_mapping.values()}
        
        # 等待有效数据
        start = time.time()
        while time.time() - start < timeout:
            real_pos = robot.GetPosition()
            if real_pos is not None and len(real_pos) > 0:
                return {sdk_id: real_pos.get(sdk_id, 0.0) for sdk_id in [info['sdk_id'] for info in joint_mapping.values()]}
            time.sleep(0.05)  # 20Hz轮询
        
        print(f"[警告] 获取真机位置超时({timeout}s)，使用0作为起始位置")
        return {info['sdk_id']: 0.0 for info in joint_mapping.values()}
    
    def ramp_down():
        """安全停止: kp/kd非线性(ease_in:越来越快)从目标值降低到安全值，然后失能
        使用 ease_in 函数: 开始慢，结束快
        """
        if not enable_real or robot is None:
            return
        if not enable_ramp_down:
            # 直接失能
            mit_cmd = {info['sdk_id']: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for info in joint_mapping.values()}
            robot.controlMIT(mit_cmd)
            print("[Ramp Down] 缓停止已禁用，直接失能")
            return
        
        print(f"[Ramp Down] 非线性降低kp/kd到安全值 ({ramp_down_time}s)...")
        start = time.time()
        # 从真机获取当前位置作为起点
        start_positions = get_real_positions()
        
        while time.time() - start < ramp_down_time:
            elapsed = time.time() - start
            t = elapsed / ramp_down_time  # 线性进度 [0,1]
            # 非线性kp/kd缩放因子 (ease_in: 开始慢，结束快)
            # t=0时 scale=1(目标kp/kd), t=1时 scale=0(安全kp/kd)
            kp_kd_scale = 1.0 - ease_in(t)
            mit_cmd = {}
            for jname, info in joint_mapping.items():
                sdk_id = info['sdk_id']
                kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
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
        mit_cmd = {info['sdk_id']: {'q': start_positions.get(info['sdk_id'], 0.0), 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} for info in joint_mapping.values()}
        robot.controlMIT(mit_cmd)
        print("[Ramp Down] 已降低到安全kp/kd并失能")
    
    def signal_handler(signum, frame):
        """信号处理: 确保先缓关闭再断开连接"""
        nonlocal running, shutdown_requested
        if shutdown_requested:
            print("\n[强制退出]")
            if enable_real:
                taks.disconnect()
            sys.exit(1)
        print("\n[收到退出信号] 开始安全关闭...")
        shutdown_requested = True
        running = False
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def print_debug_table():
        """打印调试表格(临时调试用，未来删除)"""
        if not DEBUG_TABLE or not debug_state["pos_errors"]:
            return
        table = Table(title="关节调试信息", show_header=True)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("目标位置", style="green", width=10)
        table.add_column("发送位置", style="yellow", width=10)
        table.add_column("差值", style="red", width=10)
        table.add_column("KP", style="magenta", width=8)
        for sdk_id in sorted(debug_state["pos_errors"].keys()):
            info = debug_state["pos_errors"][sdk_id]
            diff = info['target'] - info['cmd']
            table.add_row(
                str(sdk_id),
                f"{info['target']:.4f}",
                f"{info['cmd']:.4f}",
                f"{diff:.4f}",
                f"{info['kp']:.1f}"
            )
        debug_state["console"].clear()
        debug_state["console"].print(table)
    
    def control_loop(viewer=None):
        nonlocal print_counter, running
        
        # 初始化启动: 从真机获取当前位置作为起点
        if enable_ramp_up:
            ramp_state["start_time"] = time.time()
            ramp_state["active"] = True
            ramp_state["start_positions"] = get_real_positions()
            print(f"[Ramp Up] 线性启动 ({ramp_up_time}s)...")
        else:
            ramp_state["active"] = False
            ramp_state["progress"] = 1.0
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
                    ramp_state["progress"] = elapsed / ramp_up_time  # 线性进度
            
            vr_data = vr.data
            
            # VR按键事件
            if vr_data.button_events.right_b:
                do_calibrate()
            if vr_data.button_events.right_a:
                do_reset()
            
            # VR数据更新mocap
            if vr_calib["done"] and vr_data.tracking_enabled:
                data.mocap_pos[left_hand_mid] = vr_data.left_hand.position + vr_calib["left"]
                data.mocap_quat[left_hand_mid] = vr_data.left_hand.quaternion
                data.mocap_pos[right_hand_mid] = vr_data.right_hand.position + vr_calib["right"]
                data.mocap_quat[right_hand_mid] = vr_data.right_hand.quaternion
                data.mocap_pos[waist_mid] = vr_data.head.position + vr_calib["head"]
                data.mocap_quat[waist_mid] = vr_data.head.quaternion
            
            # look-at目标
            hands_center = (data.mocap_pos[left_hand_mid] + data.mocap_pos[right_hand_mid]) / 2.0
            head_pos = data.xpos[model.body("neck_pitch_link").id]
            data.mocap_quat[neck_pitch_mid] = compute_lookat_quat(head_pos, hands_center)
            neck_task.set_target(mink.SE3.from_mocap_id(data, neck_pitch_mid))
            
            # 复位处理
            if reset_state["active"]:
                reset_state["alpha"] += dt / reset_duration
                alpha = min(1.0, reset_state["alpha"])
                for name, mid in mocap_ids.items():
                    data.mocap_pos[mid] = (1 - alpha) * reset_state["start_pos"][name] + alpha * init_pos[name]
                    data.mocap_quat[mid] = slerp(reset_state["start_quat"][name], init_quat[name], alpha)
                    prev_pos[name], prev_quat[name] = data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy()
                cfg.update(reset_state["start_q"] * (1 - alpha) + init_q * alpha)
                for name in END_EFFECTORS:
                    ee_tasks[name].set_target_from_configuration(cfg)
                mask = np.zeros(model.nv, dtype=bool)
                for idx in joint_idx["neck"]:
                    mask[idx] = True
                tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.5, limits=limits)
                vel[~mask] = 0.0
                cfg.integrate_inplace(vel, dt)
                if alpha >= 1.0:
                    reset_state["active"] = False
                    tasks[1].cost[:] = 1e-2
                    print("[Reset] 复位完成")
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = data.qfrc_bias[:]
                send_to_real(cfg.q, data.qfrc_bias)
                if viewer:
                    mujoco.mj_camlight(model, data)
                    viewer.sync()
                rate.sleep()
                continue
            
            # 更新所有末端任务目标
            for name, mid in mocap_ids.items():
                ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
            
            # 检测活动肢体
            active_dofs = set(joint_idx["neck"])
            for name, mid in mocap_ids.items():
                pos_diff = data.mocap_pos[mid] - prev_pos[name]
                quat_diff = np.abs(data.mocap_quat[mid] - prev_quat[name])
                if name == "waist":
                    if np.max(quat_diff) > 0.005:
                        for limb in ee_limbs[name]:
                            active_dofs.update(joint_idx[limb])
                else:
                    if np.dot(pos_diff, pos_diff) > 1e-7 or np.max(quat_diff) > 0.005:
                        for limb in ee_limbs[name]:
                            active_dofs.update(joint_idx[limb])
                prev_pos[name], prev_quat[name] = data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy()
            
            # 构建冻结约束
            frozen_dofs = [i for i in all_dof_indices if i not in active_dofs]
            constraints = []
            if frozen_dofs:
                constraints.append(mink.DofFreezingTask(model, dof_indices=frozen_dofs))
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=1e-1, limits=limits, constraints=constraints)
            cfg.integrate_inplace(vel, dt)
            
            # 前馈扭矩补偿
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            # 发送到真机
            send_to_real(cfg.q, data.qfrc_bias, ramp_state["progress"])
            send_gripper(vr_data.left_hand.gripper, vr_data.right_hand.gripper)
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                if DEBUG_TABLE:
                    print_debug_table()
                else:
                    print(f"[VR] Tracking={'ON' if vr_data.tracking_enabled else 'OFF'}, Calibrated={'YES' if vr_calib['done'] else 'NO'}, Real={'ON' if enable_real else 'OFF'}")
            
            if viewer:
                mujoco.mj_camlight(model, data)
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
        vr.stop()
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