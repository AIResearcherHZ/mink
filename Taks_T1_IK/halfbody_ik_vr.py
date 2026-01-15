"""半身VR控制IK - SIM2REAL版 (重构版)

解决多路IK跳变: 使用DofFreezingTask作为equality constraint冻结非活动关节
支持taks SDK实现SIM2REAL控制
"""

import sys
import argparse
import signal
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from vr_interface import VRReceiver
from taks_sdk import taks

# ==================== 配置常量 ====================

_XML = Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"
TAKS_SEND_RATE = 30  # taks发送频率(Hz), None不限制

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
SDK_ID_TO_NAME = {v: k for k, v in JOINT_NAME_TO_SDK_ID.items()}

# 关节增益配置
SDK_JOINT_GAINS = {
    1: (5, 1), 2: (5, 1), 3: (5, 1), 4: (5, 1),
    5: (2.5, 1), 6: (2.5, 1), 7: (2.5, 1), 8: (1.5, 0.1),
    9: (5, 1), 10: (5, 1), 11: (5, 1), 12: (5, 1),
    13: (2.5, 1), 14: (2.5, 1), 15: (2.5, 1), 16: (1.5, 0.1),
    17: (150, 1), 18: (150, 1), 19: (150, 1),
    20: (1.5, 0.1), 21: (1.5, 0.1), 22: (1.5, 0.1),
}

SAFE_KP_KD = {
    1: (5, 1), 2: (5, 1), 3: (5, 1), 4: (5, 1),
    5: (2.5, 1), 6: (2.5, 1), 7: (2.5, 1), 8: (1.5, 0.1),
    9: (5, 1), 10: (5, 1), 11: (5, 1), 12: (5, 1),
    13: (2.5, 1), 14: (2.5, 1), 15: (2.5, 1), 16: (1.5, 0.1),
    17: (25, 1), 18: (25, 1), 19: (25, 1),
    20: (1.5, 0.1), 21: (1.5, 0.1), 22: (1.5, 0.1),
}

# 启停配置
RAMP_UP_TIME = 5.0
RAMP_DOWN_TIME = 5.0
FEEDFORWARD_SCALE = 0.8

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

RAMP_EXPONENT = 1.05

# 安全倒向配置: 停止时主动倒向此方向
SAFE_FALL_POSITIONS = {
    4: 1.2,     # right_elbow_joint
    12: 1.2,    # left_elbow_joint
    17: 0.0,    # waist_yaw_joint
    18: 0.52,    # waist_roll_joint
    19: -0.45,  # waist_pitch_joint
}

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


# ==================== 工具函数 ====================

def ease_out(t: float, exp: float = RAMP_EXPONENT) -> float:
    """缓出函数: 开始快，结束慢"""
    return 1.0 - pow(1.0 - t, exp)

def ease_in(t: float, exp: float = RAMP_EXPONENT) -> float:
    """缓入函数: 开始慢，结束快"""
    return pow(t, exp)

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


# ==================== 状态数据类 ====================

@dataclass
class VRCalibState:
    """VR校准状态"""
    done: bool = False
    left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right: np.ndarray = field(default_factory=lambda: np.zeros(3))
    head: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class ResetState:
    """复位状态"""
    active: bool = False
    alpha: float = 0.0
    start_q: Optional[np.ndarray] = None
    start_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    start_quat: Dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class RampState:
    """启动渐变状态"""
    active: bool = True
    start_time: Optional[float] = None
    progress: float = 0.0
    start_positions: Dict[int, float] = field(default_factory=dict)


# ==================== 控制器类 ====================

class HalfBodyIKController:
    """半身IK控制器"""
    
    def __init__(self, host: str = "192.168.5.4", port: int = 5555,
                 enable_real: bool = True, headless: bool = False,
                 ramp_up_time: float = RAMP_UP_TIME, ramp_down_time: float = RAMP_DOWN_TIME,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        # 配置
        self.host = host
        self.port = port
        self.enable_real = enable_real
        self.headless = headless
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        
        # 运行状态
        self.running = True
        self.shutdown_requested = False
        
        # 设备句柄
        self.robot = None
        self.left_gripper = None
        self.right_gripper = None
        self.sdk_proc = None
        
        # 状态对象
        self.vr_calib = VRCalibState()
        self.reset_state = ResetState()
        self.ramp_state = RampState(active=enable_ramp_up, progress=0.0 if enable_ramp_up else 1.0)
        
        # 统计
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.send_count = 0
        self.send_fps_start = time.time()
        self.current_send_fps = 0.0
        self.last_taks_send_time = 0.0
        self.last_mit_cmd = {}
        
        # 初始化
        self._init_mujoco()
        self._init_vr()
        self._init_real()
        self._init_tasks()
        self._save_init_state()
        
        # 控制参数
        self.rate = RateLimiter(frequency=200.0, warn=False)
        self.dt = self.rate.dt
        self.reset_duration = 1.5
        self.console = Console()
    
    def _init_mujoco(self):
        """初始化MuJoCo模型"""
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.cfg = mink.Configuration(self.model)
        self.model, self.data = self.cfg.model, self.cfg.data
        
        # 预计算DOF索引
        self.joint_idx = {k: [self.model.jnt_dofadr[self.model.joint(j).id] for j in v] 
                          for k, v in JOINT_GROUPS.items()}
        self.all_dof_indices = sorted(set(sum(self.joint_idx.values(), [])))
        
        # 构建关节映射
        self.joint_mapping = {}
        for group, names in JOINT_GROUPS.items():
            for jname in names:
                jid = self.model.joint(jname).id
                sdk_id = JOINT_NAME_TO_SDK_ID.get(jname)
                if sdk_id is not None:
                    self.joint_mapping[jname] = {
                        'qpos': self.model.jnt_qposadr[jid],
                        'dof': self.model.jnt_dofadr[jid],
                        'sdk_id': sdk_id
                    }
        
        # mocap ID
        self.mocap_ids = {name: self.model.body(mocap).mocapid[0] 
                          for name, (_, mocap, _) in END_EFFECTORS.items()}
        self.ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
        self.neck_pitch_mid = self.model.body("neck_pitch_target").mocapid[0]
    
    def _init_vr(self):
        """初始化VR接收器"""
        self.vr = VRReceiver()
        self.vr.start()
    
    def _init_real(self):
        """初始化真机连接"""
        if not self.enable_real:
            return
        
        # 本机模式自动启动SDK
        is_local = self.host in ("127.0.0.1", "localhost")
        if is_local:
            self.sdk_proc = self._start_local_sdk()
            if self.sdk_proc is None:
                print("[SDK] 本机SDK启动失败，切换到仅仿真模式")
                self.enable_real = False
                return
        
        try:
            taks.connect(self.host, cmd_port=self.port)
            self.robot = taks.register("Taks-T1-semibody")
            print(f"[TAKS] 已注册半身设备")
            time.sleep(4.0)
            self.left_gripper = taks.register("Taks-T1-leftgripper")
            print(f"[TAKS] 已注册左gripper")
            time.sleep(1.0)
            self.right_gripper = taks.register("Taks-T1-rightgripper")
            print(f"[TAKS] 已注册右gripper")
            time.sleep(1.0)
            print(f"[TAKS] 已连接 {self.host}:{self.port}")
        except Exception as e:
            print(f"[TAKS] 连接失败: {e}, 仅仿真模式")
            self.enable_real = False
    
    def _start_local_sdk(self) -> Optional[subprocess.Popen]:
        """启动本机SDK服务端"""
        sdk_path = Path(__file__).parent / "taks_sdk" / "SDK_MF.py"
        if not sdk_path.exists():
            print(f"[SDK] 错误: SDK_MF.py 不存在: {sdk_path}")
            return None
        print(f"[SDK] 本机模式: 启动 SDK 服务端...")
        proc = subprocess.Popen([sys.executable, str(sdk_path)], 
                                stdout=None, stderr=None, cwd=str(sdk_path.parent))
        time.sleep(3.0)
        if proc.poll() is not None:
            print(f"[SDK] 错误: SDK 服务端启动失败")
            return None
        print(f"[SDK] SDK 服务端已启动 (PID: {proc.pid})")
        return proc
    
    def _init_tasks(self):
        """初始化IK任务"""
        self.tasks = [
            mink.FrameTask("base_link", "body", position_cost=1e6, orientation_cost=1e6),
            mink.PostureTask(self.model, cost=1e-2),
        ]
        for name, (link, _, _) in END_EFFECTORS.items():
            cost = (0.0, 2.0) if name == "waist" else (2.0, 2.0)
            self.tasks.append(mink.FrameTask(link, "body", position_cost=cost[0], orientation_cost=cost[1]))
        self.neck_task = mink.FrameTask("neck_pitch_link", "body", position_cost=0.0, orientation_cost=1.0)
        self.tasks.append(self.neck_task)
        self.ee_tasks = {name: self.tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
        
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model),
            mink.CollisionAvoidanceLimit(self.model, COLLISION_PAIRS, gain=0.5,
                                         minimum_distance_from_collisions=0.02,
                                         collision_detection_distance=0.1)
        ]
        
        # 初始化配置
        self.cfg.update_from_keyframe("home")
        self.tasks[0].set_target_from_configuration(self.cfg)
        self.tasks[1].set_target_from_configuration(self.cfg)
        for name, (link, mocap, _) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(self.model, self.data, mocap, link, "body")
            self.ee_tasks[name].set_target_from_configuration(self.cfg)
        mink.move_mocap_to_frame(self.model, self.data, "neck_pitch_target", "neck_pitch_link", "body")
        self.neck_task.set_target_from_configuration(self.cfg)
    
    def _save_init_state(self):
        """保存初始状态"""
        self.init_q = self.cfg.q.copy()
        self.init_pos = {name: self.data.mocap_pos[mid].copy() for name, mid in self.mocap_ids.items()}
        self.init_quat = {name: self.data.mocap_quat[mid].copy() for name, mid in self.mocap_ids.items()}
        self.prev_pos = {name: pos.copy() for name, pos in self.init_pos.items()}
        self.prev_quat = {name: quat.copy() for name, quat in self.init_quat.items()}
    
    # ==================== 公共方法 ====================
    
    def calibrate(self):
        """VR校准"""
        vr_data = self.vr.data
        if not vr_data.tracking_enabled:
            print("[VR] 未启用追踪，无法校准")
            return
        left_mid = self.mocap_ids["left_hand"]
        right_mid = self.mocap_ids["right_hand"]
        waist_mid = self.mocap_ids["waist"]
        self.vr_calib.left = self.data.mocap_pos[left_mid] - vr_data.left_hand.position
        self.vr_calib.right = self.data.mocap_pos[right_mid] - vr_data.right_hand.position
        self.vr_calib.head = self.data.mocap_pos[waist_mid] - vr_data.head.position
        self.vr_calib.done = True
        self.vr.reset_smooth()
        print(f"[VR] 校准完成")
    
    def reset(self):
        """开始复位"""
        self.reset_state.active = True
        self.reset_state.alpha = 0.0
        self.reset_state.start_q = self.cfg.q.copy()
        for name, mid in self.mocap_ids.items():
            self.reset_state.start_pos[name] = self.data.mocap_pos[mid].copy()
            self.reset_state.start_quat[name] = self.data.mocap_quat[mid].copy()
        self.vr.reset_smooth()
        print("[Reset] 复位开始...")
    
    def get_real_positions(self, timeout: float = 2.0) -> Dict[int, float]:
        """获取真机当前位置"""
        if not self.enable_real or self.robot is None:
            return {info['sdk_id']: 0.0 for info in self.joint_mapping.values()}
        start = time.time()
        while time.time() - start < timeout:
            real_pos = self.robot.GetPosition()
            if real_pos is not None and len(real_pos) > 0:
                return {sdk_id: real_pos.get(sdk_id, 0.0) 
                        for sdk_id in [info['sdk_id'] for info in self.joint_mapping.values()]}
            time.sleep(0.05)
        print(f"[警告] 获取真机位置超时({timeout}s)")
        return {info['sdk_id']: 0.0 for info in self.joint_mapping.values()}
    
    def build_mit_cmd(self, q_arr: np.ndarray, tau_arr: np.ndarray, 
                      ramp_progress: float = 1.0) -> Dict[int, Dict[str, float]]:
        """构建MIT命令字典"""
        mit_cmd = {}
        kp_kd_scale = ease_out(ramp_progress) if ramp_progress < 1.0 else 1.0
        for jname, info in self.joint_mapping.items():
            sdk_id = info['sdk_id']
            kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
            q_target = float(q_arr[info['qpos']]) if info['qpos'] < len(q_arr) else 0.0
            tau_val = float(tau_arr[info['dof']]) if info['dof'] < len(tau_arr) else 0.0
            mit_cmd[sdk_id] = {
                'q': q_target, 'dq': 0.0,
                'tau': tau_val * FEEDFORWARD_SCALE * kp_kd_scale,
                'kp': kp_target * kp_kd_scale,
                'kd': kd_target * kp_kd_scale
            }
        return mit_cmd
    
    def send_to_real(self, mit_cmd: Dict[int, Dict[str, float]]) -> bool:
        """发送MIT命令到真机"""
        if not self.enable_real or self.robot is None or not mit_cmd:
            return False
        if TAKS_SEND_RATE:
            now = time.time()
            if now - self.last_taks_send_time < 1.0 / TAKS_SEND_RATE:
                return False
            self.last_taks_send_time = now
        self.robot.controlMIT(mit_cmd)
        self.send_count += 1
        return True
    
    def send_gripper(self, left_val: float, right_val: float):
        """发送夹爪控制"""
        if not self.enable_real:
            return
        left_percent = np.clip(left_val * 100.0, 0.0, 100.0)
        right_percent = np.clip(right_val * 100.0, 0.0, 100.0)
        if self.left_gripper is not None:
            kp, kd = SDK_JOINT_GAINS.get(16, (0.5, 0.05))
            self.left_gripper.controlMIT(percent=left_percent, kp=kp, kd=kd)
        if self.right_gripper is not None:
            kp, kd = SDK_JOINT_GAINS.get(8, (0.5, 0.05))
            self.right_gripper.controlMIT(percent=right_percent, kp=kp, kd=kd)
    
    def ramp_up(self):
        """缓启动"""
        if not self.enable_ramp_up:
            self.ramp_state.active = False
            self.ramp_state.progress = 1.0
            print("[Ramp Up] 缓启动已禁用")
            return
        self.ramp_state.start_time = time.time()
        self.ramp_state.active = True
        self.ramp_state.start_positions = self.get_real_positions()
        print(f"[Ramp Up] 线性启动 ({self.ramp_up_time}s)...")
    
    def ramp_down(self):
        """缓停止"""
        if not self.enable_real or self.robot is None:
            return
        if not self.enable_ramp_down:
            mit_cmd = {info['sdk_id']: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} 
                       for info in self.joint_mapping.values()}
            self.robot.controlMIT(mit_cmd)
            print("[Ramp Down] 缓停止已禁用，直接失能")
            return
        
        print(f"[Ramp Down] 非线性降低kp/kd到安全值 ({self.ramp_down_time}s)...")
        start = time.time()
        start_positions = self.get_real_positions()
        
        while time.time() - start < self.ramp_down_time:
            elapsed = time.time() - start
            t = elapsed / self.ramp_down_time
            kp_kd_scale = 1.0 - ease_in(t)
            mit_cmd = {}
            for jname, info in self.joint_mapping.items():
                sdk_id = info['sdk_id']
                kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
                kp_safe, kd_safe = SAFE_KP_KD.get(sdk_id, (5.0, 1.0))
                kp_val = kp_safe + (kp_target - kp_safe) * kp_kd_scale
                kd_val = kd_safe + (kd_target - kd_safe) * kp_kd_scale
                # 使用安全倒向位置(如果配置了)
                start_q = start_positions.get(sdk_id, 0.0)
                target_q = SAFE_FALL_POSITIONS.get(sdk_id, start_q)
                q_val = start_q + (target_q - start_q) * t
                mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp_val, 'kd': kd_val}
            self.robot.controlMIT(mit_cmd)
            time.sleep(0.001)
        
        # 失能(使用最终的安全倒向位置)
        mit_cmd = {}
        for info in self.joint_mapping.values():
            sdk_id = info['sdk_id']
            start_q = start_positions.get(sdk_id, 0.0)
            target_q = SAFE_FALL_POSITIONS.get(sdk_id, start_q)
            mit_cmd[sdk_id] = {'q': target_q, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0}
        self.robot.controlMIT(mit_cmd)
        print("[Ramp Down] 已降低到安全kp/kd并失能")
    
    def build_status_table(self) -> Table:
        """构建rich状态表格"""
        table = Table(title="TAKS MIT控制状态", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=3)
        table.add_column("关节名", width=28)
        table.add_column("q(rad)", justify="right", width=8)
        table.add_column("tau(Nm)", justify="right", width=8)
        table.add_column("kp", justify="right", width=6)
        table.add_column("kd", justify="right", width=6)
        
        for sdk_id in sorted(self.last_mit_cmd.keys()):
            cmd = self.last_mit_cmd[sdk_id]
            jname = SDK_ID_TO_NAME.get(sdk_id, f"joint_{sdk_id}")
            table.add_row(str(sdk_id), jname, f"{cmd['q']:.3f}", f"{cmd['tau']:.3f}",
                          f"{cmd['kp']:.2f}", f"{cmd['kd']:.2f}")
        
        table.add_section()
        rate_str = f"{TAKS_SEND_RATE:.0f}Hz" if TAKS_SEND_RATE else "无限制"
        vr_data = self.vr.data
        status1 = f"仿真FPS: {self.current_fps:.1f} | 实际发送: {self.current_send_fps:.1f}Hz | 目标频率: {rate_str}"
        status2 = f"VR: {'ON' if vr_data.tracking_enabled else 'OFF'} | 校准: {'YES' if self.vr_calib.done else 'NO'} | 真机: {'ON' if self.enable_real else 'OFF'}"
        table.add_row("", status1, "", "", "", "")
        table.add_row("", status2, "", "", "", "")
        return table
    
    def step(self) -> Dict[int, Dict[str, float]]:
        """执行一步控制，返回MIT命令"""
        # 更新启动进度
        if self.ramp_state.active:
            elapsed = time.time() - self.ramp_state.start_time
            if elapsed >= self.ramp_up_time:
                self.ramp_state.active = False
                self.ramp_state.progress = 1.0
                print("[Ramp Up] 启动完成")
            else:
                self.ramp_state.progress = elapsed / self.ramp_up_time
        
        vr_data = self.vr.data
        left_mid = self.mocap_ids["left_hand"]
        right_mid = self.mocap_ids["right_hand"]
        waist_mid = self.mocap_ids["waist"]
        
        # VR按键事件
        if vr_data.button_events.right_b:
            self.calibrate()
        if vr_data.button_events.right_a:
            self.reset()
        
        # VR数据更新mocap
        if self.vr_calib.done and vr_data.tracking_enabled:
            self.data.mocap_pos[left_mid] = vr_data.left_hand.position + self.vr_calib.left
            self.data.mocap_quat[left_mid] = vr_data.left_hand.quaternion
            self.data.mocap_pos[right_mid] = vr_data.right_hand.position + self.vr_calib.right
            self.data.mocap_quat[right_mid] = vr_data.right_hand.quaternion
            self.data.mocap_pos[waist_mid] = vr_data.head.position + self.vr_calib.head
            self.data.mocap_quat[waist_mid] = vr_data.head.quaternion
        
        # look-at目标
        hands_center = (self.data.mocap_pos[left_mid] + self.data.mocap_pos[right_mid]) / 2.0
        head_pos = self.data.xpos[self.model.body("neck_pitch_link").id]
        self.data.mocap_quat[self.neck_pitch_mid] = compute_lookat_quat(head_pos, hands_center)
        self.neck_task.set_target(mink.SE3.from_mocap_id(self.data, self.neck_pitch_mid))
        
        # 复位处理
        if self.reset_state.active:
            return self._step_reset()
        
        # 更新末端任务目标
        for name, mid in self.mocap_ids.items():
            self.ee_tasks[name].set_target(mink.SE3.from_mocap_id(self.data, mid))
        
        # 检测活动肢体
        active_dofs = set(self.joint_idx["neck"])
        for name, mid in self.mocap_ids.items():
            pos_diff = self.data.mocap_pos[mid] - self.prev_pos[name]
            quat_diff = np.abs(self.data.mocap_quat[mid] - self.prev_quat[name])
            if name == "waist":
                if np.max(quat_diff) > 0.005:
                    for limb in self.ee_limbs[name]:
                        active_dofs.update(self.joint_idx[limb])
            else:
                if np.dot(pos_diff, pos_diff) > 1e-7 or np.max(quat_diff) > 0.005:
                    for limb in self.ee_limbs[name]:
                        active_dofs.update(self.joint_idx[limb])
            self.prev_pos[name] = self.data.mocap_pos[mid].copy()
            self.prev_quat[name] = self.data.mocap_quat[mid].copy()
        
        # 构建冻结约束
        frozen_dofs = [i for i in self.all_dof_indices if i not in active_dofs]
        constraints = [mink.DofFreezingTask(self.model, dof_indices=frozen_dofs)] if frozen_dofs else []
        
        # 求解IK
        vel = mink.solve_ik(self.cfg, self.tasks, self.dt, "daqp", damping=1e-1, 
                            limits=self.limits, constraints=constraints)
        self.cfg.integrate_inplace(vel, self.dt)
        
        # 前馈扭矩
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
        
        # 构建MIT命令
        mit_cmd = self.build_mit_cmd(self.cfg.q, self.data.qfrc_bias, self.ramp_state.progress)
        self.last_mit_cmd = mit_cmd
        
        # 发送
        self.send_to_real(mit_cmd)
        self.send_gripper(vr_data.left_hand.gripper, vr_data.right_hand.gripper)
        
        return mit_cmd
    
    def _step_reset(self) -> Dict[int, Dict[str, float]]:
        """复位步骤"""
        self.reset_state.alpha += self.dt / self.reset_duration
        alpha = min(1.0, self.reset_state.alpha)
        
        for name, mid in self.mocap_ids.items():
            self.data.mocap_pos[mid] = (1 - alpha) * self.reset_state.start_pos[name] + alpha * self.init_pos[name]
            self.data.mocap_quat[mid] = slerp(self.reset_state.start_quat[name], self.init_quat[name], alpha)
            self.prev_pos[name] = self.data.mocap_pos[mid].copy()
            self.prev_quat[name] = self.data.mocap_quat[mid].copy()
        
        self.cfg.update(self.reset_state.start_q * (1 - alpha) + self.init_q * alpha)
        for name in END_EFFECTORS:
            self.ee_tasks[name].set_target_from_configuration(self.cfg)
        
        mask = np.zeros(self.model.nv, dtype=bool)
        for idx in self.joint_idx["neck"]:
            mask[idx] = True
        self.tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
        
        vel = mink.solve_ik(self.cfg, self.tasks, self.dt, "daqp", damping=0.5, limits=self.limits)
        vel[~mask] = 0.0
        self.cfg.integrate_inplace(vel, self.dt)
        
        if alpha >= 1.0:
            self.reset_state.active = False
            self.tasks[1].cost[:] = 1e-2
            print("[Reset] 复位完成")
        
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
        mit_cmd = self.build_mit_cmd(self.cfg.q, self.data.qfrc_bias)
        self.send_to_real(mit_cmd)
        self.last_mit_cmd = mit_cmd
        return mit_cmd
    
    def update_stats(self):
        """更新帧率统计"""
        self.frame_count += 1
        now = time.time()
        if now - self.fps_start_time >= 1.0:
            self.current_fps = self.frame_count / (now - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = now
        if now - self.send_fps_start >= 1.0:
            self.current_send_fps = self.send_count / (now - self.send_fps_start)
            self.send_count = 0
            self.send_fps_start = now
    
    def key_callback(self, keycode: int):
        """键盘回调"""
        if keycode == 259:  # BACKSPACE
            self.reset()
        elif keycode == 67:  # C
            self.calibrate()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        self.vr.stop()
        if self.enable_real:
            self.ramp_down()
            taks.disconnect()
            print("[TAKS] 已断开")
        if self.sdk_proc is not None:
            print("[SDK] 关闭本机SDK服务端...")
            self.sdk_proc.terminate()
            try:
                self.sdk_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.sdk_proc.kill()
            print("[SDK] SDK服务端已关闭")
    
    def run(self):
        """主运行循环"""
        # 信号处理
        def signal_handler(signum, frame):
            if self.shutdown_requested:
                print("\n[强制退出]")
                if self.enable_real:
                    taks.disconnect()
                sys.exit(1)
            print("\n[收到退出信号] 开始安全关闭...")
            self.shutdown_requested = True
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        mode_str = "有头" if not self.headless else "无头"
        real_str = "SIM2REAL" if self.enable_real else "仅仿真"
        rate_str = f"{TAKS_SEND_RATE:.0f}Hz" if TAKS_SEND_RATE else "无限制"
        print(f"[Info] 模式: {mode_str}, {real_str}, taks发送频率: {rate_str}")
        print("[Info] 键盘: C=校准, Backspace=复位 | VR手柄: B双击=校准, A双击=复位")
        
        try:
            if self.headless:
                print("[Info] 无头模式运行中，Ctrl+C退出")
                self._control_loop(viewer=None)
            else:
                with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, 
                                                   show_right_ui=False, 
                                                   key_callback=self.key_callback) as viewer:
                    mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
                    self._control_loop(viewer=viewer)
        except KeyboardInterrupt:
            print("\n[Info] 用户中断")
        finally:
            self.cleanup()
    
    def _control_loop(self, viewer=None):
        """控制循环"""
        self.ramp_up()
        print_interval = 1.0
        last_print_time = time.time()
        
        while self.running:
            if viewer is not None and not viewer.is_running():
                break
            
            self.step()
            self.update_stats()
            
            now = time.time()
            if now - last_print_time >= print_interval and self.last_mit_cmd:
                last_print_time = now
                self.console.clear()
                self.console.print(self.build_status_table())
            
            if viewer:
                mujoco.mj_camlight(self.model, self.data)
                viewer.sync()
            self.rate.sleep()


# ==================== 命令行入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="半身VR控制IK - SIM2REAL")
    parser.add_argument("--headless", action="store_true", default=False, help="无头模式")
    parser.add_argument("--host", type=str, default="192.168.5.4", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", default=False, help="禁用真机控制")
    parser.add_argument("--no-ramp-up", action="store_true", default=False, help="禁用缓启动")
    parser.add_argument("--no-ramp-down", action="store_true", default=False, help="禁用缓停止")
    parser.add_argument("--ramp-up-time", type=float, default=RAMP_UP_TIME, help="缓启动时间(秒)")
    parser.add_argument("--ramp-down-time", type=float, default=RAMP_DOWN_TIME, help="缓停止时间(秒)")
    return parser.parse_args()


def main():
    args = parse_args()
    controller = HalfBodyIKController(
        host=args.host,
        port=args.port,
        enable_real=not args.no_real,
        headless=args.headless,
        ramp_up_time=args.ramp_up_time,
        ramp_down_time=args.ramp_down_time,
        enable_ramp_up=not args.no_ramp_up,
        enable_ramp_down=not args.no_ramp_down,
    )
    controller.run()


if __name__ == "__main__":
    main()