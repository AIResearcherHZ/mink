"""
腰部低权重：大部分时间直立，只有yaw跟随双手，超距时才位置补偿
支持taks SDK实现SIM2REAL控制
"""

import sys
import argparse
import signal
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from vr_interface import VRReceiver
from taks_sdk import taks

# ==================== 配置 ====================

_XML = Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"
TAKS_SEND_RATE = 30
RAMP_UP_TIME = 5.0
RAMP_DOWN_TIME = 5.0
FEEDFORWARD_SCALE = 0.75

JOINT_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

JOINT_NAME_TO_SDK_ID = {
    "right_shoulder_pitch_joint": 1, "right_shoulder_roll_joint": 2, "right_shoulder_yaw_joint": 3,
    "right_elbow_joint": 4, "right_wrist_roll_joint": 5, "right_wrist_yaw_joint": 6, "right_wrist_pitch_joint": 7,
    "left_shoulder_pitch_joint": 9, "left_shoulder_roll_joint": 10, "left_shoulder_yaw_joint": 11,
    "left_elbow_joint": 12, "left_wrist_roll_joint": 13, "left_wrist_yaw_joint": 14, "left_wrist_pitch_joint": 15,
    "waist_yaw_joint": 17, "waist_roll_joint": 18, "waist_pitch_joint": 19,
    "neck_yaw_joint": 20, "neck_roll_joint": 21, "neck_pitch_joint": 22,
}
SDK_ID_TO_NAME = {v: k for k, v in JOINT_NAME_TO_SDK_ID.items()}

SDK_JOINT_GAINS = {
    1: (20, 2), 2: (20, 2), 3: (20, 2), 4: (20, 2),
    5: (10, 1), 6: (10, 1), 7: (10, 1), 8: (1.5, 0.1),
    9: (20, 2), 10: (20, 2), 11: (20, 2), 12: (20, 2),
    13: (10, 1), 14: (10, 1), 15: (10, 1), 16: (1.5, 0.1),
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

SAFE_FALL_POSITIONS = {4: 0.2, 12: 0.2, 17: 0.0, 18: 0.52, 19: -0.45}

END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target"),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target"),
}

COLLISION_PAIRS = [
    (["torso_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["torso_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_elbow_collision"], ["right_elbow_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["head_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_elbow_collision"]),
    (["right_hand_collision"], ["left_elbow_collision"]),
    (["pelvis_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["pelvis_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_shoulder_roll_collision"], ["right_hand_collision", "right_elbow_collision"]),
    (["right_shoulder_roll_collision"], ["left_hand_collision", "left_elbow_collision"]),
]

WAIST_CONFIG = {'arm_reach': 0.55, 'deadzone': 0.10, 'compensation_gain': 1.0, 'yaw_smooth': 0.03}
ARM_BIAS_CONFIG = {'outward_bias': 0.3, 'downward_bias': 0.15, 'bias_cost': 5e-2}


# ==================== 工具函数 ====================

def ease_out(t: float, exp: float = 1.1) -> float:
    return 1.0 - pow(1.0 - t, exp)

def ease_in(t: float, exp: float = 0.9) -> float:
    return pow(t, exp)

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)

def compute_waist_yaw(hands_center: np.ndarray, waist_pos: np.ndarray, 
                      safe_radius: float = 0.35, prev_yaw: float = 0.0) -> float:
    """计算腰部yaw，内外圈逻辑：安全范围内完全不动"""
    diff = hands_center - waist_pos
    diff[2] = 0  # 只考虑水平面
    dist = float(np.linalg.norm(diff))
    
    # 内圈：完全不动，保持上次yaw
    if dist <= safe_radius:
        return prev_yaw
    
    # 外圈：跟随手的方向
    target_yaw = float(np.arctan2(diff[1], diff[0]))
    return target_yaw

def compute_waist_pitch(forward_dist: float, arm_reach: float, deadzone: float, gain: float) -> float:
    """计算腰部pitch补偿，使用单手最大前伸距离"""
    threshold = arm_reach - deadzone
    if forward_dist <= threshold:
        return 0.0
    excess = forward_dist - threshold
    return float(np.clip(excess * gain, -0.3, 0.3))

def compute_neck_angles(head_pos: np.ndarray, target_pos: np.ndarray, prev_yaw: float, waist_yaw: float = 0.0) -> tuple:
    """计算neck关节角度（yaw, pitch），相对于躯干局部坐标系"""
    direction = target_pos - head_pos
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return prev_yaw, 0.0
    direction /= dist
    # 世界坐标系下的yaw
    world_yaw = float(np.arctan2(direction[1], direction[0]))
    # 转换为相对于躯干的局部yaw（减去腰部yaw）
    raw_yaw = world_yaw - waist_yaw
    # 归一化到[-pi, pi]
    while raw_yaw > np.pi:
        raw_yaw -= 2 * np.pi
    while raw_yaw < -np.pi:
        raw_yaw += 2 * np.pi
    horizontal_dist = float(np.sqrt(direction[0]**2 + direction[1]**2))
    pitch = float(-np.arctan2(direction[2], horizontal_dist))
    # 限制neck角度范围
    raw_yaw = float(np.clip(raw_yaw, -1.2, 1.2))
    pitch = float(np.clip(pitch, -0.5, 0.8))
    # yaw平滑过渡
    yaw_diff = raw_yaw - prev_yaw
    while yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    while yaw_diff < -np.pi:
        yaw_diff += 2 * np.pi
    yaw = prev_yaw + yaw_diff * 0.1
    return yaw, pitch


# ==================== 数据类 ====================

@dataclass
class VRCalibration:
    done: bool = False
    left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    right: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class MocapRateLimiter:
    """mocap目标速度限制器，防止IK奇异点跳变"""
    prev_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    max_speed: float = 2.0  # 最大速度 m/s

@dataclass
class RampState:
    active: bool = False
    direction: str = "up"
    start_time: float = 0.0
    progress: float = 0.0

@dataclass
class ResetState:
    active: bool = False
    alpha: float = 0.0
    start_q: Optional[np.ndarray] = None
    start_pos: Dict[str, np.ndarray] = field(default_factory=dict)
    start_quat: Dict[str, np.ndarray] = field(default_factory=dict)


# ==================== 控制器 ====================

class HalfBodyIKController:
    def __init__(self, sim2real: bool = False, auto_start_sdk: bool = True,
                 headless: bool = False, host: str = "192.168.5.4", port: int = 5555,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        self.sim2real = sim2real
        self.auto_start_sdk = auto_start_sdk
        self.headless = headless
        self.host = host
        self.port = port
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        self.console = Console()
        
        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.cfg = mink.Configuration(self.model)
        self.model, self.data = self.cfg.model, self.cfg.data
        
        # 索引
        self._init_indices()
        self._init_tasks()
        self._init_state()
        
        # VR
        self.vr = VRReceiver()
        self.vr.start()
        self.vr_calib = VRCalibration()
        
        # SDK
        self.sdk_proc = None
        self.taks_client = None
        self.left_gripper = None
        self.right_gripper = None
        if sim2real:
            if auto_start_sdk and host in ("localhost", "127.0.0.1"):
                self.sdk_proc = self._start_sdk_server()
            taks.connect(address=host, cmd_port=port, wait_data=True, timeout=5.0)
            self.taks_client = taks.register(device_type="Taks-T1-semibody")
            self.left_gripper = taks.register(device_type="Taks-T1-leftgripper")
            self.right_gripper = taks.register(device_type="Taks-T1-rightgripper")
        
        self.rate = RateLimiter(frequency=100.0, warn=False)
        self.dt = self.rate.dt
        self.send_interval = 1.0 / TAKS_SEND_RATE
        self.last_send_time = 0.0
    
    def _init_indices(self):
        self.joint_idx = {k: [self.model.jnt_dofadr[self.model.joint(j).id] for j in v] 
                         for k, v in JOINT_GROUPS.items()}
        self.neck_dof = [int(i) for i in self.joint_idx["neck"]]
        self.waist_dof = [int(i) for i in self.joint_idx["waist"]]
        
        self.waist_yaw_qpos = self.model.jnt_qposadr[self.model.joint("waist_yaw_joint").id]
        self.waist_roll_qpos = self.model.jnt_qposadr[self.model.joint("waist_roll_joint").id]
        self.waist_pitch_qpos = self.model.jnt_qposadr[self.model.joint("waist_pitch_joint").id]
        self.neck_yaw_qpos = self.model.jnt_qposadr[self.model.joint("neck_yaw_joint").id]
        self.neck_pitch_qpos = self.model.jnt_qposadr[self.model.joint("neck_pitch_joint").id]
        self.neck_roll_qpos = self.model.jnt_qposadr[self.model.joint("neck_roll_joint").id]
        
        self.joint_mapping = {}
        for group, names in JOINT_GROUPS.items():
            for jname in names:
                sdk_id = JOINT_NAME_TO_SDK_ID.get(jname)
                if sdk_id:
                    self.joint_mapping[jname] = {
                        'qpos': self.model.jnt_qposadr[self.model.joint(jname).id],
                        'dof': self.model.jnt_dofadr[self.model.joint(jname).id],
                        'sdk_id': sdk_id
                    }
    
    def _init_tasks(self):
        self.tasks = [
            mink.FrameTask("base_link", "body", position_cost=1e6, orientation_cost=1e6),
            mink.PostureTask(self.model, cost=1e-2),
        ]
        self.ee_tasks = {}
        for name, (link, _) in END_EFFECTORS.items():
            task = mink.FrameTask(link, "body", position_cost=2.0, orientation_cost=2.0)
            self.tasks.append(task)
            self.ee_tasks[name] = task
        
        arm_posture = mink.PostureTask(self.model, cost=ARM_BIAS_CONFIG['bias_cost'])
        bias_q = self.cfg.q.copy()
        bias_q[self.model.jnt_qposadr[self.model.joint("right_shoulder_roll_joint").id]] = -ARM_BIAS_CONFIG['outward_bias']
        bias_q[self.model.jnt_qposadr[self.model.joint("right_elbow_joint").id]] = -ARM_BIAS_CONFIG['downward_bias']
        bias_q[self.model.jnt_qposadr[self.model.joint("left_shoulder_roll_joint").id]] = ARM_BIAS_CONFIG['outward_bias']
        bias_q[self.model.jnt_qposadr[self.model.joint("left_elbow_joint").id]] = ARM_BIAS_CONFIG['downward_bias']
        arm_posture.set_target(bias_q)
        self.tasks.append(arm_posture)
        
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model),
            # mink.CollisionAvoidanceLimit(self.model, COLLISION_PAIRS, gain=0.1,
            #                              minimum_distance_from_collisions=0.05,
            #                              collision_detection_distance=0.15)
        ]
        
        self.cfg.update_from_keyframe("home")
        self.tasks[0].set_target_from_configuration(self.cfg)
        self.tasks[1].set_target_from_configuration(self.cfg)
        for name, (link, mocap) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(self.model, self.data, mocap, link, "body")
            self.ee_tasks[name].set_target_from_configuration(self.cfg)
    
    def _init_state(self):
        self.init_q = self.cfg.q.copy()
        self.mocap_ids = {}
        self.init_pos, self.init_quat = {}, {}
        for name, (link, mocap) in END_EFFECTORS.items():
            mid = self.model.body(mocap).mocapid[0]
            self.mocap_ids[name] = mid
            self.init_pos[name] = self.data.mocap_pos[mid].copy()
            self.init_quat[name] = self.data.mocap_quat[mid].copy()
        
        self.waist_init_pos = self.data.xpos[self.model.body("base_link").id].copy()
        self.prev_neck_yaw = 0.0
        self.prev_waist_yaw = 0.0
        self.prev_target_yaw = 0.0  # 用于YAW平滑追踪
        
        # mocap速度限制器
        self.mocap_limiter = MocapRateLimiter()
        for name, mid in self.mocap_ids.items():
            self.mocap_limiter.prev_pos[name] = self.data.mocap_pos[mid].copy()
        
        self.ramp_state = RampState()
        self.reset_state = ResetState()
        self.last_mit_cmd = None
    
    def _start_sdk_server(self):
        sdk_path = Path(__file__).parent / "taks_sdk" / "SDK_MF.py"
        try:
            proc = subprocess.Popen([sys.executable, str(sdk_path)], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2.0)
            if proc.poll() is not None:
                print("[SDK] 服务端启动失败")
                return None
            print(f"[SDK] 服务端已启动 (PID: {proc.pid})")
            return proc
        except Exception as e:
            print(f"[SDK] 启动错误: {e}")
            return None
    
    def calibrate(self):
        vr_data = self.vr.data
        if not vr_data.tracking_enabled:
            print("[VR] 未启用追踪")
            return
        left_mid, right_mid = self.mocap_ids["left_hand"], self.mocap_ids["right_hand"]
        self.vr_calib.left = self.data.mocap_pos[left_mid] - vr_data.left_hand.position
        self.vr_calib.right = self.data.mocap_pos[right_mid] - vr_data.right_hand.position
        self.vr_calib.done = True
        self.vr.reset_smooth()
        print("[VR] 校准完成")
    
    def reset(self):
        self.reset_state.active = True
        self.reset_state.alpha = 0.0
        self.reset_state.start_q = self.cfg.q.copy()
        for name, mid in self.mocap_ids.items():
            self.reset_state.start_pos[name] = self.data.mocap_pos[mid].copy()
            self.reset_state.start_quat[name] = self.data.mocap_quat[mid].copy()
        print("[Reset] 开始复位...")
    
    def start_ramp_up(self):
        self.ramp_state = RampState(active=True, direction="up", start_time=time.time(), progress=0.0)
        print("[Ramp Up] 启动...")
    
    def start_ramp_down(self):
        self.ramp_state = RampState(active=True, direction="down", start_time=time.time(), progress=1.0)
        print("[Ramp Down] 缓停止...")
    
    def build_mit_cmd(self, q: np.ndarray, qfrc_bias: np.ndarray, ramp_progress: float) -> Dict:
        mit_cmd = {}
        for jname, info in self.joint_mapping.items():
            sdk_id = info['sdk_id']
            qpos_idx = info['qpos']
            dof_idx = info['dof']
            
            base_kp, base_kd = SDK_JOINT_GAINS.get(sdk_id, (10, 1))
            safe_kp, safe_kd = SAFE_KP_KD.get(sdk_id, (5, 1))
            kp = safe_kp + (base_kp - safe_kp) * ease_out(ramp_progress)
            kd = safe_kd + (base_kd - safe_kd) * ease_out(ramp_progress)
            
            target_q = float(q[qpos_idx])
            if sdk_id in SAFE_FALL_POSITIONS:
                safe_q = SAFE_FALL_POSITIONS[sdk_id]
                target_q = safe_q + (target_q - safe_q) * ease_out(ramp_progress)
            
            tau = float(qfrc_bias[dof_idx]) * FEEDFORWARD_SCALE * ramp_progress
            
            mit_cmd[sdk_id] = {'q': target_q, 'dq': 0.0, 'tau': tau, 'kp': kp, 'kd': kd}
        return mit_cmd
    
    def send_to_real(self, mit_cmd: Dict):
        if not self.sim2real or not self.taks_client:
            return
        now = time.time()
        if now - self.last_send_time < self.send_interval:
            return
        self.last_send_time = now
        self.taks_client.controlMIT(mit_cmd)
    
    def send_gripper_cmd(self, left_percent: float, right_percent: float):
        """发送夹爪控制命令"""
        if not self.sim2real:
            return
        if self.left_gripper:
            self.left_gripper.controlMIT(percent=left_percent, kp=2.0, kd=0.2)
        if self.right_gripper:
            self.right_gripper.controlMIT(percent=right_percent, kp=2.0, kd=0.2)
    
    def step(self) -> Dict:
        # VR输入
        vr_data = self.vr.data
        left_mid, right_mid = self.mocap_ids["left_hand"], self.mocap_ids["right_hand"]
        
        # VR按钮事件处理
        if vr_data.button_events.right_b:
            self.calibrate()
        if vr_data.button_events.right_a:
            self.reset()
        
        # 无论tracking状态如何，只要校准完成就处理VR数据
        if self.vr_calib.done:
            # 计算目标位置
            target_left = vr_data.left_hand.position + self.vr_calib.left
            target_right = vr_data.right_hand.position + self.vr_calib.right
            
            # 应用速度限制（防止快速移动导致IK奇异点跳变）
            for name, target in [("left_hand", target_left), ("right_hand", target_right)]:
                prev = self.mocap_limiter.prev_pos[name]
                delta = target - prev
                dist = float(np.linalg.norm(delta))
                max_delta = self.mocap_limiter.max_speed * self.dt
                if dist > max_delta:
                    target = prev + delta * (max_delta / dist)
                if name == "left_hand":
                    target_left = target
                else:
                    target_right = target
                self.mocap_limiter.prev_pos[name] = target.copy()
            
            self.data.mocap_pos[left_mid] = target_left
            self.data.mocap_quat[left_mid] = vr_data.left_hand.quaternion
            self.data.mocap_pos[right_mid] = target_right
            self.data.mocap_quat[right_mid] = vr_data.right_hand.quaternion
        
        # Ramp处理
        if self.ramp_state.active:
            elapsed = time.time() - self.ramp_state.start_time
            if self.ramp_state.direction == "up":
                if elapsed >= RAMP_UP_TIME:
                    self.ramp_state.active = False
                    self.ramp_state.progress = 1.0
                    print("[Ramp Up] 完成")
                else:
                    self.ramp_state.progress = elapsed / RAMP_UP_TIME
            else:
                if elapsed >= RAMP_DOWN_TIME:
                    self.ramp_state.active = False
                    self.ramp_state.progress = 0.0
                    print("[Ramp Down] 完成")
                    if self.sim2real and self.taks_client:
                        taks.disconnect()
                else:
                    self.ramp_state.progress = 1.0 - elapsed / RAMP_DOWN_TIME
        
        # 双手中心（用于yaw跟随）
        hands_center = (self.data.mocap_pos[left_mid] + self.data.mocap_pos[right_mid]) / 2.0
        cfg_w = WAIST_CONFIG
        
        # 计算左右手分别到躯干的水平距离和前向距离
        left_diff = self.data.mocap_pos[left_mid] - self.waist_init_pos
        left_diff[2] = 0
        left_dist = float(np.linalg.norm(left_diff))
        left_forward = float(left_diff[0])  # X轴前向距离
        
        right_diff = self.data.mocap_pos[right_mid] - self.waist_init_pos
        right_diff[2] = 0
        right_dist = float(np.linalg.norm(right_diff))
        right_forward = float(right_diff[0])  # X轴前向距离
        
        # 用单手最大距离计算blend_factor（不叠加，避免双手权重叠加）
        max_hand_dist = max(left_dist, right_dist)
        # 单手最大前向距离（不叠加，只取最大值，避免双手同时伸出时权重叠加）
        max_forward_dist = max(left_forward, right_forward, 0.0)
        
        # 安全范围配置：内圈完全不动，外圈完全补偿
        waist_safe_zone_inner = 0.35
        waist_safe_zone_outer = 0.50
        
        # 计算blend_factor（基于单手最大距离）
        if max_hand_dist <= waist_safe_zone_inner:
            blend_factor = 0.0
        elif max_hand_dist >= waist_safe_zone_outer:
            blend_factor = 1.0
        else:
            blend_factor = (max_hand_dist - waist_safe_zone_inner) / (waist_safe_zone_outer - waist_safe_zone_inner)
            blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
        
        # YAW内外圈逻辑：安全范围内完全不动，超出范围才跟随
        target_waist_yaw = compute_waist_yaw(hands_center, self.waist_init_pos, 
                                             safe_radius=waist_safe_zone_inner, 
                                             prev_yaw=self.prev_waist_yaw)
        # 平滑过渡
        yaw_diff = target_waist_yaw - self.prev_waist_yaw
        while yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        while yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi
        waist_yaw = self.prev_waist_yaw + yaw_diff * 0.15
        self.prev_waist_yaw = waist_yaw
        
        # pitch补偿：使用单手最大前向距离（单手前伸也会弯腰）
        target_pitch = compute_waist_pitch(max_forward_dist, cfg_w['arm_reach'], 
                                           cfg_w['deadzone'], cfg_w['compensation_gain'])
        waist_pitch = target_pitch * blend_factor
        
        # neck look-at（使用局部坐标系）
        head_pos = self.data.xpos[self.model.body("neck_pitch_link").id]
        neck_yaw, neck_pitch = compute_neck_angles(head_pos, hands_center, self.prev_neck_yaw, waist_yaw)
        self.prev_neck_yaw = neck_yaw
        
        # 复位处理
        if self.reset_state.active:
            self.reset_state.alpha += self.dt / 1.5
            alpha = min(1.0, self.reset_state.alpha)
            for name, mid in self.mocap_ids.items():
                self.data.mocap_pos[mid] = (1-alpha) * self.reset_state.start_pos[name] + alpha * self.init_pos[name]
                self.data.mocap_quat[mid] = slerp(self.reset_state.start_quat[name], self.init_quat[name], alpha)
            self.cfg.update(self.reset_state.start_q * (1 - alpha) + self.init_q * alpha)
            for name in END_EFFECTORS:
                self.ee_tasks[name].set_target_from_configuration(self.cfg)
            if alpha >= 1.0:
                self.reset_state.active = False
                self.prev_waist_yaw = 0.0
                self.prev_target_yaw = 0.0
                # 清除校准状态，允许重新校准
                self.vr_calib.done = False
                print("[Reset] 完成，可重新校准")
        
        # IK解算
        for name, mid in self.mocap_ids.items():
            self.ee_tasks[name].set_target(mink.SE3.from_mocap_id(self.data, mid))
        
        # IK解算（增加damping提高稳定性，减少胸前卡顿）
        constraints = [mink.DofFreezingTask(self.model, dof_indices=self.neck_dof + self.waist_dof)]
        try:
            vel = mink.solve_ik(self.cfg, self.tasks, self.dt, "daqp", damping=5e-2, 
                               limits=self.limits, constraints=constraints)
        except mink.exceptions.NoSolutionFound:
            vel = mink.solve_ik(self.cfg, self.tasks, self.dt, "daqp", damping=0.5, 
                               limits=self.limits, constraints=[])
        self.cfg.integrate_inplace(vel, self.dt)
        
        # 设置腰部和neck
        q = self.cfg.q.copy()
        q[self.waist_yaw_qpos] = waist_yaw
        q[self.waist_pitch_qpos] = waist_pitch
        q[self.waist_roll_qpos] = 0.0
        q[self.neck_yaw_qpos] = neck_yaw
        q[self.neck_pitch_qpos] = neck_pitch
        q[self.neck_roll_qpos] = 0.0
        self.cfg.update(q)
        
        # 计算前馈扭矩（所有关节）
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
        
        # 构建MIT命令（加入前馈扭矩补偿）
        mit_cmd = self.build_mit_cmd(self.cfg.q, self.data.qfrc_bias, self.ramp_state.progress)
        self.last_mit_cmd = mit_cmd
        self.send_to_real(mit_cmd)
        
        # 夹爪控制（VR扳机）
        left_gripper_percent = vr_data.left_hand.gripper * 100.0
        right_gripper_percent = vr_data.right_hand.gripper * 100.0
        self.send_gripper_cmd(left_gripper_percent, right_gripper_percent)
        
        return mit_cmd
    
    def run(self):
        def key_callback(keycode):
            if keycode == 32:  # Space
                if not self.ramp_state.active and self.ramp_state.progress < 0.5:
                    self.start_ramp_up()
                elif not self.ramp_state.active:
                    self.start_ramp_down()
            elif keycode == 259:  # Backspace
                self.reset()
        
        with mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, 
                                          show_right_ui=False, key_callback=key_callback) as viewer:
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            print("\n[控制] Space=启停, Backspace=复位, RB=VR校准, RA=复位\n")
            
            while viewer.is_running():
                self.step()
                mujoco.mj_camlight(self.model, self.data)
                viewer.sync()
                self.rate.sleep()
    
    def close(self):
        if self.ramp_state.progress > 0 and self.sim2real:
            self.start_ramp_down()
            while self.ramp_state.active:
                self.step()
                time.sleep(0.01)
        self.vr.stop()
        if self.taks_client:
            taks.disconnect()
        if self.sdk_proc:
            self.sdk_proc.terminate()
            self.sdk_proc.wait(timeout=2)
        print("[Controller] 已关闭")


# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="半身VR控制IK - SIM2REAL")
    parser.add_argument("--headless", action="store_true", default=False, help="无头模式")
    parser.add_argument("--host", type=str, default="192.168.5.16", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", default=True, help="禁用真机控制")
    parser.add_argument("--no-ramp-up", action="store_true", default=True, help="禁用缓启动")
    parser.add_argument("--no-ramp-down", action="store_true", default=True, help="禁用缓停止")
    args = parser.parse_args()
    
    controller = HalfBodyIKController(
        sim2real=not args.no_real,
        headless=args.headless,
        host=args.host,
        port=args.port,
        enable_ramp_up=not args.no_ramp_up,
        enable_ramp_down=not args.no_ramp_down
    )
    
    def signal_handler(sig, frame):
        print("\n[信号] 收到中断，安全关闭...")
        controller.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        controller.run()
    finally:
        controller.close()