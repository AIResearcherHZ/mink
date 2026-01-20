"""
VR UDP接收器
功能: UDP接收VR位姿(头部/双手) + 按键事件 + EMA平滑
"""

import socket
import json
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

UDP_IP = "0.0.0.0"
UDP_PORT = 7000
BUFFER_SIZE = 4096


@dataclass
class VRPose:
    """VR位姿"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gripper: float = 0.0


@dataclass
class VRButtonEvents:
    """VR按键双击事件"""
    left_x: bool = False
    left_y: bool = False
    right_a: bool = False
    right_b: bool = False


@dataclass
class VRData:
    """VR完整数据"""
    head: VRPose = field(default_factory=VRPose)
    left_hand: VRPose = field(default_factory=VRPose)
    right_hand: VRPose = field(default_factory=VRPose)
    tracking_enabled: bool = False
    timestamp: float = 0.0
    total_offset: float = 0.0  # 总偏移量(米)
    button_events: VRButtonEvents = field(default_factory=VRButtonEvents)


def unity_to_mujoco_quat(q: np.ndarray) -> np.ndarray:
    """Unity四元数转MuJoCo(虚部取反)"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


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


class VRReceiver:
    """VR UDP接收器"""
    
    def __init__(self, ip: str = UDP_IP, port: int = UDP_PORT, convert_quat: bool = True):
        self.ip, self.port = ip, port
        self.convert_quat = convert_quat
        self._data = VRData()  # 平滑后数据
        self._init = False  # 是否已初始化
        self._btn = {"left_x": False, "left_y": False, "right_a": False, "right_b": False}
        self._btn_consumed = {"left_x": True, "left_y": True, "right_a": True, "right_b": True}
        self._btn_times = {"left_x": 0.0, "left_y": 0.0, "right_a": 0.0, "right_b": 0.0}  # 按钮事件时间
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[VRData], None]] = None
        self._last_tracking_state: Optional[bool] = None  # 上次追踪状态
        self._state_change_time: float = 0.0  # 状态变更时间
        self._state_change_callback: Optional[Callable[[bool], None]] = None  # 状态变更回调
    
    @property
    def data(self) -> VRData:
        """获取平滑后的VR数据(线程安全)"""
        with self._lock:
            events = VRButtonEvents(
                left_x=self._btn["left_x"] and not self._btn_consumed["left_x"],
                left_y=self._btn["left_y"] and not self._btn_consumed["left_y"],
                right_a=self._btn["right_a"] and not self._btn_consumed["right_a"],
                right_b=self._btn["right_b"] and not self._btn_consumed["right_b"]
            )
            for k in ["left_x", "left_y", "right_a", "right_b"]:
                if getattr(events, k):
                    self._btn_consumed[k] = True
            return VRData(
                head=VRPose(self._data.head.position.copy(), self._data.head.quaternion.copy(), self._data.head.gripper),
                left_hand=VRPose(self._data.left_hand.position.copy(), self._data.left_hand.quaternion.copy(), self._data.left_hand.gripper),
                right_hand=VRPose(self._data.right_hand.position.copy(), self._data.right_hand.quaternion.copy(), self._data.right_hand.gripper),
                tracking_enabled=self._data.tracking_enabled,
                timestamp=self._data.timestamp,
                total_offset=self._data.total_offset,
                button_events=events
            )
    
    @property
    def tracking_enabled(self) -> bool:
        with self._lock:
            return self._data.tracking_enabled
    
    def set_callback(self, callback: Callable[[VRData], None]) -> None:
        self._callback = callback
    
    def set_state_change_callback(self, callback: Callable[[bool], None]) -> None:
        """设置追踪状态变更回调"""
        self._state_change_callback = callback
    
    @property
    def total_offset(self) -> float:
        """获取总偏移量(米)"""
        with self._lock:
            return self._data.total_offset
    
    @property
    def state_just_changed(self) -> bool:
        """状态是否刚刚变更(2秒内)"""
        return (time.time() - self._state_change_time) < 2.0
    
    def get_active_button_events(self, duration: float = 1.5) -> list:
        """获取指定时间内活跃的按钮事件列表"""
        current_time = time.time()
        active = []
        names = {"left_x": "左手X", "left_y": "左手Y", "right_a": "右手A", "right_b": "右手B"}
        with self._lock:
            for key, name in names.items():
                if current_time - self._btn_times[key] < duration:
                    active.append(name)
        return active
    
    def reset_smooth(self) -> None:
        """重置平滑状态(校准后调用)"""
        with self._lock:
            self._init = False
    
    def _parse_pose(self, d: dict, pose: VRPose) -> None:
        """解析位姿（直接使用原始数据，无平滑）"""
        if "position" not in d:
            return
        pos = np.array(d["position"], dtype=np.float64)
        quat = np.array(d.get("quaternion", [1, 0, 0, 0]), dtype=np.float64)
        if self.convert_quat:
            quat = unity_to_mujoco_quat(quat)
        grip = float(d.get("gripper", 0.0))
        
        pose.position = pos.copy()
        pose.quaternion = quat.copy()
        pose.gripper = grip
    
    def _parse_buttons(self, d: dict) -> None:
        """解析按键事件"""
        events = d.get("buttonEvents", {})
        current_time = time.time()
        for unity_key, local_key in [("leftX", "left_x"), ("leftY", "left_y"), ("rightA", "right_a"), ("rightB", "right_b")]:
            if events.get(unity_key, False):
                if current_time - self._btn_times[local_key] > 0.5:  # 防止重复触发
                    self._btn[local_key] = True
                    self._btn_consumed[local_key] = False
                    self._btn_times[local_key] = current_time
    
    def _check_state_change(self, tracking_enabled: bool) -> bool:
        """检查追踪状态是否变更"""
        if self._last_tracking_state is None:
            self._last_tracking_state = tracking_enabled
            self._state_change_time = time.time()
            return True
        if tracking_enabled != self._last_tracking_state:
            self._last_tracking_state = tracking_enabled
            self._state_change_time = time.time()
            if self._state_change_callback:
                self._state_change_callback(tracking_enabled)
            return True
        return False
    
    def _receiver_loop(self) -> None:
        """接收线程"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.ip, self.port))
        sock.settimeout(0.1)
        print(f"[VR] 监听 {self.ip}:{self.port}")
        
        while self._running:
            try:
                data_bytes, _ = sock.recvfrom(BUFFER_SIZE)
                d = json.loads(data_bytes.decode('utf-8'))
                with self._lock:
                    if "head" in d:
                        self._parse_pose(d["head"], self._data.head)
                    if "leftHand" in d:
                        self._parse_pose(d["leftHand"], self._data.left_hand)
                    if "rightHand" in d:
                        self._parse_pose(d["rightHand"], self._data.right_hand)
                    # 首次收到完整数据后标记初始化完成
                    if "head" in d and "leftHand" in d and "rightHand" in d:
                        self._init = True
                    self._data.tracking_enabled = d.get("trackingEnabled", False)
                    self._data.timestamp = d.get("timestamp", 0.0)
                    self._data.total_offset = d.get("totalOffset", 0.0)
                    self._parse_buttons(d)
                    self._check_state_change(self._data.tracking_enabled)
                if self._callback:
                    self._callback(self.data)
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                pass
            except Exception as e:
                if self._running:
                    print(f"[VR] 错误: {e}")
        sock.close()
        print("[VR] 接收器已停止")
    
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()