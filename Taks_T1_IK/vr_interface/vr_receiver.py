"""VR UDP接收器 - 简洁版

功能: UDP接收VR位姿(头部/双手) + 按键事件 + EMA平滑
"""

import socket
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

UDP_IP = "0.0.0.0"
UDP_PORT = 7000
BUFFER_SIZE = 4096
EMA_ALPHA = 0.9  # EMA平滑系数(0.3-0.5平滑, 0.6-0.8快速响应)


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
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[VRData], None]] = None
    
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
                button_events=events
            )
    
    @property
    def tracking_enabled(self) -> bool:
        with self._lock:
            return self._data.tracking_enabled
    
    def set_callback(self, callback: Callable[[VRData], None]) -> None:
        self._callback = callback
    
    def reset_smooth(self) -> None:
        """重置平滑状态(校准后调用)"""
        with self._lock:
            self._init = False
    
    def _parse_pose(self, d: dict, pose: VRPose) -> None:
        """解析并平滑位姿"""
        if "position" not in d:
            return
        pos = np.array(d["position"], dtype=np.float64)
        quat = np.array(d.get("quaternion", [1, 0, 0, 0]), dtype=np.float64)
        if self.convert_quat:
            quat = unity_to_mujoco_quat(quat)
        grip = float(d.get("gripper", 0.0))
        
        # 首次初始化
        if not self._init:
            pose.position, pose.quaternion, pose.gripper = pos.copy(), quat.copy(), grip
            return
        
        # EMA平滑
        pose.position = EMA_ALPHA * pos + (1 - EMA_ALPHA) * pose.position
        pose.quaternion = slerp(pose.quaternion, quat, EMA_ALPHA)
        pose.gripper = EMA_ALPHA * grip + (1 - EMA_ALPHA) * pose.gripper
    
    def _parse_buttons(self, d: dict) -> None:
        """解析按键事件"""
        events = d.get("buttonEvents", {})
        for unity_key, local_key in [("leftX", "left_x"), ("leftY", "left_y"), ("rightA", "right_a"), ("rightB", "right_b")]:
            if events.get(unity_key, False):
                self._btn[local_key] = True
                self._btn_consumed[local_key] = False
    
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
                    self._parse_buttons(d)
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