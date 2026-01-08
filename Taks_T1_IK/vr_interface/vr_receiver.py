"""通用VR UDP接收器模块"""

import socket
import json
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

UDP_IP = "0.0.0.0"
UDP_PORT = 7000
BUFFER_SIZE = 4096


@dataclass
class VRPose:
    """VR位姿数据"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gripper: float = 0.0


@dataclass
class VRData:
    """VR完整数据"""
    head: VRPose = field(default_factory=VRPose)
    left_hand: VRPose = field(default_factory=VRPose)
    right_hand: VRPose = field(default_factory=VRPose)
    tracking_enabled: bool = False
    timestamp: float = 0.0


def unity_to_mujoco_quat(quat: np.ndarray) -> np.ndarray:
    """Unity四元数转MuJoCo四元数(虚部取反)"""
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


class VRReceiver:
    """VR UDP接收器"""
    
    def __init__(self, ip: str = UDP_IP, port: int = UDP_PORT, convert_quat: bool = True):
        self.ip = ip
        self.port = port
        self.convert_quat = convert_quat
        self._data = VRData()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[VRData], None]] = None
    
    @property
    def data(self) -> VRData:
        """获取当前VR数据(线程安全)"""
        with self._lock:
            return VRData(
                head=VRPose(
                    position=self._data.head.position.copy(),
                    quaternion=self._data.head.quaternion.copy(),
                    gripper=self._data.head.gripper
                ),
                left_hand=VRPose(
                    position=self._data.left_hand.position.copy(),
                    quaternion=self._data.left_hand.quaternion.copy(),
                    gripper=self._data.left_hand.gripper
                ),
                right_hand=VRPose(
                    position=self._data.right_hand.position.copy(),
                    quaternion=self._data.right_hand.quaternion.copy(),
                    gripper=self._data.right_hand.gripper
                ),
                tracking_enabled=self._data.tracking_enabled,
                timestamp=self._data.timestamp
            )
    
    @property
    def tracking_enabled(self) -> bool:
        """获取追踪状态"""
        with self._lock:
            return self._data.tracking_enabled
    
    def set_callback(self, callback: Callable[[VRData], None]) -> None:
        """设置数据更新回调"""
        self._callback = callback
    
    def _parse_pose(self, data: dict, pose: VRPose) -> None:
        """解析位姿数据"""
        if "position" in data:
            pose.position = np.array(data["position"], dtype=np.float64)
        if "quaternion" in data:
            quat = np.array(data["quaternion"], dtype=np.float64)
            pose.quaternion = unity_to_mujoco_quat(quat) if self.convert_quat else quat
        if "gripper" in data:
            pose.gripper = float(data["gripper"])
    
    def _receiver_loop(self) -> None:
        """接收线程主循环"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.ip, self.port))
        sock.settimeout(0.1)
        print(f"[VR] 监听 {self.ip}:{self.port}")
        
        while self._running:
            try:
                data_bytes, _ = sock.recvfrom(BUFFER_SIZE)
                data = json.loads(data_bytes.decode('utf-8'))
                
                with self._lock:
                    if "head" in data:
                        self._parse_pose(data["head"], self._data.head)
                    if "leftHand" in data:
                        self._parse_pose(data["leftHand"], self._data.left_hand)
                    if "rightHand" in data:
                        self._parse_pose(data["rightHand"], self._data.right_hand)
                    self._data.tracking_enabled = data.get("trackingEnabled", False)
                    self._data.timestamp = data.get("timestamp", 0.0)
                
                if self._callback:
                    self._callback(self.data)
                    
            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                print(f"[VR] JSON解析错误: {e}")
            except Exception as e:
                if self._running:
                    print(f"[VR] 错误: {e}")
        
        sock.close()
        print("[VR] 接收器已停止")
    
    def start(self) -> None:
        """启动接收器"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止接收器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
