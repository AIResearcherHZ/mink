"""通用VR UDP接收器模块

功能:
- UDP接收VR位姿数据(头部、双手)
- ABXY按键双击事件检测
- EMA数据平滑 + 帧率稳定
- Unity到MuJoCo四元数转换
"""

import socket
import json
import threading
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

UDP_IP = "0.0.0.0"
UDP_PORT = 7000
BUFFER_SIZE = 4096

# EMA平滑参数
DEFAULT_EMA_ALPHA = 0.3  # 平滑系数(0-1, 越小越平滑)
DEFAULT_TARGET_FPS = 100.0  # 目标输出帧率


@dataclass
class VRPose:
    """VR位姿数据"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    gripper: float = 0.0


@dataclass
class VRButtonEvents:
    """VR按键双击事件"""
    left_x: bool = False  # 左手X按钮双击
    left_y: bool = False  # 左手Y按钮双击
    right_a: bool = False  # 右手A按钮双击
    right_b: bool = False  # 右手B按钮双击


@dataclass
class VRData:
    """VR完整数据"""
    head: VRPose = field(default_factory=VRPose)
    left_hand: VRPose = field(default_factory=VRPose)
    right_hand: VRPose = field(default_factory=VRPose)
    tracking_enabled: bool = False
    timestamp: float = 0.0
    button_events: VRButtonEvents = field(default_factory=VRButtonEvents)


def unity_to_mujoco_quat(quat: np.ndarray) -> np.ndarray:
    """Unity四元数转MuJoCo四元数(虚部取反)"""
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """四元数球面插值"""
    q0, q1 = q0.copy(), q1.copy()
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = (1 - alpha) * q0 + alpha * q1
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        result = (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)
    return result / np.linalg.norm(result)


class VRReceiver:
    """VR UDP接收器，支持EMA平滑和按键双击事件"""
    
    def __init__(self, ip: str = UDP_IP, port: int = UDP_PORT, convert_quat: bool = True,
                 ema_alpha: float = DEFAULT_EMA_ALPHA, target_fps: float = DEFAULT_TARGET_FPS):
        self.ip = ip
        self.port = port
        self.convert_quat = convert_quat
        self.ema_alpha = ema_alpha  # EMA平滑系数
        self.target_fps = target_fps  # 目标帧率
        
        # 原始数据(接收线程写入)
        self._raw_data = VRData()
        # 平滑后数据(主线程读取)
        self._smooth_data = VRData()
        # 按键事件(一次性触发)
        self._button_events = VRButtonEvents()
        self._button_event_consumed = {"left_x": True, "left_y": True, "right_a": True, "right_b": True}
        
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[VRData], None]] = None
        
        # 帧率稳定相关
        self._last_update_time = 0.0
        self._data_received = False  # 是否收到过数据
    
    @property
    def data(self) -> VRData:
        """获取EMA平滑后的VR数据(线程安全)"""
        with self._lock:
            # 获取并清除按键事件
            events = VRButtonEvents(
                left_x=self._button_events.left_x and not self._button_event_consumed["left_x"],
                left_y=self._button_events.left_y and not self._button_event_consumed["left_y"],
                right_a=self._button_events.right_a and not self._button_event_consumed["right_a"],
                right_b=self._button_events.right_b and not self._button_event_consumed["right_b"]
            )
            # 标记事件已消费
            if events.left_x:
                self._button_event_consumed["left_x"] = True
            if events.left_y:
                self._button_event_consumed["left_y"] = True
            if events.right_a:
                self._button_event_consumed["right_a"] = True
            if events.right_b:
                self._button_event_consumed["right_b"] = True
            
            return VRData(
                head=VRPose(
                    position=self._smooth_data.head.position.copy(),
                    quaternion=self._smooth_data.head.quaternion.copy(),
                    gripper=self._smooth_data.head.gripper
                ),
                left_hand=VRPose(
                    position=self._smooth_data.left_hand.position.copy(),
                    quaternion=self._smooth_data.left_hand.quaternion.copy(),
                    gripper=self._smooth_data.left_hand.gripper
                ),
                right_hand=VRPose(
                    position=self._smooth_data.right_hand.position.copy(),
                    quaternion=self._smooth_data.right_hand.quaternion.copy(),
                    gripper=self._smooth_data.right_hand.gripper
                ),
                tracking_enabled=self._smooth_data.tracking_enabled,
                timestamp=self._smooth_data.timestamp,
                button_events=events
            )
    
    @property
    def tracking_enabled(self) -> bool:
        """获取追踪状态"""
        with self._lock:
            return self._smooth_data.tracking_enabled
    
    def set_callback(self, callback: Callable[[VRData], None]) -> None:
        """设置数据更新回调"""
        self._callback = callback
    
    def _parse_pose(self, data: dict, pose: VRPose) -> tuple:
        """解析位姿数据，返回(position, quaternion, gripper)"""
        pos = pose.position.copy()
        quat = pose.quaternion.copy()
        grip = pose.gripper
        
        if "position" in data:
            pos = np.array(data["position"], dtype=np.float64)
        if "quaternion" in data:
            quat = np.array(data["quaternion"], dtype=np.float64)
            if self.convert_quat:
                quat = unity_to_mujoco_quat(quat)
        if "gripper" in data:
            grip = float(data["gripper"])
        return pos, quat, grip
    
    def _apply_ema(self, old_pose: VRPose, new_pos: np.ndarray, new_quat: np.ndarray, new_grip: float) -> None:
        """应用EMA平滑到位姿"""
        alpha = self.ema_alpha
        old_pose.position = alpha * new_pos + (1 - alpha) * old_pose.position
        old_pose.quaternion = slerp(old_pose.quaternion, new_quat, alpha)
        old_pose.gripper = alpha * new_grip + (1 - alpha) * old_pose.gripper
    
    def _parse_button_events(self, data: dict) -> None:
        """解析按键双击事件"""
        events = data.get("buttonEvents", {})
        if events.get("leftX", False):
            self._button_events.left_x = True
            self._button_event_consumed["left_x"] = False
        if events.get("leftY", False):
            self._button_events.left_y = True
            self._button_event_consumed["left_y"] = False
        if events.get("rightA", False):
            self._button_events.right_a = True
            self._button_event_consumed["right_a"] = False
        if events.get("rightB", False):
            self._button_events.right_b = True
            self._button_event_consumed["right_b"] = False
    
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
                    # 解析原始数据
                    if "head" in data:
                        pos, quat, grip = self._parse_pose(data["head"], self._raw_data.head)
                        if not self._data_received:
                            # 首次接收，直接赋值
                            self._smooth_data.head.position = pos.copy()
                            self._smooth_data.head.quaternion = quat.copy()
                        else:
                            self._apply_ema(self._smooth_data.head, pos, quat, grip)
                        self._raw_data.head.position = pos
                        self._raw_data.head.quaternion = quat
                    
                    if "leftHand" in data:
                        pos, quat, grip = self._parse_pose(data["leftHand"], self._raw_data.left_hand)
                        if not self._data_received:
                            self._smooth_data.left_hand.position = pos.copy()
                            self._smooth_data.left_hand.quaternion = quat.copy()
                            self._smooth_data.left_hand.gripper = grip
                        else:
                            self._apply_ema(self._smooth_data.left_hand, pos, quat, grip)
                        self._raw_data.left_hand.position = pos
                        self._raw_data.left_hand.quaternion = quat
                        self._raw_data.left_hand.gripper = grip
                    
                    if "rightHand" in data:
                        pos, quat, grip = self._parse_pose(data["rightHand"], self._raw_data.right_hand)
                        if not self._data_received:
                            self._smooth_data.right_hand.position = pos.copy()
                            self._smooth_data.right_hand.quaternion = quat.copy()
                            self._smooth_data.right_hand.gripper = grip
                        else:
                            self._apply_ema(self._smooth_data.right_hand, pos, quat, grip)
                        self._raw_data.right_hand.position = pos
                        self._raw_data.right_hand.quaternion = quat
                        self._raw_data.right_hand.gripper = grip
                    
                    self._smooth_data.tracking_enabled = data.get("trackingEnabled", False)
                    self._smooth_data.timestamp = data.get("timestamp", 0.0)
                    
                    # 解析按键事件
                    self._parse_button_events(data)
                    
                    self._data_received = True
                
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