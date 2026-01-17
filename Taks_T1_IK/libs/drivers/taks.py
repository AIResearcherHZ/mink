#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK 客户端 - Zenoh Pub/Sub版本
使用Zenoh订阅服务端发布的状态，实现低延迟通信
"""

import zenoh
import threading
import time
import pyarrow as pa
from typing import Dict, Optional, Tuple
from threading import Lock

# CUDA检测
HAS_CUDA = False
_cuda_ctx = None
try:
    import pyarrow.cuda as pa_cuda
    num_devices = pa_cuda.Context.get_num_devices()
    if num_devices > 0:
        HAS_CUDA = True
        _cuda_ctx = pa_cuda.Context(0)
        print(f"✓ PyArrow CUDA加速已启用 (设备数: {num_devices})")
    else:
        print("✓ PyArrow CPU模式 (无CUDA设备)")
except ImportError:
    print("✓ PyArrow CPU模式 (小消息场景下性能已优化，无需CUDA)")
except Exception as e:
    print(f"✓ PyArrow CPU模式 (CUDA不可用: {type(e).__name__})")


def _serialize_msg(msg: dict) -> bytes:
    """序列化消息为Arrow IPC格式"""
    if not msg:
        return b""
    arrays = []
    names = list(msg.keys())
    for k in names:
        v = msg[k]
        if isinstance(v, dict):
            arrays.append(pa.array([str(v)]))
        elif isinstance(v, list):
            arrays.append(pa.array([str(v)]))
        else:
            arrays.append(pa.array([str(v)]))
    batch = pa.record_batch(arrays, names=names)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


def _deserialize_msg(data: bytes) -> dict:
    """反序列化Arrow IPC格式为消息"""
    reader = pa.ipc.open_stream(pa.py_buffer(data))
    batch = reader.read_next_batch()
    msg = {}
    for i, name in enumerate(batch.schema.names):
        val = batch.column(i)[0].as_py()
        if val is None:
            msg[name] = None
            continue
        if not isinstance(val, str):
            msg[name] = val
            continue
        # 尝试还原dict/list
        if val.startswith('{') or val.startswith('['):
            try:
                import ast
                msg[name] = ast.literal_eval(val)
            except:
                msg[name] = val
        elif val == 'True':
            msg[name] = True
        elif val == 'False':
            msg[name] = False
        elif val.isdigit():
            msg[name] = int(val)
        else:
            try:
                msg[name] = float(val)
            except:
                msg[name] = val
    return msg

# ============ 夹爪配置 ============
# 夹爪位置定义 (电机弧度位置)
GRIPPER_OPEN = 0.0       # 打开位置
GRIPPER_CLOSE = 1.05     # 闭合位置

# 左右手夹爪镜像配置 (预留，目前默认都一样)
RIGHT_GRIPPER_DIRECTION = 1  # 右手夹爪方向系数
LEFT_GRIPPER_DIRECTION = 1   # 左手匹爪方向系数 (如需镜像改为-1)

# 夹爪关节ID
RIGHT_GRIPPER_ID = 8   # 右手夹爪
LEFT_GRIPPER_ID = 16   # 左手夹爪

# ============ 全局状态 ============
_session: Optional[zenoh.Session] = None
_server_locator: Optional[str] = None
_lock = threading.Lock()
_registered_devices: list = []

# 订阅者
_sub_motor = None
_sub_imu = None

# 状态缓存（由订阅回调更新）
_motor_state: Dict[int, Dict] = {}
_motor_ts: float = 0
_imu_state: Dict = {}
_imu_ts: float = 0
_state_lock = Lock()


# ============ 订阅回调 ============
def _on_motor_state(sample):
    """电机状态回调"""
    global _motor_state, _motor_ts
    try:
        data = _deserialize_msg(sample.payload.to_bytes())
        with _state_lock:
            _motor_ts = data.get('ts', time.perf_counter())
            joints = data.get('joints', {})
            for k, v in joints.items():
                _motor_state[int(k)] = v
    except:
        pass


def _on_imu_state(sample):
    """IMU状态回调"""
    global _imu_state, _imu_ts
    try:
        data = _deserialize_msg(sample.payload.to_bytes())
        with _state_lock:
            _imu_ts = data.get('ts', time.perf_counter())
            _imu_state = data
    except:
        pass


# ============ 连接管理 ============
def connect(address: str = None, cmd_port: int = 5555, wait_data: bool = True, timeout: float = 5.0):
    """连接服务器"""
    global _session, _server_locator, _sub_motor, _sub_imu
    disconnect()
    
    config = zenoh.Config()
    if address:
        _server_locator = f"tcp/{address}:{cmd_port}"
        config.insert_json5("connect/endpoints", f'["{_server_locator}"]')
    
    _session = zenoh.open(config)
    
    # 订阅状态
    _sub_motor = _session.declare_subscriber("taks/state/motor", _on_motor_state)
    _sub_imu = _session.declare_subscriber("taks/state/imu", _on_imu_state)
    
    print(f"✓ 已连接到 {address}:{cmd_port}" if address else "✓ Zenoh会话已打开")
    
    # 等待数据就绪
    if wait_data:
        start = time.time()
        while time.time() - start < timeout:
            with _state_lock:
                has_imu = bool(_imu_state)
            if has_imu:
                print("✓ 数据流已就绪")
                break
            time.sleep(0.01)
        else:
            print("⚠ 等待数据超时，继续运行")


def disconnect():
    """断开连接"""
    global _session, _server_locator, _sub_motor, _sub_imu, _motor_state, _imu_state, _registered_devices
    
    # 失能所有已注册的设备
    if _session and _registered_devices:
        try:
            for device_type in _registered_devices:
                if device_type != "Taks-T1-imu":  # IMU无需失能
                    _send_cmd(device_type, "disable_all")
                    print(f"✓ 发送失能命令: {device_type}")
            time.sleep(0.5)
        except Exception as e:
            print(f"✗ 失能命令发送失败: {e}")
    
    if _sub_motor:
        _sub_motor.undeclare()
        _sub_motor = None
    if _sub_imu:
        _sub_imu.undeclare()
        _sub_imu = None
    if _session:
        _session.close()
        _session = None
    
    _server_locator = None
    _motor_state = {}
    _imu_state = {}
    _registered_devices = []


def _send_cmd(device: str, cmd: str, payload: dict = None):
    """发送命令（Pub模式，不等待响应）"""
    if not _session:
        raise RuntimeError("未连接，请先调用 connect()")
    
    key = f"taks/cmd/{device}/{cmd}"
    data = _serialize_msg(payload) if payload else b""
    _session.put(key, data)


# ============ 同步读取函数 ============
def sync_get_all(robot: "TaksDevice", imu: "IMUDevice") -> Tuple[Optional[Dict], Optional[Dict], float]:
    """
    同步读取IMU和电机状态（从缓存获取，保证时间步统一）
    返回: (motor_state, imu_data, timestamp)
    """
    with _state_lock:
        # 只返回该设备关心的关节
        motor = None
        if _motor_state:
            motor = {jid: _motor_state[jid] for jid in robot.joints if jid in _motor_state}
            if not motor:
                motor = None
        imu_data = dict(_imu_state) if _imu_state else None
        ts = max(_motor_ts, _imu_ts) if _motor_ts or _imu_ts else time.perf_counter()
    return motor, imu_data, ts


def sync_get_state_only(robot: "TaksDevice") -> Tuple[Optional[Dict], float]:
    """同步读取电机状态"""
    with _state_lock:
        motor = None
        if _motor_state:
            motor = {jid: _motor_state[jid] for jid in robot.joints if jid in _motor_state}
            if not motor:
                motor = None
        ts = _motor_ts if _motor_ts else time.perf_counter()
    return motor, ts


def sync_get_imu_only(imu: "IMUDevice") -> Tuple[Optional[Dict], float]:
    """同步读取IMU数据"""
    with _state_lock:
        imu_data = dict(_imu_state) if _imu_state else None
        ts = _imu_ts if _imu_ts else time.perf_counter()
    return imu_data, ts


# ============ 设备类 ============
class TaksDevice:
    """Taks电机设备"""
    
    JOINT_MAP = {
        "Taks-T1": [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22, 23,24,25,26,27,28, 29,30,31,32,33,34],
        "Taks-T1-leftarm": list(range(9, 16)),
        "Taks-T1-rightarm": list(range(1, 8)),
        "Taks-T1-semibody": [1,2,3,4,5,6,7, 9,10,11,12,13,14,15, 17,18,19, 20,21,22],
        "Taks-T1-rightgripper": [8],
        "Taks-T1-leftgripper": [16],
    }
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.joints = self.JOINT_MAP.get(device_type, [])
        self._joint_objs: Dict[int, "TaksDevice._JointProxy"] = {}
    
    class _JointProxy:
        def __init__(self, device: "TaksDevice", jid: int):
            self._device = device
            self._jid = jid
        
        def SetPosition(self, pos: float):
            self._device.SetPosition(**{f'j{self._jid}': pos})
        
        def GetPosition(self):
            state = self._device.GetState()
            return state[self._jid]['pos'] if state and self._jid in state else None
        
        def GetVelocity(self):
            state = self._device.GetState()
            return state[self._jid]['vel'] if state and self._jid in state else None
        
        def GetTorque(self):
            state = self._device.GetState()
            return state[self._jid]['tau'] if state and self._jid in state else None
        
        def controlMIT(self, *, q=None, dq=None, tau=None, kp=None, kd=None):
            cmd = {k: v for k, v in [('q', q), ('dq', dq), ('tau', tau), ('kp', kp), ('kd', kd)] if v is not None}
            if cmd:
                self._device.controlMIT({self._jid: cmd})
    
    def __getattr__(self, name: str):
        if name.startswith('j') and name[1:].isdigit():
            jid = int(name[1:])
            if jid in self.joints:
                if jid not in self._joint_objs:
                    self._joint_objs[jid] = self._JointProxy(self, jid)
                return self._joint_objs[jid]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def _register(self):
        _send_cmd(self.device_type, "register")
        time.sleep(2.0)  # 等待注册完成
    
    def GetState(self) -> Optional[Dict[int, Dict]]:
        """获取所有关节状态（从缓存）"""
        with _state_lock:
            if _motor_state:
                return {jid: _motor_state[jid] for jid in self.joints if jid in _motor_state}
        return None
    
    def GetPosition(self) -> Optional[Dict[int, float]]:
        state = self.GetState()
        return {jid: s['pos'] for jid, s in state.items()} if state else None
    
    def SetPosition(self, **kwargs):
        joints = {int(k[1:]): v for k, v in kwargs.items() if k.startswith('j') and v is not None}
        if joints:
            _send_cmd(self.device_type, "pos", {'joints': joints})
    
    def controlMIT(self, joints: dict):
        _send_cmd(self.device_type, "mit", {'joints': joints})


class IMUDevice:
    """IMU设备"""
    
    def __init__(self):
        self.device_type = "Taks-T1-imu"
    
    def _register(self):
        _send_cmd(self.device_type, "register")
        time.sleep(2.0)
    
    def get_all(self) -> Optional[Dict]:
        with _state_lock:
            return dict(_imu_state) if _imu_state else None
    
    def get_ang_vel(self) -> Optional[Dict]:
        data = self.get_all()
        return data.get('ang_vel') if data else None
    
    def get_lin_acc(self) -> Optional[Dict]:
        data = self.get_all()
        return data.get('lin_acc') if data else None
    
    def get_quat(self) -> Optional[Dict]:
        data = self.get_all()
        return data.get('quat') if data else None
    
    def get_rpy(self) -> Optional[Dict]:
        data = self.get_all()
        return data.get('rpy') if data else None


class GripperDevice:
    """夹爪设备"""
    
    def __init__(self, is_left: bool = False):
        """
        初始化夹爪设备
        :param is_left: True=左手夹爪, False=右手匹爪
        """
        self.is_left = is_left
        self.gripper_id = LEFT_GRIPPER_ID if is_left else RIGHT_GRIPPER_ID
        self.device_type = "Taks-T1-leftgripper" if is_left else "Taks-T1-rightgripper"
    
    def _register(self):
        _send_cmd(self.device_type, "register")
        time.sleep(2.0)
    
    def SetOC(self, close: bool):
        """
        设置夹爪开合状态
        :param close: True=闭合(1), False=打开(0)
        """
        _send_cmd(self.device_type, "gripper_oc", {
            'gripper_id': self.gripper_id,
            'close': close
        })
    
    def SetPosition(self, percent: float):
        """
        设置夹爪位置百分比
        :param percent: 0-100, 0=完全打开, 100=完全闭合
        """
        _send_cmd(self.device_type, "gripper_pos", {
            'gripper_id': self.gripper_id,
            'percent': percent
        })
    
    def controlMIT(self, percent: float, kp: float = None, kd: float = None):
        """
        MIT控制模式
        :param percent: 0-100位置百分比, 0=完全打开, 100=完全闭合
        :param kp: 可选，自定义kp值
        :param kd: 可选，自定义kd值
        """
        payload = {
            'gripper_id': self.gripper_id,
            'percent': percent
        }
        if kp is not None:
            payload['kp'] = kp
        if kd is not None:
            payload['kd'] = kd
        _send_cmd(self.device_type, "gripper_mit", payload)
    
    def open(self):
        """打开匹爪"""
        self.SetOC(False)
    
    def close(self):
        """闭合匹爪"""
        self.SetOC(True)


# ============ 全局函数 ============
def register(device_type: str):
    """注册设备"""
    global _registered_devices
    if device_type == "Taks-T1-imu":
        dev = IMUDevice()
    elif device_type == "Taks-T1-rightgripper":
        dev = GripperDevice(is_left=False)
    elif device_type == "Taks-T1-leftgripper":
        dev = GripperDevice(is_left=True)
    else:
        dev = TaksDevice(device_type)
    dev._register()
    if device_type not in _registered_devices:
        _registered_devices.append(device_type)
    return dev