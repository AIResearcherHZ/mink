#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK 服务端 - Zenoh Pub/Sub版本
使用Zenoh Pub/Sub模式实现低延迟通信
服务端持续发布IMU和电机状态，客户端订阅获取
"""

import zenoh
import sys
import os
import time
import signal
import serial
import threading
import pyarrow as pa
from multiprocessing import Process, Queue, Event
from threading import Lock
from queue import Empty, Full
from typing import Dict, List, Optional

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
    arrays = []
    names = []
    for k, v in msg.items():
        names.append(k)
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libs.drivers.DM_Motor import Motor as DM_Motor, MotorControl as DM_MotorControl, DM_Motor_Type, Control_Type
from libs.drivers.DM_IMU import DM_IMU
from libs.drivers.ankle_kinematics import ankle_ik, ankle_fk, motor_vel_to_ankle_vel, motor_tau_to_ankle_tau

# ============ 夹爪配置 ============
# 夹爪位置定义 (电机弧度位置)
GRIPPER_OPEN = 0.0       # 打开位置
GRIPPER_CLOSE = 1.05      # 闭合位置

# 左右手夹爪镜像配置 (预留，目前默认都一样)
RIGHT_GRIPPER_DIRECTION = 1  # 右手夹爪方向系数
LEFT_GRIPPER_DIRECTION = 1   # 左手夹爪方向系数 (如需镜像改为-1)

# 夹爪关节ID
RIGHT_GRIPPER_ID = 8   # 右手夹爪
LEFT_GRIPPER_ID = 16   # 左手夹爪

# ============ 配置 ============
JOINT_CONFIG = {
    1: (DM_Motor_Type.DM4340, 20.0, 2.0), 2: (DM_Motor_Type.DM4340, 20.0, 2.0),
    3: (DM_Motor_Type.DM4340, 20.0, 2.0), 4: (DM_Motor_Type.DM4340, 20.0, 2.0),
    5: (DM_Motor_Type.DM4310, 10.0, 1.0), 6: (DM_Motor_Type.DM4310, 10.0, 1.0),
    7: (DM_Motor_Type.DM4310, 10.0, 1.0), 8: (DM_Motor_Type.DM3507, 1.5, 0.1),
    9: (DM_Motor_Type.DM4340, 20.0, 2.0), 10: (DM_Motor_Type.DM4340, 20.0, 2.0),
    11: (DM_Motor_Type.DM4340, 20.0, 2.0), 12: (DM_Motor_Type.DM4340, 20.0, 2.0),
    13: (DM_Motor_Type.DM4310, 10.0, 1.0), 14: (DM_Motor_Type.DM4310, 10.0, 1.0),
    15: (DM_Motor_Type.DM4310, 10.0, 1.0), 16: (DM_Motor_Type.DM3507, 1.5, 0.1),
    17: (DM_Motor_Type.DM6248P, 250.0, 5.0), 18: (DM_Motor_Type.DM6248P, 250.0, 5.0),
    19: (DM_Motor_Type.DM6248P, 250.0, 5.0), 20: (DM_Motor_Type.DM3507, 1.0, 0.5),
    21: (DM_Motor_Type.DM3507, 1.0, 0.5), 22: (DM_Motor_Type.DM3507, 1.0, 0.5),
    23: (DM_Motor_Type.DM10010L, 300.0, 8.0), 24: (DM_Motor_Type.DM6248P, 250.0, 5.0),
    25: (DM_Motor_Type.DM6248P, 250.0, 5.0), 26: (DM_Motor_Type.DM10010L, 300.0, 8.0),
    27: (DM_Motor_Type.DM4340, 20.0, 2.0), 28: (DM_Motor_Type.DM4340, 20.0, 2.0),
    29: (DM_Motor_Type.DM10010L, 300.0, 8.0), 30: (DM_Motor_Type.DM6248P, 250.0, 5.0),
    31: (DM_Motor_Type.DM6248P, 250.0, 5.0), 32: (DM_Motor_Type.DM10010L, 300.0, 8.0),
    33: (DM_Motor_Type.DM4340, 20.0, 2.0), 34: (DM_Motor_Type.DM4340, 20.0, 2.0),
}

ANKLE_PAIRS = {27: (27, 28), 33: (33, 34)}

CAN_JOINT_MAP = {
    'right_hand': [1, 2, 3, 4, 5, 6, 7, 8],
    'left_hand': [9, 10, 11, 12, 13, 14, 15, 16],
    'waist_neck': [17, 18, 19, 20, 21, 22],
    'right_leg': [23, 24, 25, 26, 27, 28],
    'left_leg': [29, 30, 31, 32, 33, 34],
}

def get_can_for_joint(jid: int) -> Optional[str]:
    for can, joints in CAN_JOINT_MAP.items():
        if jid in joints:
            return can
    return None


# ============ IMU Worker ============
class IMUWorker:
    def __init__(self, path: str, state_q: Queue, baud: int, init_evt: Event):
        self.path, self.state_q, self.baud = path, state_q, baud
        self.init_evt = init_evt
        self.imu = None

    def run(self):
        try:
            self.imu = DM_IMU(port=self.path, baudrate=self.baud)
            self.imu.start()
            print(f"✓ IMU初始化: {self.path}")
        except Exception as e:
            print(f"✗ IMU初始化失败: {e}")
        self.init_evt.set()

        # 持续读取IMU数据并放入队列
        while True:
            try:
                if self.imu:
                    e = self.imu.get_euler()
                    data = {
                        'ang_vel': self.imu.get_gyro(),
                        'lin_acc': self.imu.get_accel(),
                        'quat': self.imu.get_quat(),
                        'rpy': {'roll': e['roll'], 'pitch': e['pitch'], 'yaw': e['yaw']},
                        'ts': time.perf_counter()
                    }
                    try:
                        self.state_q.put_nowait(('imu', data))
                    except Full:
                        try:
                            self.state_q.get_nowait()
                        except Empty:
                            pass
                        try:
                            self.state_q.put_nowait(('imu', data))
                        except:
                            pass
                time.sleep(0.001)  # 1000Hz采样
            except Exception as e:
                print(f"IMU错误: {e}")
                time.sleep(0.01)


# ============ CAN Worker ============
class CANWorker:
    def __init__(self, name: str, path: str, cmd_q: Queue, state_q: Queue, baud: int, init_evt: Event):
        self.name, self.path, self.cmd_q, self.state_q, self.baud = name, path, cmd_q, state_q, baud
        self.init_evt = init_evt
        self.motors: Dict[int, tuple] = {}
        self.dm_mc = None
        self.lock = Lock()
        self.registered = False
        # MIT命令覆盖缓存
        self._latest_mit_cmd: Optional[Dict] = None
        # 夹爪命令覆盖缓存
        self._latest_gripper_cmd: Dict[int, Dict] = {}  # {gripper_id: cmd_dict}
        # 复用线程池
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def run(self):
        try:
            ser = serial.Serial(port=self.path, baudrate=self.baud, timeout=0.01)
            self.dm_mc = DM_MotorControl(ser)
            print(f"✓ CAN初始化: {self.name}")
        except Exception as e:
            print(f"✗ CAN初始化失败 {self.name}: {e}")
        self.init_evt.set()

        last_query_time = 0
        query_interval = 0.001  # 1000Hz查询

        while True:
            try:
                # 1. 处理队列命令(批量取出，MIT命令覆盖)
                while True:
                    try:
                        msg = self.cmd_q.get_nowait()
                        cmd = msg.get('cmd', '')
                        if cmd == 'shutdown':
                            self._disable_all()
                            return
                        elif cmd == 'disable_all':
                            self._disable_all()
                        elif cmd == 'register':
                            self._register_motors()
                        elif cmd == 'mit':
                            # MIT命令覆盖策略
                            joints = msg.get('joints', {})
                            if self._latest_mit_cmd is None:
                                self._latest_mit_cmd = joints
                            else:
                                self._latest_mit_cmd.update(joints)
                        elif cmd == 'pos':
                            self._pos(msg.get('joints', {}))
                        elif cmd == 'gripper_oc':
                            # 夹爪命令覆盖策略
                            gripper_id = msg.get('gripper_id')
                            self._latest_gripper_cmd[gripper_id] = {'type': 'oc', 'close': msg.get('close', False)}
                        elif cmd == 'gripper_pos':
                            gripper_id = msg.get('gripper_id')
                            self._latest_gripper_cmd[gripper_id] = {'type': 'pos', 'percent': msg.get('percent', 0)}
                        elif cmd == 'gripper_mit':
                            gripper_id = msg.get('gripper_id')
                            self._latest_gripper_cmd[gripper_id] = {'type': 'mit', 'percent': msg.get('percent', 0), 'kp': msg.get('kp'), 'kd': msg.get('kd')}
                    except Empty:
                        break
                
                # 2. 执行最新MIT命令
                if self._latest_mit_cmd:
                    self._mit(self._latest_mit_cmd)
                    self._latest_mit_cmd = None
                
                # 3. 执行最新夹爪命令
                if self._latest_gripper_cmd:
                    gripper_cmds = dict(self._latest_gripper_cmd)
                    self._latest_gripper_cmd.clear()
                    for gripper_id, cmd_data in gripper_cmds.items():
                        cmd_type = cmd_data.get('type')
                        if cmd_type == 'oc':
                            self._gripper_oc(gripper_id, cmd_data.get('close', False))
                        elif cmd_type == 'pos':
                            self._gripper_pos(gripper_id, cmd_data.get('percent', 0))
                        elif cmd_type == 'mit':
                            self._gripper_mit(gripper_id, cmd_data.get('percent', 0), cmd_data.get('kp'), cmd_data.get('kd'))
                
                # 4. 定期查询电机状态
                now = time.perf_counter()
                if self.registered and now - last_query_time >= query_interval:
                    result = self._query([])
                    if result.get('ok') and result.get('joints'):
                        try:
                            self.state_q.put_nowait((self.name, result['joints']))
                        except Full:
                            try:
                                self.state_q.get_nowait()
                            except Empty:
                                pass
                            try:
                                self.state_q.put_nowait((self.name, result['joints']))
                            except:
                                pass
                    last_query_time = now
                else:
                    time.sleep(0.001) 
                    
            except Exception as e:
                print(f"CAN错误 {self.name}: {e}")

    def _register_motors(self):
        if not self.dm_mc:
            return
        for jid in CAN_JOINT_MAP.get(self.name, []):
            cfg = JOINT_CONFIG.get(jid)
            if not cfg:
                continue
            if jid in self.motors:
                motor, mc = self.motors[jid]
                try:
                    mc.switchControlMode(motor, Control_Type.MIT)
                    mc.enable(motor)
                    print(f"✓ 电机 J{jid} 重新使能")
                except Exception as e:
                    print(f"电机重新使能失败 J{jid}: {e}")
                continue
            motor = DM_Motor(cfg[0], jid, jid + 0x80)
            self.dm_mc.addMotor(motor)
            try:
                self.dm_mc.switchControlMode(motor, Control_Type.MIT)
                self.dm_mc.enable(motor)
                self.motors[jid] = (motor, self.dm_mc)
            except Exception as e:
                print(f"电机使能失败 J{jid}: {e}")
        self.registered = True
        print(f"✓ {self.name} 注册 {len(self.motors)} 个电机")

    def _query(self, jids: list) -> dict:
        with self.lock:
            if not jids:
                jids = list(self.motors.keys())
            motors_to_query = [(jid, self.motors[jid]) for jid in jids if jid in self.motors]
        
        def query_motor(args):
            jid, (motor, mc) = args
            try:
                mc.refresh_motor_status(motor)
                mc.recv()
                return (str(jid), {
                    'pos': float(motor.getPosition() or 0),
                    'vel': float(motor.getVelocity() or 0),
                    'tau': float(motor.getTorque() or 0)
                })
            except:
                return (str(jid), {'pos': 0, 'vel': 0, 'tau': 0})
        
        if motors_to_query:
            results = list(self._executor.map(query_motor, motors_to_query))
            result = dict(results)
        else:
            result = {}
        
        return {'ok': True, 'joints': result}

    def _mit(self, joints: dict):
        control_tasks = []
        with self.lock:
            for jid, p in joints.items():
                jid = int(jid)
                if jid not in self.motors:
                    continue
                motor, mc = self.motors[jid]
                cfg = JOINT_CONFIG.get(jid, (None, 5.0, 1.0))
                q = p.get('q', 0)
                kp = p.get('kp') if p.get('kp') is not None else cfg[1]
                kd = p.get('kd') if p.get('kd') is not None else cfg[2]
                control_tasks.append((motor, mc, kp, kd, q, p.get('dq', 0), p.get('tau', 0)))
        
        def send_mit(args):
            motor, mc, kp, kd, q, dq, tau = args
            mc.controlMIT(motor, kp, kd, q, dq, tau)
        
        if control_tasks:
            list(self._executor.map(send_mit, control_tasks))

    def _pos(self, joints: dict):
        control_tasks = []
        with self.lock:
            for jid, val in joints.items():
                jid = int(jid)
                if jid not in self.motors:
                    continue
                motor, mc = self.motors[jid]
                cfg = JOINT_CONFIG.get(jid, (None, 5.0, 1.0))
                control_tasks.append((motor, mc, cfg[1], cfg[2], val))
        
        def send_pos(args):
            motor, mc, kp, kd, pos = args
            mc.controlMIT(motor, kp, kd, pos, 0, 0)
        
        if control_tasks:
            list(self._executor.map(send_pos, control_tasks))

    def _disable_all(self):
        for jid, (motor, mc) in self.motors.items():
            try:
                mc.controlMIT(motor, 0, 0, 0, 0, 0)
                time.sleep(0.01)
                mc.disable(motor)
            except:
                pass
        self.registered = False
        # 关闭线程池
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        print(f"✓ {self.name} 电机已失能")

    def _get_gripper_direction(self, gripper_id: int) -> int:
        """获取夹爪方向系数"""
        if gripper_id == LEFT_GRIPPER_ID:
            return LEFT_GRIPPER_DIRECTION
        return RIGHT_GRIPPER_DIRECTION

    def _gripper_percent_to_pos(self, percent: float, gripper_id: int) -> float:
        """将百分比转换为电机位置"""
        percent = max(0, min(100, percent))
        pos = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (percent / 100.0)
        return pos * self._get_gripper_direction(gripper_id)

    def _gripper_oc(self, gripper_id: int, close: bool):
        """夹爪开合控制"""
        if gripper_id not in self.motors:
            return
        motor, mc = self.motors[gripper_id]
        cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
        pos = GRIPPER_CLOSE if close else GRIPPER_OPEN
        actual_pos = pos * self._get_gripper_direction(gripper_id)
        mc.controlMIT(motor, cfg[1], cfg[2], actual_pos, 0, 0)

    def _gripper_pos(self, gripper_id: int, percent: float):
        """夹爪位置百分比控制"""
        if gripper_id not in self.motors:
            return
        motor, mc = self.motors[gripper_id]
        cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
        actual_pos = self._gripper_percent_to_pos(percent, gripper_id)
        mc.controlMIT(motor, cfg[1], cfg[2], actual_pos, 0, 0)

    def _gripper_mit(self, gripper_id: int, percent: float, kp: float = None, kd: float = None):
        """夹爪MIT控制"""
        if gripper_id not in self.motors:
            return
        motor, mc = self.motors[gripper_id]
        cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
        use_kp = kp if kp is not None else cfg[1]
        use_kd = kd if kd is not None else cfg[2]
        actual_pos = self._gripper_percent_to_pos(percent, gripper_id)
        mc.controlMIT(motor, use_kp, use_kd, actual_pos, 0, 0)


# ============ Zenoh Pub/Sub服务器 ============
class MotorServer:
    def __init__(self, can_devices: Dict[str, str], baudrate: int = 921600):
        self.cmd_queues = {name: Queue(maxsize=100) for name in can_devices if name != 'imu'}
        self.state_queue = Queue(maxsize=1000)
        self.processes = []
        self.running = True
        self.closed = False  # 防止重复关闭
        
        # 状态缓存
        self.motor_state = {}  # {jid: {pos, vel, tau}}
        self.imu_state = {}
        self.state_lock = Lock()

        # 启动工作进程
        for name, path in can_devices.items():
            evt = Event()
            if name == 'imu':
                p = Process(target=self._run_imu, args=(path, self.state_queue, baudrate, evt), daemon=True)
            else:
                p = Process(target=self._run_can, args=(name, path, self.cmd_queues[name], self.state_queue, baudrate, evt), daemon=True)
            p.start()
            self.processes.append(p)
            evt.wait(timeout=5.0)

        # 初始化Zenoh
        config = zenoh.Config()
        self.session = zenoh.open(config)
        
        # 声明发布者
        self.pub_motor = self.session.declare_publisher("taks/state/motor")
        self.pub_imu = self.session.declare_publisher("taks/state/imu")
        
        # 声明订阅者接收命令
        self.sub_cmd = self.session.declare_subscriber("taks/cmd/**", self._handle_cmd)
        
        # 启动状态收集和发布线程
        self.collect_thread = threading.Thread(target=self._collect_state, daemon=True)
        self.collect_thread.start()
        
        self.publish_thread = threading.Thread(target=self._publish_state, daemon=True)
        self.publish_thread.start()

        print(f"\n{'='*50}\n✓ 服务器启动\n{'='*50}\n")

    def _run_imu(self, path, state_q, baud, evt):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        IMUWorker(path, state_q, baud, evt).run()

    def _run_can(self, name, path, cmd_q, state_q, baud, evt):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        CANWorker(name, path, cmd_q, state_q, baud, evt).run()

    def _safe_queue_put(self, q, item, priority=False):
        """安全放入队列，队列满时丢弃旧消息"""
        try:
            q.put_nowait(item)
            return True
        except Full:
            if priority:  # 优先级命令（如register/disable）丢弃旧消息
                try:
                    q.get_nowait()
                except Empty:
                    pass
                try:
                    q.put_nowait(item)
                    return True
                except Full:
                    return False
            return False

    def _handle_cmd(self, sample):
        """处理命令订阅"""
        try:
            key = str(sample.key_expr)
            payload = sample.payload.to_bytes()
            msg = _deserialize_msg(payload) if payload else {}
            
            # 解析: taks/cmd/{device}/{cmd}
            parts = key.split('/')
            if len(parts) < 4:
                return
            
            device = parts[2]
            cmd = parts[3]
            
            # 确定目标CAN
            if device == 'Taks-T1':
                targets = list(CAN_JOINT_MAP.keys())
            elif device == 'Taks-T1-semibody':
                targets = ['right_hand', 'left_hand', 'waist_neck']
            elif device == 'Taks-T1-rightgripper':
                targets = ['right_hand']
            elif device == 'Taks-T1-leftgripper':
                targets = ['left_hand']
            elif device == 'Taks-T1-imu':
                targets = ['imu']
            else:
                targets = []
            
            if cmd == 'register':
                for can in targets:
                    if can in self.cmd_queues:
                        self._safe_queue_put(self.cmd_queues[can], {'cmd': 'register'}, priority=True)
                # IMU无需发送register命令，自动发布
                if device == 'Taks-T1-imu':
                    print(f"✓ 客户端注册IMU设备")
                else:
                    print(f"✓ 客户端注册设备: {device}")
            
            elif cmd == 'disable_all':
                for can in targets:
                    if can in self.cmd_queues:
                        self._safe_queue_put(self.cmd_queues[can], {'cmd': 'disable_all'}, priority=True)
            
            elif cmd in ('mit', 'pos'):
                joints = msg.get('joints', {})
                joints = {int(k): v for k, v in joints.items()}
                by_can = {}
                for jid, val in joints.items():
                    can = get_can_for_joint(jid)
                    if can:
                        by_can.setdefault(can, {})[jid] = val
                
                for can, cj in by_can.items():
                    if can in self.cmd_queues:
                        # 控制命令不优先，队列满时丢弃新命令
                        self._safe_queue_put(self.cmd_queues[can], {'cmd': cmd, 'joints': cj}, priority=False)
            
            elif cmd == 'gripper_oc':
                # 夹爪开合控制: close=True闭合, close=False打开
                gripper_id = msg.get('gripper_id')
                close = msg.get('close', False)
                can = get_can_for_joint(gripper_id)
                if can and can in self.cmd_queues:
                    self._safe_queue_put(self.cmd_queues[can], {
                        'cmd': 'gripper_oc', 'gripper_id': gripper_id, 'close': close
                    }, priority=False)
            
            elif cmd == 'gripper_pos':
                # 夹爪位置百分比控制: percent=0-100
                gripper_id = msg.get('gripper_id')
                percent = msg.get('percent', 0)
                can = get_can_for_joint(gripper_id)
                if can and can in self.cmd_queues:
                    self._safe_queue_put(self.cmd_queues[can], {
                        'cmd': 'gripper_pos', 'gripper_id': gripper_id, 'percent': percent
                    }, priority=False)
            
            elif cmd == 'gripper_mit':
                # 夹爪MIT控制: percent=0-100, 可选kp/kd
                gripper_id = msg.get('gripper_id')
                percent = msg.get('percent', 0)
                kp = msg.get('kp')
                kd = msg.get('kd')
                can = get_can_for_joint(gripper_id)
                if can and can in self.cmd_queues:
                    self._safe_queue_put(self.cmd_queues[can], {
                        'cmd': 'gripper_mit', 'gripper_id': gripper_id, 'percent': percent, 'kp': kp, 'kd': kd
                    }, priority=False)
                            
        except Exception as e:
            print(f"命令处理错误: {e}")

    def _collect_state(self):
        """收集工作进程的状态"""
        while self.running:
            try:
                try:
                    src, data = self.state_queue.get(timeout=0.001)
                except Empty:
                    continue
                
                with self.state_lock:
                    if src == 'imu':
                        self.imu_state = data
                    else:
                        # 合并电机状态
                        if isinstance(data, dict):
                            self.motor_state.update(data)
                            
            except Exception as e:
                print(f"状态收集错误: {e}")

    def _publish_state(self):
        """发布状态到Zenoh"""
        publish_interval = 0.001  # 1000Hz发布
        
        while self.running:
            try:
                with self.state_lock:
                    motor_data = dict(self.motor_state)
                    imu_data = dict(self.imu_state)
                
                # 发布电机状态
                if motor_data:
                    self.pub_motor.put(_serialize_msg({
                        'ts': time.perf_counter(),
                        'joints': motor_data
                    }))
                
                # 发布IMU状态
                if imu_data:
                    self.pub_imu.put(_serialize_msg(imu_data))
                
                time.sleep(publish_interval)
                
            except Exception as e:
                print(f"状态发布错误: {e}")
                time.sleep(0.01)

    def run(self):
        """运行服务器"""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n✓ 收到中断")
        except SystemExit:
            pass  # 信号处理器已调用close
        finally:
            if not self.closed:
                self.close()

    def close(self):
        """关闭服务器"""
        if self.closed:
            return
        self.closed = True
        
        print("✓ 关闭服务器...")
        self.running = False
        
        for name, q in self.cmd_queues.items():
            try:
                q.put_nowait({'cmd': 'shutdown'})
            except:
                pass
        
        time.sleep(0.5)
        
        for p in self.processes:
            p.terminate()
            p.join(timeout=1)
        
        try:
            self.sub_cmd.undeclare()
        except:
            pass
        try:
            self.pub_motor.undeclare()
        except:
            pass
        try:
            self.pub_imu.undeclare()
        except:
            pass
        try:
            self.session.close()
        except:
            pass
        
        print("✓ 服务器已关闭")


def run_motor_server():
    can_devices = {
        'right_hand': '/dev/can_right_hand',
        'left_hand': '/dev/can_left_hand',
        'waist_neck': '/dev/can_waist_and_neck',
        'right_leg': '/dev/can_right_leg',
        'left_leg': '/dev/can_left_leg',
        'imu': '/dev/imu',
    }
    server = MotorServer(can_devices, baudrate=921600)

    def sig_handler(signum, frame):
        server.close()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    server.run()


if __name__ == "__main__":
    run_motor_server()