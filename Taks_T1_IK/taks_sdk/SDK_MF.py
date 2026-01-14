#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taks SDK 服务端 - Zenoh Pub/Sub版本 (CAN FD版)
使用Zenoh Pub/Sub模式实现低延迟通信
服务端持续发布IMU和电机状态，客户端订阅获取
所有电机通过地瓜X5的can0接口通信
"""

import zenoh
import sys
import os
import time
import signal
import threading
import pyarrow as pa
from threading import Lock, Thread, Event
from queue import Queue, Empty, Full
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
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

from libs.drivers.DM_CAN_FD import Motor as DM_Motor, MotorControlFD as DM_MotorControl, DM_Motor_Type, Control_Type
from libs.drivers.DM_IMU import DM_IMU
from libs.drivers.ankle_kinematics import ankle_ik, ankle_fk, motor_vel_to_ankle_vel, motor_tau_to_ankle_tau

# ============ 夹爪配置 ============
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.05
RIGHT_GRIPPER_DIRECTION = 1
LEFT_GRIPPER_DIRECTION = 1
RIGHT_GRIPPER_ID = 8
LEFT_GRIPPER_ID = 16

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

ALL_JOINT_IDS = list(range(1, 35))

def get_can_for_joint(jid: int) -> Optional[str]:
    for can, joints in CAN_JOINT_MAP.items():
        if jid in joints:
            return can
    return None


# ============ IMU Worker (独立进程) ============
class IMUWorker:
    def __init__(self, path: str, state_q: MPQueue, baud: int, init_evt: MPEvent):
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
                time.sleep(0.001)
            except Exception as e:
                print(f"IMU错误: {e}")
                time.sleep(0.01)


# ============ Zenoh Pub/Sub服务器 ============
class MotorServer:
    def __init__(self, imu_path: str = '/dev/imu', can_interface: str = 'can1', imu_baudrate: int = 921600):
        self.running = True
        self.closed = False
        self.can_interface = can_interface
        
        # 电机控制器和电机对象
        self.dm_mc = None
        self.motors: Dict[int, DM_Motor] = {}
        self.registered = False
        self.motor_lock = Lock()
        
        # 命令队列
        self.cmd_queue = Queue(maxsize=100)
        
        # 状态缓存
        self.motor_state = {}
        self.imu_state = {}
        self.state_lock = Lock()
        
        # IMU进程
        self.imu_state_queue = MPQueue(maxsize=100)
        self.imu_process = None
        if imu_path:
            evt = MPEvent()
            self.imu_process = Process(target=self._run_imu, args=(imu_path, self.imu_state_queue, imu_baudrate, evt), daemon=True)
            self.imu_process.start()
            evt.wait(timeout=5.0)
        
        # 初始化CAN FD控制器
        try:
            self.dm_mc = DM_MotorControl(can_interface=can_interface)
            print(f"✓ CAN FD初始化: {can_interface}")
        except Exception as e:
            print(f"✗ CAN FD初始化失败: {e}")
            raise

        # 初始化Zenoh
        config = zenoh.Config()
        self.session = zenoh.open(config)
        self.pub_motor = self.session.declare_publisher("taks/state/motor")
        self.pub_imu = self.session.declare_publisher("taks/state/imu")
        self.sub_cmd = self.session.declare_subscriber("taks/cmd/**", self._handle_cmd)
        
        # 启动工作线程
        self.cmd_thread = Thread(target=self._process_commands, daemon=True)
        self.cmd_thread.start()
        
        self.query_thread = Thread(target=self._query_motors, daemon=True)
        self.query_thread.start()
        
        self.imu_collect_thread = Thread(target=self._collect_imu_state, daemon=True)
        self.imu_collect_thread.start()
        
        self.publish_thread = Thread(target=self._publish_state, daemon=True)
        self.publish_thread.start()

        print(f"\n{'='*50}\n✓ 服务器启动 (CAN FD: {can_interface})\n{'='*50}\n")

    def _run_imu(self, path, state_q, baud, evt):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        IMUWorker(path, state_q, baud, evt).run()

    def _register_motors(self, joint_ids: List[int]):
        """注册并使能指定的电机 (多线程并行使能)"""
        import concurrent.futures
        
        start_time = time.perf_counter()
        
        # 第一步: 串行添加电机对象 (必须串行，因为 addMotor 修改共享状态)
        motors_to_enable = []
        with self.motor_lock:
            for jid in joint_ids:
                cfg = JOINT_CONFIG.get(jid)
                if not cfg:
                    continue
                if jid in self.motors:
                    motors_to_enable.append((jid, self.motors[jid], True))  # (jid, motor, is_re_enable)
                else:
                    motor = DM_Motor(cfg[0], jid, jid + 0x80)
                    self.dm_mc.addMotor(motor)
                    self.motors[jid] = motor
                    motors_to_enable.append((jid, motor, False))
        
        # 第二步: 并行使能电机
        success_count = 0
        fail_count = 0
        
        def enable_motor(args):
            jid, motor, is_re_enable = args
            try:
                self.dm_mc.switchControlMode(motor, Control_Type.MIT)
                self.dm_mc.enable(motor)
                return (jid, True, is_re_enable)
            except Exception as e:
                return (jid, False, str(e))
        
        # 使用线程池并行使能
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(motors_to_enable), 8)) as executor:
            results = list(executor.map(enable_motor, motors_to_enable))
        
        for result in results:
            jid, ok, info = result
            if ok:
                success_count += 1
                status = "重新使能" if info else "使能成功"
                print(f"✓ 电机 J{jid} {status}")
            else:
                fail_count += 1
                print(f"✗ 电机使能失败 J{jid}: {info}")
        
        self.registered = success_count > 0
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"✓ 共注册 {success_count} 个电机 (耗时 {elapsed:.1f}ms)")
        if fail_count > 0:
            print(f"✗ {fail_count} 个电机使能失败")

    def _disable_motors(self, joint_ids: List[int] = None):
        """失能指定电机 (多线程并行失能)"""
        import concurrent.futures
        
        start_time = time.perf_counter()
        
        with self.motor_lock:
            if joint_ids is None:
                joint_ids = list(self.motors.keys())
            motors_to_disable = [(jid, self.motors[jid]) for jid in joint_ids if jid in self.motors]
        
        def disable_motor(args):
            jid, motor = args
            try:
                self.dm_mc.controlMIT(motor, 0, 0, 0, 0, 0)
                time.sleep(0.01)
                self.dm_mc.disable(motor)
                return (jid, True)
            except:
                return (jid, False)
        
        # 使用线程池并行失能
        if motors_to_disable:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(motors_to_disable), 8)) as executor:
                list(executor.map(disable_motor, motors_to_disable))
        
        with self.motor_lock:
            if set(joint_ids) == set(self.motors.keys()):
                self.motors.clear()
                self.registered = False
        
        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"✓ 电机已失能 (耗时 {elapsed:.1f}ms)")

    def _get_gripper_direction(self, gripper_id: int) -> int:
        if gripper_id == LEFT_GRIPPER_ID:
            return LEFT_GRIPPER_DIRECTION
        return RIGHT_GRIPPER_DIRECTION

    def _gripper_percent_to_pos(self, percent: float, gripper_id: int) -> float:
        percent = max(0, min(100, percent))
        pos = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (percent / 100.0)
        return pos * self._get_gripper_direction(gripper_id)

    def _process_commands(self):
        """处理命令队列"""
        while self.running:
            try:
                msg = self.cmd_queue.get(timeout=0.01)
                cmd = msg.get('cmd', '')
                
                if cmd == 'register':
                    self._register_motors(msg.get('joint_ids', ALL_JOINT_IDS))
                elif cmd == 'disable_all':
                    self._disable_motors()
                elif cmd == 'mit':
                    self._mit(msg.get('joints', {}))
                elif cmd == 'pos':
                    self._pos(msg.get('joints', {}))
                elif cmd == 'gripper_oc':
                    self._gripper_oc(msg.get('gripper_id'), msg.get('close', False))
                elif cmd == 'gripper_pos':
                    self._gripper_pos(msg.get('gripper_id'), msg.get('percent', 0))
                elif cmd == 'gripper_mit':
                    self._gripper_mit(msg.get('gripper_id'), msg.get('percent', 0), msg.get('kp'), msg.get('kd'))
            except Empty:
                pass
            except Exception as e:
                print(f"命令处理错误: {e}")

    def _mit(self, joints: dict):
        """MIT控制 (多线程并行发送)"""
        import concurrent.futures
        
        # 准备控制数据
        control_tasks = []
        with self.motor_lock:
            for jid, p in joints.items():
                jid = int(jid)
                if jid not in self.motors:
                    continue
                motor = self.motors[jid]
                cfg = JOINT_CONFIG.get(jid, (None, 5.0, 1.0))
                q = p.get('q', 0)
                kp = p.get('kp') if p.get('kp') is not None else cfg[1]
                kd = p.get('kd') if p.get('kd') is not None else cfg[2]
                control_tasks.append((motor, kp, kd, q, p.get('dq', 0), p.get('tau', 0)))
        
        # 并行发送控制命令
        def send_mit(args):
            motor, kp, kd, q, dq, tau = args
            self.dm_mc.controlMIT(motor, kp, kd, q, dq, tau)
        
        if control_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(control_tasks), 8)) as executor:
                list(executor.map(send_mit, control_tasks))

    def _pos(self, joints: dict):
        """位置控制 (多线程并行发送)"""
        import concurrent.futures
        
        # 准备控制数据
        control_tasks = []
        with self.motor_lock:
            for jid, val in joints.items():
                jid = int(jid)
                if jid not in self.motors:
                    continue
                motor = self.motors[jid]
                cfg = JOINT_CONFIG.get(jid, (None, 5.0, 1.0))
                control_tasks.append((motor, cfg[1], cfg[2], val))
        
        # 并行发送控制命令
        def send_pos(args):
            motor, kp, kd, pos = args
            self.dm_mc.controlMIT(motor, kp, kd, pos, 0, 0)
        
        if control_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(control_tasks), 8)) as executor:
                list(executor.map(send_pos, control_tasks))

    def _gripper_oc(self, gripper_id: int, close: bool):
        with self.motor_lock:
            if gripper_id not in self.motors:
                return
            motor = self.motors[gripper_id]
            cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
            pos = GRIPPER_CLOSE if close else GRIPPER_OPEN
            actual_pos = pos * self._get_gripper_direction(gripper_id)
            self.dm_mc.controlMIT(motor, cfg[1], cfg[2], actual_pos, 0, 0)

    def _gripper_pos(self, gripper_id: int, percent: float):
        with self.motor_lock:
            if gripper_id not in self.motors:
                return
            motor = self.motors[gripper_id]
            cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
            actual_pos = self._gripper_percent_to_pos(percent, gripper_id)
            self.dm_mc.controlMIT(motor, cfg[1], cfg[2], actual_pos, 0, 0)

    def _gripper_mit(self, gripper_id: int, percent: float, kp: float = None, kd: float = None):
        with self.motor_lock:
            if gripper_id not in self.motors:
                return
            motor = self.motors[gripper_id]
            cfg = JOINT_CONFIG.get(gripper_id, (None, 0.5, 0.05))
            use_kp = kp if kp is not None else cfg[1]
            use_kd = kd if kd is not None else cfg[2]
            actual_pos = self._gripper_percent_to_pos(percent, gripper_id)
            self.dm_mc.controlMIT(motor, use_kp, use_kd, actual_pos, 0, 0)

    def _query_motors(self):
        """定期查询电机状态 (多线程并行查询)"""
        import concurrent.futures
        
        query_interval = 0.001
        while self.running:
            try:
                if not self.registered:
                    time.sleep(0.01)
                    continue
                
                # 获取电机列表快照
                with self.motor_lock:
                    motors_snapshot = list(self.motors.items())
                
                # 并行查询电机状态
                def query_motor(args):
                    jid, motor = args
                    try:
                        self.dm_mc.refresh_motor_status(motor)
                        return (str(jid), {
                            'pos': float(motor.getPosition() or 0),
                            'vel': float(motor.getVelocity() or 0),
                            'tau': float(motor.getTorque() or 0)
                        })
                    except:
                        return (str(jid), {'pos': 0, 'vel': 0, 'tau': 0})
                
                if motors_snapshot:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(motors_snapshot), 8)) as executor:
                        results = list(executor.map(query_motor, motors_snapshot))
                    
                    result = dict(results)
                    with self.state_lock:
                        self.motor_state.update(result)
                
                time.sleep(query_interval)
            except Exception as e:
                print(f"电机查询错误: {e}")
                time.sleep(0.01)

    def _collect_imu_state(self):
        """收集IMU状态"""
        while self.running:
            try:
                src, data = self.imu_state_queue.get(timeout=0.01)
                if src == 'imu':
                    with self.state_lock:
                        self.imu_state = data
            except Empty:
                pass
            except Exception as e:
                print(f"IMU状态收集错误: {e}")

    def _publish_state(self):
        """发布状态到Zenoh"""
        publish_interval = 0.001
        while self.running:
            try:
                with self.state_lock:
                    motor_data = dict(self.motor_state)
                    imu_data = dict(self.imu_state)
                
                if motor_data:
                    self.pub_motor.put(_serialize_msg({
                        'ts': time.perf_counter(),
                        'joints': motor_data
                    }))
                
                if imu_data:
                    self.pub_imu.put(_serialize_msg(imu_data))
                
                time.sleep(publish_interval)
            except Exception as e:
                print(f"状态发布错误: {e}")
                time.sleep(0.01)

    def _handle_cmd(self, sample):
        """处理命令订阅"""
        try:
            key = str(sample.key_expr)
            payload = sample.payload.to_bytes()
            msg = _deserialize_msg(payload) if payload else {}
            
            parts = key.split('/')
            if len(parts) < 4:
                return
            
            device = parts[2]
            cmd = parts[3]
            
            # 确定目标关节
            if device == 'Taks-T1':
                target_joints = ALL_JOINT_IDS
            elif device == 'Taks-T1-semibody':
                target_joints = CAN_JOINT_MAP['right_hand'] + CAN_JOINT_MAP['left_hand'] + CAN_JOINT_MAP['waist_neck']
            elif device == 'Taks-T1-rightgripper':
                target_joints = [RIGHT_GRIPPER_ID]
            elif device == 'Taks-T1-leftgripper':
                target_joints = [LEFT_GRIPPER_ID]
            elif device == 'Taks-T1-imu':
                if cmd == 'register':
                    print(f"✓ 客户端注册IMU设备")
                return
            else:
                target_joints = []
            
            if cmd == 'register':
                self._safe_queue_put({'cmd': 'register', 'joint_ids': target_joints}, priority=True)
                print(f"✓ 客户端注册设备: {device}")
            elif cmd == 'disable_all':
                self._safe_queue_put({'cmd': 'disable_all'}, priority=True)
            elif cmd in ('mit', 'pos'):
                joints = msg.get('joints', {})
                joints = {int(k): v for k, v in joints.items()}
                # 过滤只保留目标关节
                filtered = {k: v for k, v in joints.items() if k in target_joints or not target_joints}
                self._safe_queue_put({'cmd': cmd, 'joints': filtered})
            elif cmd == 'gripper_oc':
                gripper_id = msg.get('gripper_id')
                close = msg.get('close', False)
                self._safe_queue_put({'cmd': 'gripper_oc', 'gripper_id': gripper_id, 'close': close})
            elif cmd == 'gripper_pos':
                gripper_id = msg.get('gripper_id')
                percent = msg.get('percent', 0)
                self._safe_queue_put({'cmd': 'gripper_pos', 'gripper_id': gripper_id, 'percent': percent})
            elif cmd == 'gripper_mit':
                gripper_id = msg.get('gripper_id')
                percent = msg.get('percent', 0)
                kp = msg.get('kp')
                kd = msg.get('kd')
                self._safe_queue_put({'cmd': 'gripper_mit', 'gripper_id': gripper_id, 'percent': percent, 'kp': kp, 'kd': kd})
                            
        except Exception as e:
            print(f"命令处理错误: {e}")

    def _safe_queue_put(self, item, priority=False):
        """安全放入队列"""
        try:
            self.cmd_queue.put_nowait(item)
            return True
        except Full:
            if priority:
                try:
                    self.cmd_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.cmd_queue.put_nowait(item)
                    return True
                except Full:
                    return False
            return False

    def run(self):
        """运行服务器"""
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n✓ 收到中断")
        except SystemExit:
            pass
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
        
        # 失能所有电机
        self._disable_motors()
        
        # 关闭CAN控制器
        if self.dm_mc:
            try:
                self.dm_mc.close()
            except:
                pass
        
        # 终止IMU进程
        if self.imu_process:
            self.imu_process.terminate()
            self.imu_process.join(timeout=1)
        
        # 关闭Zenoh
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
    server = MotorServer(imu_path='/dev/imu', can_interface='can1', imu_baudrate=921600)

    def sig_handler(signum, frame):
        server.close()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    server.run()


if __name__ == "__main__":
    run_motor_server()