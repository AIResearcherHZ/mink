#!/usr/bin/env python3
"""
Taks SDK 服务端 - Zenoh Pub/Sub版本 (CAN FD版)
使用Zenoh Pub/Sub模式实现低延迟通信
服务端持续发布IMU和电机状态，客户端订阅获取
所有电机通过地瓜X5的can1接口通信 (CAN FD)
"""

import zenoh
import sys
import os
import time
import signal
import threading
from typing import Optional
from collections import deque

import pyarrow as pa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DM_CAN_FD import Motor as DM_Motor, MotorControlFD as DM_MotorControl, DM_Motor_Type, Control_Type
from DM_IMU import DM_IMU

# ==================== 序列化 ====================

def serialize(msg: dict) -> bytes:
    arrays, names = [], []
    for k, v in msg.items():
        names.append(k)
        arrays.append(pa.array([str(v)]))
    batch = pa.record_batch(arrays, names=names)
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()

def deserialize(data: bytes) -> dict:
    reader = pa.ipc.open_stream(pa.py_buffer(data))
    batch = reader.read_next_batch()
    msg = {}
    for i, name in enumerate(batch.schema.names):
        val = batch.column(i)[0].as_py()
        if val is None:
            msg[name] = None
        elif isinstance(val, str):
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
        else:
            msg[name] = val
    return msg

# ==================== 配置 ====================

GRIPPER_OPEN, GRIPPER_CLOSE = 0.0, 1.05
RIGHT_GRIPPER_ID, LEFT_GRIPPER_ID = 8, 16

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

ANKLE_MOTOR_IDS = {27, 28, 33, 34}

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


# ==================== 部位工作线程 ====================

class BodyPartWorker:
    """处理单个部位的命令"""
    
    def __init__(self, name: str, joint_ids: list, dm_mc, motors: dict, motor_lock: threading.Lock):
        self.name = name
        self.joint_ids = joint_ids
        self.dm_mc = dm_mc
        self.motors = motors
        self.motor_lock = motor_lock
        self.running = True
        self.registered = False
        
        # 命令覆盖缓存
        self._mit_cmd = None
        self._gripper_cmds = {}
        self._pending_register = False
        self._pending_disable = False
        
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def set_mit_cmd(self, joints: dict):
        if self._mit_cmd is None:
            self._mit_cmd = joints
        else:
            self._mit_cmd.update(joints)
    
    def set_gripper_cmd(self, gid: int, cmd: dict):
        self._gripper_cmds[gid] = cmd
    
    def request_register(self):
        self._pending_register = True
    
    def request_disable(self):
        self._pending_disable = True
    
    def _run(self):
        while self.running:
            try:
                if self._pending_disable:
                    self._disable_motors()
                    self._pending_disable = False
                    continue
                
                if self._pending_register:
                    self._register_motors()
                    self._pending_register = False
                
                if self._mit_cmd:
                    self._exec_mit(self._mit_cmd)
                    self._mit_cmd = None
                
                if self._gripper_cmds:
                    for gid, cmd in self._gripper_cmds.items():
                        self._exec_gripper(gid, cmd)
                    self._gripper_cmds.clear()
                
                time.sleep(0.0002)
            except Exception as e:
                print(f"{self.name} 错误: {e}")
    
    def _register_motors(self):
        for jid in self.joint_ids:
            cfg = JOINT_CONFIG.get(jid)
            if not cfg:
                continue
            with self.motor_lock:
                if jid in self.motors:
                    motor = self.motors[jid]
                    try:
                        self.dm_mc.switchControlMode(motor, Control_Type.MIT)
                        self.dm_mc.enable(motor)
                    except:
                        pass
                    continue
                motor = DM_Motor(cfg[0], jid, jid + 0x80)
                self.dm_mc.addMotor(motor)
                try:
                    self.dm_mc.switchControlMode(motor, Control_Type.MIT)
                    self.dm_mc.enable(motor)
                    self.motors[jid] = motor
                except Exception as e:
                    print(f"电机使能失败 J{jid}: {e}")
        self.registered = True
        print(f"✓ {self.name} 注册完成")
    
    def _disable_motors(self):
        with self.motor_lock:
            for jid in list(self.motors.keys()):
                if jid in self.joint_ids:
                    try:
                        motor = self.motors[jid]
                        self.dm_mc.controlMIT(motor, 0, 0, 0, 0, 0)
                        time.sleep(0.005)
                        self.dm_mc.disable(motor)
                        del self.motors[jid]
                    except:
                        pass
        self.registered = False
        print(f"✓ {self.name} 已失能")
    
    def _exec_mit(self, joints: dict):
        ankle_data = {}
        with self.motor_lock:
            for jid, p in joints.items():
                jid = int(jid)
                if jid not in self.motors:
                    continue
                motor = self.motors[jid]
                cfg = JOINT_CONFIG.get(jid, (None, 5.0, 1.0))
                q = p.get('q', 0)
                kp = p.get('kp', cfg[1])
                kd = p.get('kd', cfg[2])
                dq = p.get('dq', 0)
                tau = p.get('tau', 0)
                
                if jid in ANKLE_MOTOR_IDS:
                    pair = (27, 28) if jid in (27, 28) else (33, 34)
                    if pair not in ankle_data:
                        ankle_data[pair] = {}
                    key = 'pitch' if jid == pair[0] else 'roll'
                    ankle_data[pair][key] = (motor, q, dq, tau, kp, kd)
                else:
                    self.dm_mc.controlMIT(motor, kp, kd, q, dq, tau)
            
            for pair, data in ankle_data.items():
                if 'pitch' in data and 'roll' in data:
                    m_pitch, pitch, pitch_vel, tau_pitch, kp, kd = data['pitch']
                    m_roll, roll, roll_vel, tau_roll, _, _ = data['roll']
                    self.dm_mc.controlMIT_ankle(m_pitch, m_roll, kp, kd, pitch, roll, 
                                                pitch_vel, roll_vel, tau_pitch, tau_roll)
    
    def _exec_gripper(self, gid: int, cmd: dict):
        with self.motor_lock:
            if gid not in self.motors:
                return
            motor = self.motors[gid]
            cfg = JOINT_CONFIG.get(gid, (None, 0.5, 0.05))
            
            cmd_type = cmd.get('type')
            if cmd_type == 'oc':
                pos = GRIPPER_CLOSE if cmd.get('close') else GRIPPER_OPEN
            elif cmd_type in ('pos', 'mit'):
                percent = max(0, min(100, cmd.get('percent', 0)))
                pos = GRIPPER_OPEN + (GRIPPER_CLOSE - GRIPPER_OPEN) * (percent / 100.0)
            else:
                return
            
            kp = cmd.get('kp', cfg[1])
            kd = cmd.get('kd', cfg[2])
            self.dm_mc.controlMIT(motor, kp, kd, pos, 0, 0)


# ==================== CAN控制器 ====================

class CANController:
    """CAN FD控制器"""
    
    def __init__(self, can_interface: str, state_buffer: deque, init_evt: threading.Event):
        self.can_interface = can_interface
        self.state_buffer = state_buffer
        self.init_evt = init_evt
        
        self.dm_mc = None
        self.motors = {}
        self.motor_lock = threading.Lock()
        self.workers = {}
        self.running = True
        
        self._init_can()
        self._start_workers()
        self._start_query_thread()
    
    def _init_can(self):
        try:
            self.dm_mc = DM_MotorControl(self.can_interface)
            print(f"✓ CAN FD初始化: {self.can_interface}")
        except Exception as e:
            print(f"✗ CAN FD初始化失败: {e}")
        self.init_evt.set()
    
    def _start_workers(self):
        for name, joint_ids in CAN_JOINT_MAP.items():
            worker = BodyPartWorker(name, joint_ids, self.dm_mc, self.motors, self.motor_lock)
            worker.start()
            self.workers[name] = worker
    
    def _start_query_thread(self):
        self.query_thread = threading.Thread(target=self._query_loop, daemon=True)
        self.query_thread.start()
    
    def _query_loop(self):
        query_interval = 0.002
        while self.running:
            try:
                result = {}
                with self.motor_lock:
                    for jid, motor in self.motors.items():
                        try:
                            self.dm_mc.refresh_motor_status(motor)
                            self.dm_mc.recv()
                            result[str(jid)] = {
                                'pos': float(motor.getPosition() or 0),
                                'vel': float(motor.getVelocity() or 0),
                                'tau': float(motor.getTorque() or 0)
                            }
                        except:
                            result[str(jid)] = {'pos': 0, 'vel': 0, 'tau': 0}
                    
                    # 踝关节转换
                    for pair in [(27, 28), (33, 34)]:
                        if pair[0] in self.motors and pair[1] in self.motors:
                            try:
                                m_pitch, m_roll = self.motors[pair[0]], self.motors[pair[1]]
                                pitch, roll = self.dm_mc.getAnklePosition(m_pitch, m_roll)
                                pitch_vel, roll_vel = self.dm_mc.getAnkleVelocity(m_pitch, m_roll)
                                tau_pitch, tau_roll = self.dm_mc.getAnkleTorque(m_pitch, m_roll)
                                result[str(pair[0])] = {'pos': pitch, 'vel': pitch_vel, 'tau': tau_pitch}
                                result[str(pair[1])] = {'pos': roll, 'vel': roll_vel, 'tau': tau_roll}
                            except:
                                pass
                
                if result:
                    self.state_buffer.append(('motor', result))
                
                time.sleep(query_interval)
            except Exception as e:
                print(f"查询错误: {e}")
                time.sleep(0.01)
    
    def set_mit_cmd(self, can_name: str, joints: dict):
        if can_name in self.workers:
            self.workers[can_name].set_mit_cmd(joints)
    
    def set_gripper_cmd(self, can_name: str, gid: int, cmd: dict):
        if can_name in self.workers:
            self.workers[can_name].set_gripper_cmd(gid, cmd)
    
    def request_register(self, can_name: str):
        if can_name in self.workers:
            self.workers[can_name].request_register()
    
    def request_disable(self, can_name: str):
        if can_name in self.workers:
            self.workers[can_name].request_disable()
    
    def stop(self):
        self.running = False
        for worker in self.workers.values():
            worker.stop()


# ==================== IMU Worker ====================

class IMUWorker:
    def __init__(self, path: str, state_buffer: deque, baud: int, init_evt: threading.Event):
        self.path = path
        self.state_buffer = state_buffer
        self.baud = baud
        self.init_evt = init_evt
        self.imu = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _run(self):
        try:
            self.imu = DM_IMU(port=self.path, baudrate=self.baud)
            self.imu.start()
            print(f"✓ IMU初始化: {self.path}")
        except Exception as e:
            print(f"✗ IMU初始化失败: {e}")
        self.init_evt.set()
        
        while self.running:
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
                    self.state_buffer.append(('imu', data))
                time.sleep(0.001)
            except Exception as e:
                print(f"IMU错误: {e}")
                time.sleep(0.01)


# ==================== 服务器 ====================

class MotorServer:
    def __init__(self, can_interface: str = "can1", imu_path: str = "/dev/imu", baudrate: int = 921600):
        self.running = True
        self.closed = False
        
        self.state_buffer = deque(maxlen=100)
        self.motor_state = {}
        self.imu_state = {}
        
        # CAN控制器
        can_evt = threading.Event()
        self.can_ctrl = CANController(can_interface, self.state_buffer, can_evt)
        can_evt.wait(timeout=5.0)
        
        # IMU
        imu_evt = threading.Event()
        self.imu_worker = IMUWorker(imu_path, self.state_buffer, baudrate, imu_evt)
        self.imu_worker.start()
        imu_evt.wait(timeout=5.0)
        
        # Zenoh
        self.session = zenoh.open(zenoh.Config())
        self.pub_motor = self.session.declare_publisher("taks/state/motor")
        self.pub_imu = self.session.declare_publisher("taks/state/imu")
        self.sub_cmd = self.session.declare_subscriber("taks/cmd/**", self._handle_cmd)
        
        self.pub_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.pub_thread.start()
        
        print(f"\n{'='*50}\n✓ 服务器启动 (CAN FD)\n{'='*50}\n")
    
    def _handle_cmd(self, sample):
        try:
            key = str(sample.key_expr)
            payload = sample.payload.to_bytes()
            msg = deserialize(payload) if payload else {}
            
            parts = key.split('/')
            if len(parts) < 4:
                return
            
            device, cmd = parts[2], parts[3]
            
            if device == 'Taks-T1':
                targets = list(CAN_JOINT_MAP.keys())
            elif device == 'Taks-T1-semibody':
                targets = ['right_hand', 'left_hand', 'waist_neck']
            elif device in ('Taks-T1-rightgripper', 'Taks-T1-leftgripper'):
                targets = ['right_hand'] if 'right' in device else ['left_hand']
            else:
                targets = []
            
            if cmd == 'register':
                for can in targets:
                    self.can_ctrl.request_register(can)
                print(f"✓ 注册: {device}")
            
            elif cmd == 'disable_all':
                for can in targets:
                    self.can_ctrl.request_disable(can)
            
            elif cmd == 'mit':
                joints = {int(k): v for k, v in msg.get('joints', {}).items()}
                by_can = {}
                for jid, val in joints.items():
                    can = get_can_for_joint(jid)
                    if can:
                        by_can.setdefault(can, {})[jid] = val
                for can, cj in by_can.items():
                    self.can_ctrl.set_mit_cmd(can, cj)
            
            elif cmd in ('gripper_oc', 'gripper_pos', 'gripper_mit'):
                gid = msg.get('gripper_id')
                if gid is not None:
                    can = get_can_for_joint(gid)
                    if can:
                        if cmd == 'gripper_oc':
                            self.can_ctrl.set_gripper_cmd(can, gid, {'type': 'oc', 'close': msg.get('close', False)})
                        elif cmd == 'gripper_pos':
                            self.can_ctrl.set_gripper_cmd(can, gid, {'type': 'pos', 'percent': msg.get('percent', 0)})
                        else:
                            self.can_ctrl.set_gripper_cmd(can, gid, {'type': 'mit', 'percent': msg.get('percent', 0),
                                                                     'kp': msg.get('kp'), 'kd': msg.get('kd')})
        except Exception as e:
            print(f"命令错误: {e}")
    
    def _publish_loop(self):
        while self.running:
            try:
                while self.state_buffer:
                    try:
                        src, data = self.state_buffer.popleft()
                        if src == 'imu':
                            self.imu_state = data
                        else:
                            self.motor_state.update(data)
                    except IndexError:
                        break
                
                if self.motor_state:
                    self.pub_motor.put(serialize({'ts': time.perf_counter(), 'joints': self.motor_state}))
                if self.imu_state:
                    self.pub_imu.put(serialize(self.imu_state))
                
                time.sleep(0.001)
            except Exception as e:
                print(f"发布错误: {e}")
                time.sleep(0.01)
    
    def run(self):
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n✓ 收到中断")
        finally:
            self.close()
    
    def close(self):
        if self.closed:
            return
        self.closed = True
        print("✓ 关闭服务器...")
        self.running = False
        
        for can in CAN_JOINT_MAP.keys():
            self.can_ctrl.request_disable(can)
        time.sleep(0.3)
        self.can_ctrl.stop()
        self.imu_worker.stop()
        
        try:
            self.sub_cmd.undeclare()
            self.pub_motor.undeclare()
            self.pub_imu.undeclare()
            self.session.close()
        except:
            pass
        
        print("✓ 服务器已关闭")


def main():
    server = MotorServer(can_interface="can1", imu_path="/dev/imu", baudrate=921600)
    
    def sig_handler(signum, frame):
        server.close()
    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    server.run()


if __name__ == "__main__":
    main()