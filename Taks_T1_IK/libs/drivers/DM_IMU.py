"""
DM-IMU-L1 六轴IMU模块 Python驱动

使用方法:
=========

1. 回调模式 (推荐):
import time
from dm_imu import DM_IMU

imu = DM_IMU(port='/dev/imu', baudrate=921600)
imu.start()

counter = 0
try:
    while True:
        counter += 1
        euler = imu.get_euler()
        print(f"计数: {counter:4d} | 欧拉角(°): Roll={euler['roll']:7.2f} Pitch={euler['pitch']:7.2f} Yaw={euler['yaw']:7.2f}")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\n程序已退出")
finally:
    imu.stop()

2. 轮询模式:
   from dm_imu import DM_IMU
   
   imu = DM_IMU(port='/dev/imu', baudrate=921600)
   imu.start()
   
   while True:
       data = imu.get_data()
       print(f"加速度: {data['accel']}")
       print(f"角速度: {data['gyro']}")
       print(f"欧拉角: {data['euler']}")
       time.sleep(0.01)
   
   imu.stop()

3. 上下文管理器:
   from dm_imu import DM_IMU
   
   with DM_IMU(port='/dev/imu') as imu:
       while True:
           data = imu.get_data()
           print(data)

4. 标定指令:
   imu.calibrate_zero()   # 角度值标定成零位，发送 AA 0C 01 0D，自动保存参数
   imu.calibrate_gyro()   # 启动陀螺静态校准，发送 AA 03 02 0D，自动保存参数
   
5. 参数管理:
   imu.save_params()           # 手动保存参数到Flash
   imu.enter_setting_mode()    # 进入设置模式
   imu.exit_setting_mode()     # 退出设置模式

数据单位:
  - 加速度: m/s²
  - 角速度: rad/s
  - 欧拉角: rad (弧度)
"""

import serial
import struct
import threading
import time
from typing import Callable, Optional, Dict

# 帧格式常量
_FRAME_HEADER = 0x55
_FRAME_FLAG = 0xAA
_FRAME_TAIL = 0x0A
_FRAME_LENGTH = 19      # 加速度/角速度/欧拉角帧长度
_FRAME_LENGTH_QUAT = 23  # 四元数帧长度

# 寄存器ID
_REG_ACCEL = 0x01  # 加速度
_REG_GYRO = 0x02   # 角速度
_REG_EULER = 0x03  # 欧拉角
_REG_QUAT = 0x04   # 四元数

# 指令常量（十六进制）
_CMD_ENTER_SETTING_MODE = bytes([0xAA, 0x69, 0x88, 0x0D])  # 进入设置模式
_CMD_EXIT_SETTING_MODE = bytes([0xAA, 0x6E, 0x88, 0x0D])   # 退出设置模式
_CMD_SAVE_PARAMS = bytes([0xAA, 0x00, 0x00, 0x0D])         # 保存参数到Flash
_CMD_CALIBRATE_ZERO = bytes([0xAA, 0x0C, 0x01, 0x0D])     # 角度值标定成零位
_CMD_CALIBRATE_GYRO = bytes([0xAA, 0x03, 0x02, 0x0D])     # 启动陀螺静态校准

# 通信接口序号
_COMM_INTERFACE_USB = 0x00
_COMM_INTERFACE_485 = 0x01
_COMM_INTERFACE_CAN = 0x02
_COMM_INTERFACE_VOFA = 0x03


def _parse_float(data_bytes):
    """将4字节数据转换为浮点数"""
    return struct.unpack('<f', bytes(data_bytes))[0]


class DM_IMU:
    """DM-IMU-L1 六轴IMU模块驱动类"""
    
    def __init__(self, port: str = '/dev/imu', baudrate: int = 921600):
        """
        初始化IMU
        
        Args:
            port: 串口号，如 '/dev/imu' 或 '/dev/ttyUSB0'
            baudrate: 波特率，默认 921600
        """
        self.port = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._lock = threading.Lock()
        
        # IMU数据 (欧拉角单位：弧度)
        self._accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._gyro = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self._euler = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self._quat = {'w': 1.0, 'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    def set_callback(self, callback: Callable[[Dict, Dict, Dict], None]):
        """
        设置数据回调函数
        
        Args:
            callback: 回调函数，接收三个参数 (accel, gyro, euler)
        """
        self._callback = callback
    
    def start(self):
        """启动IMU数据读取"""
        if self._running:
            return
        
        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """停止IMU数据读取"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial:
            self._serial.close()
            self._serial = None
    
    def get_data(self) -> Dict:
        """
        获取最新的IMU数据
        
        Returns:
            包含 accel, gyro, euler 的字典
        """
        with self._lock:
            return {
                'accel': self._accel.copy(),
                'gyro': self._gyro.copy(),
                'euler': self._euler.copy()
            }
    
    def get_accel(self) -> Dict:
        """获取加速度数据 (m/s²)"""
        with self._lock:
            return self._accel.copy()
    
    def get_gyro(self) -> Dict:
        """获取角速度数据 (rad/s)"""
        with self._lock:
            return self._gyro.copy()
    
    def get_euler(self) -> Dict:
        """获取欧拉角数据 (rad)"""
        with self._lock:
            return self._euler.copy()
    
    def get_quat(self) -> Dict:
        """获取四元数数据"""
        with self._lock:
            return self._quat.copy()
    
    def _parse_packet(self, packet: bytearray, is_quat: bool = False) -> Optional[Dict]:
        """解析数据包"""
        expected_len = _FRAME_LENGTH_QUAT if is_quat else _FRAME_LENGTH
        tail_idx = 22 if is_quat else 18
        
        if len(packet) != expected_len:
            return None
        
        if packet[0] != _FRAME_HEADER or packet[1] != _FRAME_FLAG or packet[tail_idx] != _FRAME_TAIL:
            return None
        
        slave_id = packet[2]
        reg_id = packet[3]
        
        if is_quat:
            val_w = _parse_float(packet[4:8])
            val_x = _parse_float(packet[8:12])
            val_y = _parse_float(packet[12:16])
            val_z = _parse_float(packet[16:20])
            return {
                'slave_id': slave_id,
                'reg_id': reg_id,
                'w': val_w,
                'x': val_x,
                'y': val_y,
                'z': val_z
            }
        else:
            val_x = _parse_float(packet[4:8])
            val_y = _parse_float(packet[8:12])
            val_z = _parse_float(packet[12:16])
            return {
                'slave_id': slave_id,
                'reg_id': reg_id,
                'x': val_x,
                'y': val_y,
                'z': val_z
            }
    
    def _read_loop(self):
        """数据读取循环"""
        buffer = bytearray()
        
        while self._running:
            try:
                data = self._serial.read(256)
                if data:
                    buffer.extend(data)
                
                while len(buffer) >= _FRAME_LENGTH:
                    try:
                        idx = buffer.index(_FRAME_HEADER)
                        if idx > 0:
                            buffer = buffer[idx:]
                    except ValueError:
                        buffer.clear()
                        break
                    
                    if len(buffer) < _FRAME_LENGTH:
                        break
                    
                    if buffer[1] != _FRAME_FLAG:
                        buffer = buffer[1:]
                        continue
                    
                    reg_id = buffer[3] if len(buffer) > 3 else 0
                    is_quat = (reg_id == _REG_QUAT)
                    frame_len = _FRAME_LENGTH_QUAT if is_quat else _FRAME_LENGTH
                    
                    if len(buffer) < frame_len:
                        break
                    
                    packet = buffer[:frame_len]
                    result = self._parse_packet(packet, is_quat)
                    
                    if result:
                        reg_id = result['reg_id']
                        with self._lock:
                            if reg_id == _REG_ACCEL:
                                self._accel = {'x': result['x'], 'y': result['y'], 'z': result['z']}
                            elif reg_id == _REG_GYRO:
                                self._gyro = {'x': result['x'], 'y': result['y'], 'z': result['z']}
                            elif reg_id == _REG_EULER:
                                import math
                                self._euler = {
                                    'roll': math.radians(result['x']),
                                    'pitch': math.radians(result['y']),
                                    'yaw': math.radians(result['z'])
                                }
                            elif reg_id == _REG_QUAT:
                                self._quat = {'w': result['w'], 'x': result['x'], 'y': result['y'], 'z': result['z']}
                        
                        if self._callback:
                            self._callback(self._accel.copy(), self._gyro.copy(), self._euler.copy())
                        
                        buffer = buffer[frame_len:]
                    else:
                        buffer = buffer[1:]
            except Exception:
                pass
    
    def _send_command(self, command: bytes, delay: float = 0.01) -> bool:
        """
        发送指令到IMU
        
        Args:
            command: 指令字节序列
            delay: 发送后延迟时间（秒）
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(command)
                time.sleep(delay)
                return True
            except Exception as e:
                print(f"✗ IMU指令发送失败: {e}", flush=True)
                return False
        return False
    
    def enter_setting_mode(self) -> bool:
        """
        进入设置模式
        发送指令: AA 69 88 0D
        
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        return self._send_command(_CMD_ENTER_SETTING_MODE)
    
    def exit_setting_mode(self) -> bool:
        """
        退出设置模式
        发送指令: AA 6E 88 0D
        
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        return self._send_command(_CMD_EXIT_SETTING_MODE)
    
    def save_params(self) -> bool:
        """
        保存参数到Flash
        发送指令: AA 00 00 0D
        
        注意：对模块参数进行修改后务必执行此指令将参数保存，防止掉电参数丢失
        
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        return self._send_command(_CMD_SAVE_PARAMS, delay=0.1)
    
    def calibrate_zero(self, save: bool = True) -> bool:
        """
        角度值标定成零位
        发送指令: AA 0C 01 0D
        
        注意：此指令不需要在设置模式中执行，但保存参数需要在设置模式中
        
        Args:
            save: 是否自动保存参数到Flash（默认True）
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        success = self._send_command(_CMD_CALIBRATE_ZERO, delay=0.05)
        if not success:
            return False
        
        if save:
            time.sleep(0.05)
            if not self.enter_setting_mode():
                return False
            time.sleep(0.02)
            if not self.save_params():
                self.exit_setting_mode()
                return False
            time.sleep(0.02)
            self.exit_setting_mode()
        
        return True
    
    def calibrate_gyro(self, save: bool = True) -> bool:
        """
        启动陀螺静态校准
        发送指令: AA 03 02 0D
        
        注意：此指令需要在设置模式中执行
        
        Args:
            save: 是否自动保存参数到Flash（默认True）
            
        Returns:
            bool: 发送成功返回True，否则返回False
        """
        if not self.enter_setting_mode():
            return False
        
        time.sleep(0.02)
        success = self._send_command(_CMD_CALIBRATE_GYRO, delay=0.5)
        if not success:
            self.exit_setting_mode()
            return False
        
        if save:
            time.sleep(0.05)
            if not self.save_params():
                self.exit_setting_mode()
                return False
        
        time.sleep(0.02)
        self.exit_setting_mode()
        return True
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def main():
    """测试程序"""
    imu = DM_IMU(port='/dev/imu', baudrate=921600)
    
    try:
        imu.start()
        print("已连接到 IMU，按 Ctrl+C 退出...")
        
        # 执行标定指令（自动保存参数到Flash）
        time.sleep(0.1)  # 等待串口稳定

        # if imu.calibrate_gyro():
        #     print("已发送陀螺静态校准指令并保存参数")
        # time.sleep(0.5)  # 等待校准完成

        # if imu.calibrate_zero():
        #     print("已发送角度值标定零位指令并保存参数")
        
        while True:
            data = imu.get_data()
            accel = data['accel']
            gyro = data['gyro']
            euler = data['euler']
            quat = imu.get_quat()
            
            output = (
                f"加速度(m/s²): X={accel['x']:8.3f} Y={accel['y']:8.3f} Z={accel['z']:8.3f} | "
                f"角速度(rad/s): X={gyro['x']:8.4f} Y={gyro['y']:8.4f} Z={gyro['z']:8.4f} | "
                f"欧拉角(rad): Roll={euler['roll']:7.4f} Pitch={euler['pitch']:7.4f} Yaw={euler['yaw']:7.4f}"
                f"四元数: W={quat['w']:7.4f} X={quat['x']:7.4f} Y={quat['y']:7.4f} Z={quat['z']:7.4f}"
            )
            print(f"\r{output}", end='', flush=True)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        imu.stop()


if __name__ == '__main__':
    main()