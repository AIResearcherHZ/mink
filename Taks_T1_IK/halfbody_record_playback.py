"""半身IK录制回放程序

功能:
- 录制: 记录时间戳和MIT命令参数到.npz文件
- 回放: 加载录制文件，支持循环播放、缓启动/停止
- Ctrl+C安全退出: 先缓停止再断开连接
"""

import sys
import argparse
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

sys.path.insert(0, str(Path(__file__).parent))
from halfbody_ik_vr_refactored import (
    HalfBodyIKController, SDK_JOINT_GAINS, SAFE_KP_KD, SDK_ID_TO_NAME,
    TAKS_SEND_RATE, RAMP_UP_TIME, RAMP_DOWN_TIME, ease_in, ease_out
)
from taks_sdk import taks


# ==================== 数据结构 ====================

@dataclass
class RecordFrame:
    """单帧录制数据"""
    timestamp: float  # 相对时间戳(秒)
    mit_cmd: Dict[int, Dict[str, float]]  # SDK_ID -> {q, dq, tau, kp, kd}


@dataclass
class RecordData:
    """录制数据集"""
    frames: List[RecordFrame] = field(default_factory=list)
    start_time: float = 0.0
    
    def add_frame(self, mit_cmd: Dict[int, Dict[str, float]]):
        """添加一帧"""
        if not self.frames:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time
        self.frames.append(RecordFrame(timestamp=timestamp, mit_cmd=mit_cmd.copy()))
    
    def save(self, filepath: str):
        """保存到npz文件"""
        if not self.frames:
            print("[录制] 无数据可保存")
            return
        
        # 提取所有SDK ID
        sdk_ids = sorted(self.frames[0].mit_cmd.keys())
        n_frames = len(self.frames)
        n_joints = len(sdk_ids)
        
        # 构建数组
        timestamps = np.array([f.timestamp for f in self.frames], dtype=np.float64)
        q_arr = np.zeros((n_frames, n_joints), dtype=np.float64)
        dq_arr = np.zeros((n_frames, n_joints), dtype=np.float64)
        tau_arr = np.zeros((n_frames, n_joints), dtype=np.float64)
        kp_arr = np.zeros((n_frames, n_joints), dtype=np.float64)
        kd_arr = np.zeros((n_frames, n_joints), dtype=np.float64)
        
        for i, frame in enumerate(self.frames):
            for j, sdk_id in enumerate(sdk_ids):
                cmd = frame.mit_cmd.get(sdk_id, {})
                q_arr[i, j] = cmd.get('q', 0.0)
                dq_arr[i, j] = cmd.get('dq', 0.0)
                tau_arr[i, j] = cmd.get('tau', 0.0)
                kp_arr[i, j] = cmd.get('kp', 0.0)
                kd_arr[i, j] = cmd.get('kd', 0.0)
        
        np.savez(filepath, 
                 timestamps=timestamps, 
                 sdk_ids=np.array(sdk_ids, dtype=np.int32),
                 q=q_arr, dq=dq_arr, tau=tau_arr, kp=kp_arr, kd=kd_arr)
        print(f"[录制] 已保存 {n_frames} 帧到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RecordData':
        """从npz文件加载"""
        data = np.load(filepath)
        timestamps = data['timestamps']
        sdk_ids = data['sdk_ids'].tolist()
        q_arr = data['q']
        dq_arr = data['dq']
        tau_arr = data['tau']
        kp_arr = data['kp']
        kd_arr = data['kd']
        
        record = cls()
        record.start_time = 0.0
        for i in range(len(timestamps)):
            mit_cmd = {}
            for j, sdk_id in enumerate(sdk_ids):
                mit_cmd[sdk_id] = {
                    'q': float(q_arr[i, j]),
                    'dq': float(dq_arr[i, j]),
                    'tau': float(tau_arr[i, j]),
                    'kp': float(kp_arr[i, j]),
                    'kd': float(kd_arr[i, j]),
                }
            record.frames.append(RecordFrame(timestamp=float(timestamps[i]), mit_cmd=mit_cmd))
        
        print(f"[回放] 已加载 {len(record.frames)} 帧, 时长 {timestamps[-1]:.2f}s")
        return record


# ==================== 录制器 ====================

class Recorder:
    """录制器: 基于HalfBodyIKController录制MIT命令"""
    
    def __init__(self, output_path: str, **controller_kwargs):
        self.output_path = output_path
        self.controller = HalfBodyIKController(**controller_kwargs)
        self.record_data = RecordData()
        self.running = True
        self.shutdown_requested = False
        self.console = Console()
        
        # 禁用控制器的缓启动/停止(录制时不需要)
        self.controller.enable_ramp_up = False
        self.controller.enable_ramp_down = False
        self.controller.ramp_state.active = False
        self.controller.ramp_state.progress = 1.0
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        if self.shutdown_requested:
            print("\n[强制退出]")
            sys.exit(1)
        print("\n[收到退出信号] 停止录制...")
        self.shutdown_requested = True
        self.running = False
    
    def run(self):
        """运行录制"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"[录制] 开始录制，输出: {self.output_path}")
        print("[录制] Ctrl+C 停止录制并保存")
        
        try:
            # 等待VR校准
            print("[录制] 等待VR校准 (按C键或VR手柄B键)...")
            while self.running and not self.controller.vr_calib.done:
                vr_data = self.controller.vr.data
                if vr_data.button_events.right_b:
                    self.controller.calibrate()
                time.sleep(0.1)
            
            if not self.running:
                return
            
            print("[录制] 开始录制帧数据...")
            frame_count = 0
            start_time = time.time()
            
            while self.running:
                # 执行一步控制
                mit_cmd = self.controller.step()
                
                # 录制(跳过缓启动阶段，但这里已禁用)
                self.record_data.add_frame(mit_cmd)
                frame_count += 1
                
                # 更新统计
                self.controller.update_stats()
                
                # 打印状态
                elapsed = time.time() - start_time
                if frame_count % 30 == 0:  # 每30帧打印一次
                    self.console.clear()
                    print(f"[录制] 帧数: {frame_count} | 时长: {elapsed:.1f}s | FPS: {self.controller.current_fps:.1f}")
                
                self.controller.rate.sleep()
        
        except Exception as e:
            print(f"[录制] 错误: {e}")
        finally:
            # 保存录制
            self.record_data.save(self.output_path)
            self.controller.cleanup()


# ==================== 回放器 ====================

class Player:
    """回放器: 加载录制文件并回放到真机"""
    
    def __init__(self, input_path: str, loop: bool = False,
                 host: str = "192.168.5.4", port: int = 5555,
                 ramp_up_time: float = RAMP_UP_TIME, ramp_down_time: float = RAMP_DOWN_TIME,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        self.input_path = input_path
        self.loop = loop
        self.host = host
        self.port = port
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        
        self.running = True
        self.shutdown_requested = False
        self.console = Console()
        self.robot = None
        
        # 加载录制数据
        self.record_data = RecordData.load(input_path)
        if not self.record_data.frames:
            raise ValueError("录制文件为空")
        
        # 统计
        self.last_send_time = 0.0
        self.send_count = 0
        self.send_fps_start = time.time()
        self.current_send_fps = 0.0
    
    def _connect(self):
        """连接真机"""
        try:
            taks.connect(self.host, cmd_port=self.port)
            self.robot = taks.register("Taks-T1-semibody")
            print(f"[回放] 已连接 {self.host}:{self.port}")
            time.sleep(2.0)
        except Exception as e:
            print(f"[回放] 连接失败: {e}")
            raise
    
    def _disconnect(self):
        """断开连接"""
        if self.robot is not None:
            taks.disconnect()
            print("[回放] 已断开")
    
    def _get_real_positions(self, timeout: float = 2.0) -> Dict[int, float]:
        """获取真机当前位置"""
        if self.robot is None:
            return {}
        start = time.time()
        while time.time() - start < timeout:
            real_pos = self.robot.GetPosition()
            if real_pos is not None and len(real_pos) > 0:
                return real_pos
            time.sleep(0.05)
        print(f"[警告] 获取真机位置超时")
        return {}
    
    def _ramp_up(self, target_cmd: Dict[int, Dict[str, float]]):
        """缓启动: 从当前位置渐变到目标位置"""
        if not self.enable_ramp_up:
            print("[回放] 缓启动已禁用")
            return
        
        print(f"[回放] 缓启动 ({self.ramp_up_time}s)...")
        start_positions = self._get_real_positions()
        start = time.time()
        
        while time.time() - start < self.ramp_up_time and self.running:
            elapsed = time.time() - start
            t = elapsed / self.ramp_up_time
            kp_kd_scale = ease_out(t)
            
            mit_cmd = {}
            for sdk_id, cmd in target_cmd.items():
                start_q = start_positions.get(sdk_id, cmd['q'])
                # 位置线性插值
                q_val = start_q + (cmd['q'] - start_q) * t
                # kp/kd非线性渐变
                kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
                mit_cmd[sdk_id] = {
                    'q': q_val, 'dq': 0.0, 'tau': 0.0,
                    'kp': kp_target * kp_kd_scale,
                    'kd': kd_target * kp_kd_scale
                }
            
            self.robot.controlMIT(mit_cmd)
            time.sleep(0.001)
        
        print("[回放] 缓启动完成")
    
    def _ramp_down(self):
        """缓停止"""
        if not self.enable_ramp_down or self.robot is None:
            if self.robot is not None:
                # 直接失能
                sdk_ids = list(self.record_data.frames[0].mit_cmd.keys())
                mit_cmd = {sdk_id: {'q': 0.0, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} 
                           for sdk_id in sdk_ids}
                self.robot.controlMIT(mit_cmd)
            print("[回放] 缓停止已禁用，直接失能")
            return
        
        print(f"[回放] 缓停止 ({self.ramp_down_time}s)...")
        start_positions = self._get_real_positions()
        start = time.time()
        
        while time.time() - start < self.ramp_down_time:
            elapsed = time.time() - start
            t = elapsed / self.ramp_down_time
            kp_kd_scale = 1.0 - ease_in(t)
            
            mit_cmd = {}
            for sdk_id in start_positions.keys():
                kp_target, kd_target = SDK_JOINT_GAINS.get(sdk_id, (10.0, 1.0))
                kp_safe, kd_safe = SAFE_KP_KD.get(sdk_id, (5.0, 1.0))
                kp_val = kp_safe + (kp_target - kp_safe) * kp_kd_scale
                kd_val = kd_safe + (kd_target - kd_safe) * kp_kd_scale
                q_val = start_positions.get(sdk_id, 0.0)
                mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp_val, 'kd': kd_val}
            
            self.robot.controlMIT(mit_cmd)
            time.sleep(0.001)
        
        # 失能
        mit_cmd = {sdk_id: {'q': start_positions.get(sdk_id, 0.0), 
                            'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0} 
                   for sdk_id in start_positions.keys()}
        self.robot.controlMIT(mit_cmd)
        print("[回放] 缓停止完成")
    
    def _send_cmd(self, mit_cmd: Dict[int, Dict[str, float]]) -> bool:
        """发送命令(受频率限制)"""
        if TAKS_SEND_RATE:
            now = time.time()
            if now - self.last_send_time < 1.0 / TAKS_SEND_RATE:
                return False
            self.last_send_time = now
        self.robot.controlMIT(mit_cmd)
        self.send_count += 1
        return True
    
    def _update_stats(self):
        """更新统计"""
        now = time.time()
        if now - self.send_fps_start >= 1.0:
            self.current_send_fps = self.send_count / (now - self.send_fps_start)
            self.send_count = 0
            self.send_fps_start = now
    
    def _build_status_table(self, frame_idx: int, loop_count: int, mit_cmd: Dict) -> Table:
        """构建状态表格"""
        table = Table(title="回放状态", show_header=True, header_style="bold cyan")
        table.add_column("ID", style="dim", width=3)
        table.add_column("关节名", width=28)
        table.add_column("q(rad)", justify="right", width=8)
        table.add_column("kp", justify="right", width=6)
        table.add_column("kd", justify="right", width=6)
        
        for sdk_id in sorted(mit_cmd.keys()):
            cmd = mit_cmd[sdk_id]
            jname = SDK_ID_TO_NAME.get(sdk_id, f"joint_{sdk_id}")
            table.add_row(str(sdk_id), jname, f"{cmd['q']:.3f}", 
                          f"{cmd['kp']:.2f}", f"{cmd['kd']:.2f}")
        
        table.add_section()
        total_frames = len(self.record_data.frames)
        duration = self.record_data.frames[-1].timestamp
        rate_str = f"{TAKS_SEND_RATE:.0f}Hz" if TAKS_SEND_RATE else "无限制"
        status1 = f"帧: {frame_idx+1}/{total_frames} | 循环: {loop_count} | 发送: {self.current_send_fps:.1f}Hz"
        status2 = f"时长: {duration:.1f}s | 循环模式: {'ON' if self.loop else 'OFF'}"
        table.add_row("", status1, "", "", "")
        table.add_row("", status2, "", "", "")
        return table
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        if self.shutdown_requested:
            print("\n[强制退出]")
            sys.exit(1)
        print("\n[收到退出信号] 开始安全关闭...")
        self.shutdown_requested = True
        self.running = False
    
    def run(self):
        """运行回放"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"[回放] 文件: {self.input_path}")
        print(f"[回放] 循环模式: {'ON' if self.loop else 'OFF'}")
        print("[回放] Ctrl+C 安全停止")
        
        try:
            self._connect()
            
            # 缓启动到第一帧位置
            first_cmd = self.record_data.frames[0].mit_cmd
            self._ramp_up(first_cmd)
            
            if not self.running:
                return
            
            # 回放循环
            loop_count = 0
            print_interval = 0.5
            last_print_time = time.time()
            
            while self.running:
                loop_count += 1
                print(f"[回放] 开始第 {loop_count} 次播放")
                
                play_start = time.time()
                frame_idx = 0
                
                while frame_idx < len(self.record_data.frames) and self.running:
                    frame = self.record_data.frames[frame_idx]
                    
                    # 等待到达帧时间
                    target_time = play_start + frame.timestamp
                    while time.time() < target_time and self.running:
                        time.sleep(0.001)
                    
                    if not self.running:
                        break
                    
                    # 发送命令
                    self._send_cmd(frame.mit_cmd)
                    self._update_stats()
                    
                    # 打印状态
                    now = time.time()
                    if now - last_print_time >= print_interval:
                        last_print_time = now
                        self.console.clear()
                        self.console.print(self._build_status_table(frame_idx, loop_count, frame.mit_cmd))
                    
                    frame_idx += 1
                
                if not self.loop:
                    break
                
                print(f"[回放] 第 {loop_count} 次播放完成，重新开始...")
            
            print("[回放] 播放结束")
        
        except Exception as e:
            print(f"[回放] 错误: {e}")
        finally:
            self._ramp_down()
            self._disconnect()


# ==================== 命令行入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="半身IK录制回放")
    subparsers = parser.add_subparsers(dest="mode", help="模式")
    
    # 录制模式
    rec_parser = subparsers.add_parser("record", help="录制模式")
    rec_parser.add_argument("-o", "--output", type=str, default="recording.npz", help="输出文件路径")
    rec_parser.add_argument("--host", type=str, default="192.168.5.4", help="taks服务器地址")
    rec_parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    rec_parser.add_argument("--no-real", action="store_true", help="禁用真机控制")
    rec_parser.add_argument("--headless", action="store_true", help="无头模式")
    
    # 回放模式
    play_parser = subparsers.add_parser("play", help="回放模式")
    play_parser.add_argument("-i", "--input", type=str, required=True, help="输入文件路径")
    play_parser.add_argument("--loop", action="store_true", help="循环播放")
    play_parser.add_argument("--host", type=str, default="192.168.5.4", help="taks服务器地址")
    play_parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    play_parser.add_argument("--no-ramp-up", action="store_true", help="禁用缓启动")
    play_parser.add_argument("--no-ramp-down", action="store_true", help="禁用缓停止")
    play_parser.add_argument("--ramp-up-time", type=float, default=RAMP_UP_TIME, help="缓启动时间(秒)")
    play_parser.add_argument("--ramp-down-time", type=float, default=RAMP_DOWN_TIME, help="缓停止时间(秒)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == "record":
        recorder = Recorder(
            output_path=args.output,
            host=args.host,
            port=args.port,
            enable_real=not args.no_real,
            headless=args.headless,
        )
        recorder.run()
    
    elif args.mode == "play":
        player = Player(
            input_path=args.input,
            loop=args.loop,
            host=args.host,
            port=args.port,
            ramp_up_time=args.ramp_up_time,
            ramp_down_time=args.ramp_down_time,
            enable_ramp_up=not args.no_ramp_up,
            enable_ramp_down=not args.no_ramp_down,
        )
        player.run()
    
    else:
        print("用法:")
        print("  录制: python halfbody_record_playback.py record -o recording.npz")
        print("  回放: python halfbody_record_playback.py play -i recording.npz --loop")


if __name__ == "__main__":
    main()