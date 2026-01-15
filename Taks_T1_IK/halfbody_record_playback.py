"""半身IK录制回放程序

功能:
- 录制: 记录时间戳、MIT命令参数、夹爪数据到.npz文件
- 回放: 加载录制文件，支持循环播放、缓启动/停止、夹爪控制
- 支持仅MuJoCo仿真回放验证
- Ctrl+C安全退出: 先缓停止再断开连接

启动命令：
# 默认录制模式:
python halfbody_record_playback.py
# 录制(指定帧率):
python halfbody_record_playback.py record -o recording.npz --fps 30
# 回放(真机+MuJoCo):
python halfbody_record_playback.py play -i recording.npz --loop
# 回放(仅MuJoCo仿真验证):
python halfbody_record_playback.py play -i recording.npz --no-real
"""

import sys
import argparse
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from halfbody_ik_vr import (
    HalfBodyIKController, SDK_JOINT_GAINS, SAFE_KP_KD, SDK_ID_TO_NAME,
    TAKS_SEND_RATE, RAMP_UP_TIME, RAMP_DOWN_TIME, ease_in, ease_out,
    JOINT_NAME_TO_SDK_ID, _XML, SAFE_FALL_POSITIONS
)
from taks_sdk import taks


# ==================== 数据结构 ====================

@dataclass
class RecordFrame:
    """单帧录制数据"""
    timestamp: float  # 相对时间戳(秒)
    mit_cmd: Dict[int, Dict[str, float]]  # SDK_ID -> {q, dq, tau, kp, kd}
    left_gripper: float = 0.0  # 左夹爪百分比(0-100)
    right_gripper: float = 0.0  # 右夹爪百分比(0-100)


@dataclass
class RecordData:
    """录制数据集"""
    frames: List[RecordFrame] = field(default_factory=list)
    start_time: float = 0.0
    
    def add_frame(self, mit_cmd: Dict[int, Dict[str, float]], 
                   left_gripper: float = 0.0, right_gripper: float = 0.0):
        """添加一帧"""
        if not self.frames:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time
        self.frames.append(RecordFrame(
            timestamp=timestamp, mit_cmd=mit_cmd.copy(),
            left_gripper=left_gripper, right_gripper=right_gripper
        ))
    
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
        
        # 夹爪数据
        left_gripper_arr = np.array([f.left_gripper for f in self.frames], dtype=np.float64)
        right_gripper_arr = np.array([f.right_gripper for f in self.frames], dtype=np.float64)
        
        np.savez(filepath, 
                 timestamps=timestamps, 
                 sdk_ids=np.array(sdk_ids, dtype=np.int32),
                 q=q_arr, dq=dq_arr, tau=tau_arr, kp=kp_arr, kd=kd_arr,
                 left_gripper=left_gripper_arr, right_gripper=right_gripper_arr)
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
        
        # 夹爪数据(兼容旧格式)
        left_gripper_arr = data.get('left_gripper', np.zeros(len(timestamps)))
        right_gripper_arr = data.get('right_gripper', np.zeros(len(timestamps)))
        
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
            record.frames.append(RecordFrame(
                timestamp=float(timestamps[i]), mit_cmd=mit_cmd,
                left_gripper=float(left_gripper_arr[i]),
                right_gripper=float(right_gripper_arr[i])
            ))
        
        print(f"[回放] 已加载 {len(record.frames)} 帧, 时长 {timestamps[-1]:.2f}s")
        return record


# ==================== 录制器 ====================

class Recorder:
    """录制器: 基于HalfBodyIKController录制MIT命令"""
    
    def __init__(self, output_path: str, record_fps: int = 30, headless: bool = False, **controller_kwargs):
        self.output_path = output_path
        self.record_fps = record_fps
        self.record_interval = 1.0 / record_fps if record_fps > 0 else 0.0
        self.headless = headless
        self.controller = HalfBodyIKController(headless=True, **controller_kwargs)
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
        print(f"[录制] 模式: {'无头' if self.headless else '有头(MuJoCo)'} | 录制帧率: {self.record_fps}Hz")
        print("[录制] Ctrl+C 停止录制并保存")
        
        try:
            if self.headless:
                self._run_headless()
            else:
                self._run_with_viewer()
        except Exception as e:
            print(f"[录制] 错误: {e}")
        finally:
            self.record_data.save(self.output_path)
            self.controller.cleanup()
    
    def _run_headless(self):
        """无头模式录制"""
        self._wait_calibration()
        if not self.running:
            return
        self._record_loop(viewer=None)
    
    def _run_with_viewer(self):
        """有头模式录制(MuJoCo可视化)"""
        with mujoco.viewer.launch_passive(self.controller.model, self.controller.data,
                                           show_left_ui=False, show_right_ui=False,
                                           key_callback=self.controller.key_callback) as viewer:
            mujoco.mjv_defaultFreeCamera(self.controller.model, viewer.cam)
            self._wait_calibration(viewer)
            if not self.running:
                return
            self._record_loop(viewer=viewer)
    
    def _wait_calibration(self, viewer=None):
        """等待VR校准"""
        print("[录制] 等待VR校准 (按C键或VR手柄B键)...")
        while self.running and not self.controller.vr_calib.done:
            if viewer is not None:
                if not viewer.is_running():
                    self.running = False
                    break
                viewer.sync()
            vr_data = self.controller.vr.data
            if vr_data.button_events.right_b:
                self.controller.calibrate()
            time.sleep(0.05)
    
    def _record_loop(self, viewer=None):
        """录制循环"""
        print(f"[录制] 开始录制帧数据...")
        frame_count = 0
        start_time = time.time()
        last_record_time = 0.0
        print_interval = 1.0  # 每秒打印一次
        last_print_time = time.time()
        
        while self.running:
            if viewer is not None and not viewer.is_running():
                break
            
            # 执行一步控制
            mit_cmd = self.controller.step()
            vr_data = self.controller.vr.data
            
            # 获取夹爪值
            left_gripper = vr_data.left_hand.gripper * 100.0
            right_gripper = vr_data.right_hand.gripper * 100.0
            
            # 按指定帧率录制
            now = time.time()
            if now - last_record_time >= self.record_interval:
                self.record_data.add_frame(mit_cmd, left_gripper, right_gripper)
                frame_count += 1
                last_record_time = now
            
            # 更新统计
            self.controller.update_stats()
            
            # 同步viewer
            if viewer is not None:
                mujoco.mj_camlight(self.controller.model, self.controller.data)
                viewer.sync()
            
            # 打印状态(每秒一次)
            if now - last_print_time >= print_interval:
                last_print_time = now
                elapsed = now - start_time
                self.console.clear()
                print(f"[录制] 帧数: {frame_count} | 时长: {elapsed:.1f}s | 控制FPS: {self.controller.current_fps:.1f}")
            
            self.controller.rate.sleep()


# ==================== 回放器 ====================

class Player:
    """回放器: 加载录制文件并回放到真机/MuJoCo"""
    
    def __init__(self, input_path: str, loop: bool = False,
                 host: str = "192.168.5.4", port: int = 5555,
                 enable_real: bool = True, headless: bool = False,
                 ramp_up_time: float = RAMP_UP_TIME, ramp_down_time: float = RAMP_DOWN_TIME,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        self.input_path = input_path
        self.loop = loop
        self.host = host
        self.port = port
        self.enable_real = enable_real
        self.headless = headless
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        
        self.running = True
        self.shutdown_requested = False
        self.console = Console()
        self.robot = None
        self.left_gripper = None
        self.right_gripper = None
        
        # 加载录制数据
        self.record_data = RecordData.load(input_path)
        if not self.record_data.frames:
            raise ValueError("录制文件为空")
        
        # MuJoCo仿真回放
        self.model = None
        self.data = None
        self.joint_mapping = {}  # sdk_id -> qpos_idx
        if not headless:
            self._init_mujoco()
        
        # 统计
        self.last_send_time = 0.0
        self.send_count = 0
        self.send_fps_start = time.time()
        self.current_send_fps = 0.0
    
    def _init_mujoco(self):
        """初始化MuJoCo模型用于仿真回放"""
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        # 构建sdk_id -> qpos索引映射
        for jname, sdk_id in JOINT_NAME_TO_SDK_ID.items():
            try:
                jid = self.model.joint(jname).id
                self.joint_mapping[sdk_id] = self.model.jnt_qposadr[jid]
            except:
                pass
    
    def _connect(self):
        """连接真机"""
        if not self.enable_real:
            print("[回放] 仅仿真模式，跳过真机连接")
            return
        try:
            taks.connect(self.host, cmd_port=self.port)
            self.robot = taks.register("Taks-T1-semibody")
            print(f"[回放] 已注册半身设备")
            time.sleep(4.0)
            self.left_gripper = taks.register("Taks-T1-leftgripper")
            print(f"[回放] 已注册左gripper")
            time.sleep(1.0)
            self.right_gripper = taks.register("Taks-T1-rightgripper")
            print(f"[回放] 已注册右gripper")
            time.sleep(1.0)
            print(f"[回放] 已连接 {self.host}:{self.port}")
        except Exception as e:
            print(f"[回放] 连接失败: {e}")
            raise
    
    def _disconnect(self):
        """断开连接"""
        if self.enable_real and self.robot is not None:
            taks.disconnect()
            print("[回放] 已断开")
    
    def _get_real_positions(self, timeout: float = 2.0) -> Dict[int, float]:
        """获取真机当前位置"""
        if not self.enable_real or self.robot is None:
            # 仿真模式返回第一帧位置
            return {sdk_id: cmd['q'] for sdk_id, cmd in self.record_data.frames[0].mit_cmd.items()}
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
        if not self.enable_ramp_up or not self.enable_real:
            print("[回放] 缓启动已禁用/仅仿真模式")
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
                q_val = start_q + (cmd['q'] - start_q) * t
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
        if not self.enable_real or self.robot is None:
            print("[回放] 仅仿真模式，跳过缓停止")
            return
        if not self.enable_ramp_down:
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
                # 使用安全倒向位置(如果配置了)
                start_q = start_positions.get(sdk_id, 0.0)
                target_q = SAFE_FALL_POSITIONS.get(sdk_id, start_q)
                q_val = start_q + (target_q - start_q) * t
                mit_cmd[sdk_id] = {'q': q_val, 'dq': 0.0, 'tau': 0.0, 'kp': kp_val, 'kd': kd_val}
            
            self.robot.controlMIT(mit_cmd)
            time.sleep(0.001)
        
        # 失能(使用最终的安全倒向位置)
        mit_cmd = {}
        for sdk_id in start_positions.keys():
            start_q = start_positions.get(sdk_id, 0.0)
            target_q = SAFE_FALL_POSITIONS.get(sdk_id, start_q)
            mit_cmd[sdk_id] = {'q': target_q, 'dq': 0.0, 'tau': 0.0, 'kp': 0.0, 'kd': 0.0}
        self.robot.controlMIT(mit_cmd)
        print("[回放] 缓停止完成")
    
    def _send_cmd(self, mit_cmd: Dict[int, Dict[str, float]], 
                   left_gripper: float = 0.0, right_gripper: float = 0.0) -> bool:
        """发送命令(受频率限制)"""
        if TAKS_SEND_RATE:
            now = time.time()
            if now - self.last_send_time < 1.0 / TAKS_SEND_RATE:
                return False
            self.last_send_time = now
        # 发送到真机
        if self.enable_real and self.robot is not None:
            self.robot.controlMIT(mit_cmd)
            if self.left_gripper is not None:
                kp, kd = SDK_JOINT_GAINS.get(16, (0.5, 0.05))
                self.left_gripper.controlMIT(percent=left_gripper, kp=kp, kd=kd)
            if self.right_gripper is not None:
                kp, kd = SDK_JOINT_GAINS.get(8, (0.5, 0.05))
                self.right_gripper.controlMIT(percent=right_gripper, kp=kp, kd=kd)
        self.send_count += 1
        return True
    
    def _update_mujoco(self, mit_cmd: Dict[int, Dict[str, float]]):
        """更新MuJoCo仿真状态"""
        if self.model is None or self.data is None:
            return
        for sdk_id, cmd in mit_cmd.items():
            if sdk_id in self.joint_mapping:
                qpos_idx = self.joint_mapping[sdk_id]
                self.data.qpos[qpos_idx] = cmd['q']
        mujoco.mj_forward(self.model, self.data)
    
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
        real_str = "ON" if self.enable_real else "OFF"
        status2 = f"时长: {duration:.1f}s | 循环: {'ON' if self.loop else 'OFF'} | 真机: {real_str}"
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
        
        mode_str = "真机+仿真" if self.enable_real else "仅仿真"
        print(f"[回放] 文件: {self.input_path}")
        print(f"[回放] 模式: {mode_str} | 循环: {'ON' if self.loop else 'OFF'}")
        print("[回放] Ctrl+C 安全停止")
        
        if self.headless:
            self._run_headless()
        else:
            self._run_with_viewer()
    
    def _run_headless(self):
        """无头模式回放"""
        try:
            self._connect()
            first_cmd = self.record_data.frames[0].mit_cmd
            self._ramp_up(first_cmd)
            if not self.running:
                return
            self._playback_loop(viewer=None)
        except Exception as e:
            print(f"[回放] 错误: {e}")
        finally:
            self._ramp_down()
            self._disconnect()
    
    def _run_with_viewer(self):
        """有头模式回放(MuJoCo可视化)"""
        try:
            self._connect()
            first_cmd = self.record_data.frames[0].mit_cmd
            self._ramp_up(first_cmd)
            if not self.running:
                return
            with mujoco.viewer.launch_passive(self.model, self.data, 
                                               show_left_ui=False, show_right_ui=False) as viewer:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
                self._playback_loop(viewer=viewer)
        except Exception as e:
            print(f"[回放] 错误: {e}")
        finally:
            self._ramp_down()
            self._disconnect()
    
    def _playback_loop(self, viewer=None):
        """回放循环"""
        loop_count = 0
        print_interval = 0.5
        last_print_time = time.time()
        
        while self.running:
            if viewer is not None and not viewer.is_running():
                break
            
            loop_count += 1
            print(f"[回放] 开始第 {loop_count} 次播放")
            
            play_start = time.time()
            frame_idx = 0
            
            while frame_idx < len(self.record_data.frames) and self.running:
                if viewer is not None and not viewer.is_running():
                    self.running = False
                    break
                
                frame = self.record_data.frames[frame_idx]
                
                # 等待到达帧时间
                target_time = play_start + frame.timestamp
                while time.time() < target_time and self.running:
                    time.sleep(0.001)
                
                if not self.running:
                    break
                
                # 发送命令
                self._send_cmd(frame.mit_cmd, frame.left_gripper, frame.right_gripper)
                # 更新MuJoCo仿真
                self._update_mujoco(frame.mit_cmd)
                self._update_stats()
                
                # 同步viewer
                if viewer is not None:
                    viewer.sync()
                
                # 打印状态
                now = time.time()
                if now - last_print_time >= print_interval:
                    last_print_time = now
                    if self.headless:
                        self.console.clear()
                        self.console.print(self._build_status_table(frame_idx, loop_count, frame.mit_cmd))
                
                frame_idx += 1
            
            if not self.loop:
                break
            
            print(f"[回放] 第 {loop_count} 次播放完成，重新开始...")
        
        print("[回放] 播放结束")


# ==================== 命令行入口 ====================

def parse_args():
    parser = argparse.ArgumentParser(description="半身IK录制回放")
    subparsers = parser.add_subparsers(dest="mode", help="模式(默认record)")
    
    # 公共参数
    parser.add_argument("--host", type=str, default="192.168.5.4", help="taks服务器地址")
    parser.add_argument("--port", type=int, default=5555, help="taks服务器端口")
    parser.add_argument("--no-real", action="store_true", default=False, help="禁用真机控制(仅仿真)")
    parser.add_argument("--headless", action="store_true", default=False, help="无头模式(无GUI)")
    
    # 录制模式
    rec_parser = subparsers.add_parser("record", help="录制模式")
    rec_parser.add_argument("-o", "--output", type=str, default="recording.npz", help="输出文件路径")
    rec_parser.add_argument("--fps", type=int, default=30, help="录制帧率(Hz)")
    
    # 回放模式
    play_parser = subparsers.add_parser("play", help="回放模式")
    play_parser.add_argument("-i", "--input", type=str, required=True, help="输入文件路径")
    play_parser.add_argument("--loop", action="store_true", help="循环播放")
    play_parser.add_argument("--no-ramp-up", action="store_true", default=False, help="禁用缓启动")
    play_parser.add_argument("--no-ramp-down", action="store_true", default=False, help="禁用缓停止")
    play_parser.add_argument("--ramp-up-time", type=float, default=RAMP_UP_TIME, help="缓启动时间(秒)")
    play_parser.add_argument("--ramp-down-time", type=float, default=RAMP_DOWN_TIME, help="缓停止时间(秒)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 公共参数
    enable_real = not args.no_real
    headless = args.headless
    
    # 默认录制模式
    if args.mode is None or args.mode == "record":
        output = getattr(args, 'output', 'recording.npz')
        fps = getattr(args, 'fps', 30)
        recorder = Recorder(
            output_path=output,
            record_fps=fps,
            headless=headless,
            host=args.host,
            port=args.port,
            enable_real=enable_real,
        )
        recorder.run()
    
    elif args.mode == "play":
        player = Player(
            input_path=args.input,
            loop=args.loop,
            host=args.host,
            port=args.port,
            enable_real=enable_real,
            headless=headless,
            ramp_up_time=args.ramp_up_time,
            ramp_down_time=args.ramp_down_time,
            enable_ramp_up=not args.no_ramp_up,
            enable_ramp_down=not args.no_ramp_down,
        )
        player.run()


if __name__ == "__main__":
    main()