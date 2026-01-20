"""
半身IK录制回放
录制: 记录MIT命令参数到.npz文件
回放: 加载录制文件，支持循环播放、缓启动/停止
"""

import sys
import argparse
import signal
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import mujoco
import mujoco.viewer
from rich.console import Console

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
    timestamp: float
    mit_cmd: Dict[int, Dict[str, float]]
    left_gripper: float = 0.0
    right_gripper: float = 0.0


@dataclass
class RecordData:
    frames: List[RecordFrame] = field(default_factory=list)
    start_time: float = 0.0
    
    def add_frame(self, mit_cmd: Dict, left_gripper: float = 0.0, right_gripper: float = 0.0):
        if not self.frames:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time
        self.frames.append(RecordFrame(timestamp=timestamp, mit_cmd=mit_cmd.copy(),
                                       left_gripper=left_gripper, right_gripper=right_gripper))
    
    def save(self, filepath: str):
        if not self.frames:
            print("[录制] 无数据")
            return
        sdk_ids = sorted(self.frames[0].mit_cmd.keys())
        n_frames, n_joints = len(self.frames), len(sdk_ids)
        
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
        
        left_gripper_arr = np.array([f.left_gripper for f in self.frames], dtype=np.float64)
        right_gripper_arr = np.array([f.right_gripper for f in self.frames], dtype=np.float64)
        
        np.savez(filepath, timestamps=timestamps, sdk_ids=np.array(sdk_ids, dtype=np.int32),
                 q=q_arr, dq=dq_arr, tau=tau_arr, kp=kp_arr, kd=kd_arr,
                 left_gripper=left_gripper_arr, right_gripper=right_gripper_arr)
        print(f"[录制] 已保存 {n_frames} 帧到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RecordData':
        data = np.load(filepath)
        timestamps, sdk_ids = data['timestamps'], data['sdk_ids'].tolist()
        q_arr, dq_arr, tau_arr = data['q'], data['dq'], data['tau']
        kp_arr, kd_arr = data['kp'], data['kd']
        left_gripper_arr = data.get('left_gripper', np.zeros(len(timestamps)))
        right_gripper_arr = data.get('right_gripper', np.zeros(len(timestamps)))
        
        record = cls()
        for i in range(len(timestamps)):
            mit_cmd = {sdk_ids[j]: {'q': float(q_arr[i, j]), 'dq': float(dq_arr[i, j]),
                                    'tau': float(tau_arr[i, j]), 'kp': float(kp_arr[i, j]),
                                    'kd': float(kd_arr[i, j])} for j in range(len(sdk_ids))}
            record.frames.append(RecordFrame(timestamp=float(timestamps[i]), mit_cmd=mit_cmd,
                                            left_gripper=float(left_gripper_arr[i]),
                                            right_gripper=float(right_gripper_arr[i])))
        print(f"[回放] 已加载 {len(record.frames)} 帧, 时长 {timestamps[-1]:.2f}s")
        return record


# ==================== 录制器 ====================

class Recorder:
    def __init__(self, output_path: str, record_fps: int = 30, headless: bool = False,
                 host: str = "192.168.5.4", port: int = 5555, enable_real: bool = True):
        self.output_path = output_path
        self.record_fps = record_fps
        self.record_interval = 1.0 / record_fps if record_fps > 0 else 0.0
        self.headless = headless
        self.controller = HalfBodyIKController(sim2real=enable_real, headless=headless, host=host, port=port)
        self.record_data = RecordData()
        self.running = True
        self.shutdown_requested = False
        self.console = Console()
        
        # 夹爪控制器（复用controller中的夹爪对象）
        self.left_gripper = self.controller.left_gripper
        self.right_gripper = self.controller.right_gripper
        
        # 禁用控制器的缓启动/停止(录制时不需要)
        self.controller.enable_ramp_up = False
        self.controller.enable_ramp_down = False
        self.controller.ramp_state.active = False
        self.controller.ramp_state.progress = 1.0
    
    def _signal_handler(self, signum, frame):
        if self.shutdown_requested:
            print("\n[强制退出]")
            sys.exit(1)
        print("\n[收到退出信号] 停止录制...")
        self.shutdown_requested = True
        self.running = False
    
    def run(self):
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
            self.controller.close()
    
    def _run_headless(self):
        self._wait_calibration()
        if not self.running:
            return
        self._record_loop(viewer=None)
    
    def _run_with_viewer(self):
        def key_callback(keycode):
            if keycode == 259:
                self.controller.reset()
            elif keycode == 67:  # C键校准
                self.controller.calibrate()
        
        with mujoco.viewer.launch_passive(self.controller.model, self.controller.data,
                                          show_left_ui=False, show_right_ui=False,
                                          key_callback=key_callback) as viewer:
            mujoco.mjv_defaultFreeCamera(self.controller.model, viewer.cam)
            self._wait_calibration(viewer)
            if not self.running:
                return
            self._record_loop(viewer=viewer)
    
    def _wait_calibration(self, viewer=None):
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
        print("[录制] 开始录制帧数据...")
        frame_count = 0
        start_time = time.time()
        last_record_time = 0.0
        print_interval = 1.0
        last_print_time = time.time()
        
        while self.running:
            if viewer is not None and not viewer.is_running():
                break
            
            mit_cmd = self.controller.step()
            vr_data = self.controller.vr.data
            
            left_gripper = vr_data.left_hand.gripper * 100.0
            right_gripper = vr_data.right_hand.gripper * 100.0
            
            # 发送夹爪控制命令到真机
            if self.controller.sim2real:
                if self.left_gripper:
                    self.left_gripper.controlMIT(percent=left_gripper, kp=2.0, kd=0.2)
                if self.right_gripper:
                    self.right_gripper.controlMIT(percent=right_gripper, kp=2.0, kd=0.2)
            
            now = time.time()
            if now - last_record_time >= self.record_interval:
                self.record_data.add_frame(mit_cmd, left_gripper, right_gripper)
                frame_count += 1
                last_record_time = now
            
            if viewer is not None:
                mujoco.mj_camlight(self.controller.model, self.controller.data)
                viewer.sync()
            
            if now - last_print_time >= print_interval:
                last_print_time = now
                elapsed = now - start_time
                print(f"[录制] 帧数: {frame_count} | 时长: {elapsed:.1f}s")
            
            self.controller.rate.sleep()


# ==================== 回放器 ====================

class Player:
    def __init__(self, input_path: str, loop: bool = False, enable_real: bool = True,
                 headless: bool = False, host: str = "192.168.5.4", port: int = 5555,
                 ramp_up_time: float = RAMP_UP_TIME, ramp_down_time: float = RAMP_DOWN_TIME,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        self.loop = loop
        self.enable_real = enable_real
        self.headless = headless
        self.host = host
        self.port = port
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        self.running = True
        self.console = Console()
        
        self.record_data = RecordData.load(input_path)
        if not self.record_data.frames:
            raise ValueError("录制文件为空")
        
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        self.joint_mapping = {}
        for jname, sdk_id in JOINT_NAME_TO_SDK_ID.items():
            try:
                self.joint_mapping[sdk_id] = self.model.jnt_qposadr[self.model.joint(jname).id]
            except:
                pass
        
        self.taks_client = None
        self.left_gripper = None
        self.right_gripper = None
        if enable_real:
            taks.connect(address=host, cmd_port=port, wait_data=True, timeout=2.0)
            self.taks_client = taks.register(device_type="Taks-T1-semibody")
            self.left_gripper = taks.register(device_type="Taks-T1-leftgripper")
            self.right_gripper = taks.register(device_type="Taks-T1-rightgripper")
        
        self.last_send_time = 0.0
        self.send_interval = 1.0 / TAKS_SEND_RATE
    
    def _ramp_up(self, target_cmd: Dict):
        if not self.enable_ramp_up or not self.enable_real or not self.taks_client:
            return
        print(f"[回放] 缓启动 ({self.ramp_up_time}s)...")
        
        start = time.time()
        while time.time() - start < self.ramp_up_time and self.running:
            t = (time.time() - start) / self.ramp_up_time
            scale = ease_out(t)
            mit_cmd = {}
            for sdk_id, cmd in target_cmd.items():
                kp, kd = SDK_JOINT_GAINS.get(sdk_id, (10, 1))
                safe_kp, safe_kd = SAFE_KP_KD.get(sdk_id, (5, 1))
                mit_cmd[sdk_id] = {
                    'q': cmd['q'] * t, 'dq': 0.0, 'tau': 0.0,
                    'kp': safe_kp + (kp - safe_kp) * scale,
                    'kd': safe_kd + (kd - safe_kd) * scale
                }
            self.taks_client.controlMIT(mit_cmd)
            time.sleep(0.01)
        print("[回放] 缓启动完成")
    
    def _ramp_down(self):
        if not self.enable_ramp_down or not self.enable_real or not self.taks_client:
            return
        print(f"[回放] 缓停止 ({self.ramp_down_time}s)...")
        start = time.time()
        while time.time() - start < self.ramp_down_time:
            t = (time.time() - start) / self.ramp_down_time
            scale = 1.0 - ease_in(t)
            mit_cmd = {}
            for sdk_id in self.record_data.frames[0].mit_cmd.keys():
                kp, kd = SDK_JOINT_GAINS.get(sdk_id, (10, 1))
                safe_kp, safe_kd = SAFE_KP_KD.get(sdk_id, (5, 1))
                target_q = SAFE_FALL_POSITIONS.get(sdk_id, 0.0)
                mit_cmd[sdk_id] = {
                    'q': target_q, 'dq': 0.0, 'tau': 0.0,
                    'kp': safe_kp + (kp - safe_kp) * scale,
                    'kd': safe_kd + (kd - safe_kd) * scale
                }
            self.taks_client.controlMIT(mit_cmd)
            time.sleep(0.01)
        taks.disconnect()
        print("[回放] 缓停止完成")
    
    def _send_cmd(self, mit_cmd: Dict, left_gripper: float = 0.0, right_gripper: float = 0.0):
        now = time.time()
        if now - self.last_send_time < self.send_interval:
            return
        self.last_send_time = now
        if self.enable_real and self.taks_client:
            self.taks_client.controlMIT(mit_cmd)
        # 发送夹爪控制命令
        if self.enable_real:
            if self.left_gripper:
                self.left_gripper.controlMIT(percent=left_gripper, kp=2.0, kd=0.2)
            if self.right_gripper:
                self.right_gripper.controlMIT(percent=right_gripper, kp=2.0, kd=0.2)
    
    def _update_mujoco(self, mit_cmd: Dict):
        for sdk_id, cmd in mit_cmd.items():
            if sdk_id in self.joint_mapping:
                self.data.qpos[self.joint_mapping[sdk_id]] = cmd['q']
        mujoco.mj_forward(self.model, self.data)
    
    def _signal_handler(self, signum, frame):
        print("\n[回放] 停止...")
        self.running = False
    
    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"[回放] 模式: {'真机+仿真' if self.enable_real else '仅仿真'}")
        
        try:
            first_cmd = self.record_data.frames[0].mit_cmd
            self._ramp_up(first_cmd)
            
            with mujoco.viewer.launch_passive(self.model, self.data,
                                              show_left_ui=False, show_right_ui=False) as viewer:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
                
                loop_count = 0
                while self.running and viewer.is_running():
                    loop_count += 1
                    print(f"[回放] 第 {loop_count} 次播放")
                    
                    play_start = time.time()
                    for frame in self.record_data.frames:
                        if not self.running or not viewer.is_running():
                            break
                        
                        target_time = play_start + frame.timestamp
                        while time.time() < target_time and self.running:
                            time.sleep(0.001)
                        
                        self._send_cmd(frame.mit_cmd, frame.left_gripper, frame.right_gripper)
                        self._update_mujoco(frame.mit_cmd)
                        viewer.sync()
                    
                    if not self.loop:
                        break
        finally:
            self._ramp_down()
            if self.taks_client:
                taks.disconnect()


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="半身IK录制回放")
    subparsers = parser.add_subparsers(dest="mode", help="模式(默认record)")
    
    # 公共参数
    parser.add_argument("--host", type=str, default="192.168.5.16", help="taks服务器地址")
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
    
    args = parser.parse_args()
    
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