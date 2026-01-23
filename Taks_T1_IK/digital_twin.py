"""MuJoCo数字孪生控制器
键盘控制关节，真机同步执行，支持夹爪控制
参考halfbody_ik_vr.py的清晰架构重构
"""

import sys
import argparse
import signal
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from rich.console import Console
from pynput import keyboard

from taks_sdk import taks

# ==================== 配置 ====================

MODELS = {
    "full": Path(__file__).parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml",
    "semi": Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml",
}

TAKS_SEND_RATE = 30
RAMP_UP_TIME = 5.0
RAMP_DOWN_TIME = 5.0

# 半身关节组
JOINT_GROUPS_SEMI = {
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

# 全身关节组（包含腿部）
JOINT_GROUPS_FULL = {
    **JOINT_GROUPS_SEMI,
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                 "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
}

# 关节名到SDK ID映射
JOINT_NAME_TO_SDK_ID = {
    "right_shoulder_pitch_joint": 1, "right_shoulder_roll_joint": 2, "right_shoulder_yaw_joint": 3,
    "right_elbow_joint": 4, "right_wrist_roll_joint": 5, "right_wrist_yaw_joint": 6, "right_wrist_pitch_joint": 7,
    "left_shoulder_pitch_joint": 9, "left_shoulder_roll_joint": 10, "left_shoulder_yaw_joint": 11,
    "left_elbow_joint": 12, "left_wrist_roll_joint": 13, "left_wrist_yaw_joint": 14, "left_wrist_pitch_joint": 15,
    "waist_yaw_joint": 17, "waist_roll_joint": 18, "waist_pitch_joint": 19,
    "neck_yaw_joint": 20, "neck_roll_joint": 21, "neck_pitch_joint": 22,
    "right_hip_pitch_joint": 23, "right_hip_roll_joint": 24, "right_hip_yaw_joint": 25,
    "right_knee_joint": 26, "right_ankle_pitch_joint": 27, "right_ankle_roll_joint": 28,
    "left_hip_pitch_joint": 29, "left_hip_roll_joint": 30, "left_hip_yaw_joint": 31,
    "left_knee_joint": 32, "left_ankle_pitch_joint": 33, "left_ankle_roll_joint": 34,
}

SDK_ID_TO_NAME = {v: k for k, v in JOINT_NAME_TO_SDK_ID.items()}

# SDK关节增益
SDK_JOINT_GAINS = {
    1: (20, 2), 2: (20, 2), 3: (20, 2), 4: (20, 2),
    5: (10, 1), 6: (10, 1), 7: (10, 1), 8: (2.0, 0.2),
    9: (20, 2), 10: (20, 2), 11: (20, 2), 12: (20, 2),
    13: (10, 1), 14: (10, 1), 15: (10, 1), 16: (2.0, 0.2),
    17: (150, 1), 18: (150, 1), 19: (150, 1),
    20: (1.5, 0.1), 21: (1.5, 0.1), 22: (1.5, 0.1),
    23: (50, 50), 24: (150, 50), 25: (150, 50),
    26: (50, 50), 27: (40, 2), 28: (40, 2),
    29: (50, 50), 30: (150, 50), 31: (150, 50),
    32: (50, 50), 33: (40, 2), 34: (40, 2),
}

# 安全模式KP/KD
SAFE_KP_KD = {
    1: (5, 1), 2: (5, 1), 3: (5, 1), 4: (5, 1),
    5: (2.5, 1), 6: (2.5, 1), 7: (2.5, 1), 8: (2.0, 0.2),
    9: (5, 1), 10: (5, 1), 11: (5, 1), 12: (5, 1),
    13: (2.5, 1), 14: (2.5, 1), 15: (2.5, 1), 16: (2.0, 0.2),
    17: (25, 1), 18: (25, 1), 19: (25, 1),
    20: (1.5, 0.1), 21: (1.5, 0.1), 22: (1.5, 0.1),
    23: (20, 2), 24: (20, 2), 25: (20, 2),
    26: (20, 2), 27: (10, 1), 28: (10, 1),
    29: (20, 2), 30: (20, 2), 31: (20, 2),
    32: (20, 2), 33: (10, 1), 34: (10, 1),
}

# 安全倒向位置
SAFE_FALL_POSITIONS = {
    4: 0.2,     # right_elbow
    12: 0.2,    # left_elbow
    17: 0.0,    # waist_yaw
    18: 0.52,   # waist_roll
    19: -0.45,  # waist_pitch
}

# 夹爪控制参数
GRIPPER_KP = 1.5
GRIPPER_KD = 0.1
GRIPPER_OPEN = 0.0  # 百分比：0.0=打开
GRIPPER_CLOSE = 100.0  # 百分比：100.0=闭合

# 关节调整步长
JOINT_STEP = 0.05


# ==================== 工具函数 ====================

def ease_out(t: float, exp: float = 1.1) -> float:
    """缓出函数：开始快，结束慢（用于ramp up）"""
    return 1.0 - pow(1.0 - t, exp)

def ease_in(t: float, exp: float = 0.9) -> float:
    """缓入函数：开始慢，结束快（用于ramp down）"""
    return pow(t, exp)


# ==================== 数据类 ====================

@dataclass
class RampState:
    """启停渐变状态"""
    active: bool = False
    direction: str = "up"  # "up" or "down"
    start_time: float = 0.0
    progress: float = 0.0

@dataclass
class GripperState:
    """夹爪状态"""
    left_position: float = GRIPPER_OPEN
    right_position: float = GRIPPER_OPEN


# ==================== 控制器 ====================

class DigitalTwinController:
    """数字孪生控制器：键盘控制MuJoCo关节，实时同步到真机"""
    
    def __init__(self, model_type: str = "semi", sim2real: bool = False, 
                 auto_start_sdk: bool = True, headless: bool = False,
                 host: str = "192.168.5.4", port: int = 5555,
                 enable_ramp_up: bool = True, enable_ramp_down: bool = True):
        self.model_type = model_type
        self.sim2real = sim2real
        self.auto_start_sdk = auto_start_sdk
        self.headless = headless
        self.host = host
        self.port = port
        self.enable_ramp_up = enable_ramp_up
        self.enable_ramp_down = enable_ramp_down
        self.console = Console()
        
        # MuJoCo
        xml_path = MODELS[model_type]
        self.model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 索引
        self._init_indices()
        self._init_state()
        
        # SDK
        self.sdk_proc = None
        self.taks_client = None
        self.left_gripper = None
        self.right_gripper = None
        if sim2real:
            if auto_start_sdk and host in ("localhost", "127.0.0.1"):
                self.sdk_proc = self._start_sdk_server()
            self._connect_to_robot()
        
        self.rate = RateLimiter(frequency=50.0, warn=False)
        self.dt = self.rate.dt
        self.send_interval = 1.0 / TAKS_SEND_RATE
        self.last_send_time = 0.0
        
        # 当前选中的关节索引（用于键盘控制）
        self.current_joint_idx = 0
        self.need_send = True  # 是否需要发送到真机
    
    def _init_indices(self):
        """初始化关节索引映射"""
        joint_groups = JOINT_GROUPS_FULL if self.model_type == "full" else JOINT_GROUPS_SEMI
        
        # 构建关节映射列表：[(jname, qpos_idx, sdk_id), ...]
        self.joint_list = []
        for group_name, joint_names in joint_groups.items():
            for jname in joint_names:
                sdk_id = JOINT_NAME_TO_SDK_ID.get(jname)
                if sdk_id:
                    try:
                        jid = self.model.joint(jname).id
                        qpos_idx = self.model.jnt_qposadr[jid]
                        self.joint_list.append((jname, qpos_idx, sdk_id))
                    except:
                        pass
        
        # 按SDK ID排序
        self.joint_list.sort(key=lambda x: x[2])
        
        self.console.print(f"[cyan][Info] 可控关节数: {len(self.joint_list)}[/cyan]")
    
    def _init_state(self):
        """初始化状态"""
        self.init_q = self.data.qpos.copy()
        # 初始progress=1.0，避免启动时无控制（用户可以按Space切换到ramp模式）
        self.ramp_state = RampState(progress=1.0)
        self.gripper_state = GripperState()
        self.last_mit_cmd = None
    
    def _start_sdk_server(self):
        """启动本地SDK服务端"""
        sdk_path = Path(__file__).parent / "taks_sdk" / "SDK_MF.py"
        try:
            proc = subprocess.Popen([sys.executable, str(sdk_path)], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2.0)
            if proc.poll() is not None:
                self.console.print("[red][SDK] 服务端启动失败[/red]")
                return None
            self.console.print(f"[green][SDK] 服务端已启动 (PID: {proc.pid})[/green]")
            return proc
        except Exception as e:
            self.console.print(f"[red][SDK] 启动错误: {e}[/red]")
            return None
    
    def _connect_to_robot(self):
        """连接到真机"""
        try:
            taks.connect(address=self.host, cmd_port=self.port, wait_data=True, timeout=5.0)
            device_type = "Taks-T1" if self.model_type == "full" else "Taks-T1-semibody"
            self.taks_client = taks.register(device_type=device_type)
            self.console.print(f"[green][TAKS] 已注册设备: {device_type}[/green]")
            
            # 注册夹爪
            time.sleep(1.0)
            self.left_gripper = taks.register(device_type="Taks-T1-leftgripper")
            self.console.print("[green][TAKS] 已注册左夹爪[/green]")
            time.sleep(0.5)
            self.right_gripper = taks.register(device_type="Taks-T1-rightgripper")
            self.console.print("[green][TAKS] 已注册右夹爪[/green]")
            
            self.console.print(f"[green][TAKS] 已连接 {self.host}:{self.port}[/green]")
        except Exception as e:
            self.console.print(f"[red][TAKS] 连接失败: {e}[/red]")
            self.sim2real = False
    
    def reset(self):
        """复位到初始位置"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.need_send = True
        self.console.print("[yellow][Reset] 已复位到初始位置[/yellow]")
    
    def start_ramp_up(self):
        """启动缓启动"""
        self.ramp_state = RampState(active=True, direction="up", start_time=time.time(), progress=0.0)
        self.console.print("[cyan][Ramp Up] 启动中...[/cyan]")
    
    def start_ramp_down(self):
        """启动缓停止"""
        self.ramp_state = RampState(active=True, direction="down", start_time=time.time(), progress=1.0)
        self.console.print("[cyan][Ramp Down] 缓停止中...[/cyan]")
    
    def build_mit_cmd(self, ramp_progress: float) -> Dict:
        """构建MIT控制命令"""
        mit_cmd = {}
        for jname, qpos_idx, sdk_id in self.joint_list:
            base_kp, base_kd = SDK_JOINT_GAINS.get(sdk_id, (10, 1))
            safe_kp, safe_kd = SAFE_KP_KD.get(sdk_id, (5, 1))
            
            # 使用ease_out进行kp/kd渐变
            kp = safe_kp + (base_kp - safe_kp) * ease_out(ramp_progress)
            kd = safe_kd + (base_kd - safe_kd) * ease_out(ramp_progress)
            
            # 目标位置
            target_q = float(self.data.qpos[qpos_idx])
            
            # 如果在安全倒向列表中，进行位置插值
            if sdk_id in SAFE_FALL_POSITIONS:
                safe_q = SAFE_FALL_POSITIONS[sdk_id]
                target_q = safe_q + (target_q - safe_q) * ease_out(ramp_progress)
            
            mit_cmd[sdk_id] = {'q': target_q, 'dq': 0.0, 'tau': 0.0, 'kp': kp, 'kd': kd}
        
        return mit_cmd
    
    def send_to_real(self, mit_cmd: Dict):
        """发送命令到真机"""
        if not self.sim2real or not self.taks_client:
            return
        
        now = time.time()
        if now - self.last_send_time < self.send_interval:
            return
        self.last_send_time = now
        
        try:
            self.taks_client.controlMIT(mit_cmd)
        except Exception:
            pass
    
    def send_gripper_cmd(self, ramp_progress: float = 1.0):
        """发送夹爪控制命令（支持缓启动/缓停止）"""
        if not self.sim2real:
            return
        
        try:
            # 夹爪使用独立的渐变逻辑：最小30%增益，避免完全无控制
            min_gain_ratio = 0.3
            gain_ratio = min_gain_ratio + (1.0 - min_gain_ratio) * ease_out(ramp_progress)
            kp = GRIPPER_KP * gain_ratio
            kd = GRIPPER_KD * gain_ratio
            
            if self.left_gripper:
                self.left_gripper.controlMIT(percent=self.gripper_state.left_position, kp=kp, kd=kd)
            if self.right_gripper:
                self.right_gripper.controlMIT(percent=self.gripper_state.right_position, kp=kp, kd=kd)
        except Exception as e:
            pass
    
    def toggle_left_gripper(self):
        """切换左夹爪开关"""
        if self.gripper_state.left_position == GRIPPER_OPEN:
            self.gripper_state.left_position = GRIPPER_CLOSE
            self.console.print("[green][Gripper] 左夹爪闭合[/green]")
        else:
            self.gripper_state.left_position = GRIPPER_OPEN
            self.console.print("[green][Gripper] 左夹爪打开[/green]")
        self.send_gripper_cmd(self.ramp_state.progress)
    
    def toggle_right_gripper(self):
        """切换右夹爪开关"""
        if self.gripper_state.right_position == GRIPPER_OPEN:
            self.gripper_state.right_position = GRIPPER_CLOSE
            self.console.print("[green][Gripper] 右夹爪闭合[/green]")
        else:
            self.gripper_state.right_position = GRIPPER_OPEN
            self.console.print("[green][Gripper] 右夹爪打开[/green]")
        self.send_gripper_cmd(self.ramp_state.progress)
    
    def step(self):
        """单步更新"""
        # Ramp处理
        if self.ramp_state.active:
            elapsed = time.time() - self.ramp_state.start_time
            if self.ramp_state.direction == "up":
                if elapsed >= RAMP_UP_TIME:
                    self.ramp_state.active = False
                    self.ramp_state.progress = 1.0
                    self.console.print("[green][Ramp Up] 完成[/green]")
                else:
                    self.ramp_state.progress = elapsed / RAMP_UP_TIME
            else:  # down
                if elapsed >= RAMP_DOWN_TIME:
                    self.ramp_state.active = False
                    self.ramp_state.progress = 0.0
                    self.console.print("[green][Ramp Down] 完成[/green]")
                    if self.sim2real and self.taks_client:
                        taks.disconnect()
                else:
                    self.ramp_state.progress = 1.0 - elapsed / RAMP_DOWN_TIME
        
        # MuJoCo前向动力学
        mujoco.mj_forward(self.model, self.data)
        
        # 构建MIT命令
        mit_cmd = self.build_mit_cmd(self.ramp_state.progress)
        self.last_mit_cmd = mit_cmd
        
        # 发送到真机
        if self.need_send:
            self.send_to_real(mit_cmd)
            self.send_gripper_cmd(self.ramp_state.progress)  # 同步发送夹爪命令
            self.need_send = False
        else:
            self.send_to_real(mit_cmd)
            self.send_gripper_cmd(self.ramp_state.progress)  # 同步发送夹爪命令
    
    def print_current_joint(self):
        """打印当前选中的关节信息"""
        jname, qpos_idx, sdk_id = self.joint_list[self.current_joint_idx]
        self.console.print(
            f"[cyan][{self.current_joint_idx+1}/{len(self.joint_list)}] "
            f"{jname} (SDK:{sdk_id}) = {self.data.qpos[qpos_idx]:.4f}[/cyan]"
        )
    
    def key_callback(self, keycode):
        """键盘回调函数"""
        jname, qpos_idx, sdk_id = self.joint_list[self.current_joint_idx]
        
        if keycode == 265:  # UP - 上一个关节
            self.current_joint_idx = (self.current_joint_idx - 1) % len(self.joint_list)
            self.print_current_joint()
        elif keycode == 264:  # DOWN - 下一个关节
            self.current_joint_idx = (self.current_joint_idx + 1) % len(self.joint_list)
            self.print_current_joint()
        elif keycode == 263:  # LEFT - 减小关节值
            self.data.qpos[qpos_idx] -= JOINT_STEP
            self.need_send = True
            self.print_current_joint()
        elif keycode == 262:  # RIGHT - 增加关节值
            self.data.qpos[qpos_idx] += JOINT_STEP
            self.need_send = True
            self.print_current_joint()
        elif keycode == 82 or keycode == 114:  # R/r - 复位
            self.reset()
        elif keycode == 48:  # 0 - 当前关节归零
            self.data.qpos[qpos_idx] = 0.0
            self.need_send = True
            self.print_current_joint()
        elif keycode == 32:  # Space - 启停
            if not self.ramp_state.active and self.ramp_state.progress < 0.5:
                self.start_ramp_up()
            elif not self.ramp_state.active:
                self.start_ramp_down()
        elif keycode == 81 or keycode == 113:  # Q/q - 左夹爪
            self.toggle_left_gripper()
        elif keycode == 69 or keycode == 101:  # E/e - 右夹爪
            self.toggle_right_gripper()
    
    def run(self):
        """运行控制循环"""
        if self.headless:
            self.console.print("\n[cyan][控制] 无头模式运行中...[/cyan]\n")
            self.console.print("[yellow][提示] 按Ctrl+C停止[/yellow]\n")
            self.console.print("\n[cyan]==================== 操作说明 ====================[/cyan]")
            self.console.print("[yellow]  ↑/↓     = 切换关节[/yellow]")
            self.console.print("[yellow]  ←/→     = 减少/增加关节值[/yellow]")
            self.console.print("[yellow]  R       = 复位所有关节[/yellow]")
            self.console.print("[yellow]  0       = 当前关节归零[/yellow]")
            self.console.print("[yellow]  Space   = 启停[/yellow]")
            self.console.print("[yellow]  Q       = 左夹爪开关[/yellow]")
            self.console.print("[yellow]  E       = 右夹爪开关[/yellow]")
            self.console.print("[cyan]=================================================[/cyan]\n")
            
            self.print_current_joint()
            
            # 自动启动ramp up
            if self.enable_ramp_up and self.sim2real:
                self.start_ramp_up()
            
            # 启动pynput键盘监听
            def on_press(key):
                try:
                    if hasattr(key, 'char') and key.char:
                        # 字符键
                        if key.char == 'r' or key.char == 'R':
                            self.reset()
                        elif key.char == '0':
                            jname, qpos_idx, sdk_id = self.joint_list[self.current_joint_idx]
                            self.data.qpos[qpos_idx] = 0.0
                            self.need_send = True
                            self.print_current_joint()
                        elif key.char == 'q' or key.char == 'Q':
                            self.toggle_left_gripper()
                        elif key.char == 'e' or key.char == 'E':
                            self.toggle_right_gripper()
                        elif key.char == ' ':
                            if not self.ramp_state.active and self.ramp_state.progress < 0.5:
                                self.start_ramp_up()
                            elif not self.ramp_state.active:
                                self.start_ramp_down()
                    else:
                        # 特殊键
                        jname, qpos_idx, sdk_id = self.joint_list[self.current_joint_idx]
                        if key == keyboard.Key.up:
                            self.current_joint_idx = (self.current_joint_idx - 1) % len(self.joint_list)
                            self.print_current_joint()
                        elif key == keyboard.Key.down:
                            self.current_joint_idx = (self.current_joint_idx + 1) % len(self.joint_list)
                            self.print_current_joint()
                        elif key == keyboard.Key.left:
                            self.data.qpos[qpos_idx] -= JOINT_STEP
                            self.need_send = True
                            self.print_current_joint()
                        elif key == keyboard.Key.right:
                            self.data.qpos[qpos_idx] += JOINT_STEP
                            self.need_send = True
                            self.print_current_joint()
                except Exception:
                    pass
            
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            
            try:
                while True:
                    self.step()
                    self.rate.sleep()
            finally:
                listener.stop()
        else:
            with mujoco.viewer.launch_passive(
                self.model, self.data, 
                show_left_ui=False, show_right_ui=False,
                key_callback=self.key_callback
            ) as viewer:
                mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
                
                self.console.print("\n[cyan]==================== 操作说明 ====================[/cyan]")
                self.console.print("[yellow]  ↑/↓     = 切换关节[/yellow]")
                self.console.print("[yellow]  ←/→     = 减少/增加关节值[/yellow]")
                self.console.print("[yellow]  R       = 复位所有关节[/yellow]")
                self.console.print("[yellow]  0       = 当前关节归零[/yellow]")
                self.console.print("[yellow]  Space   = 启停（缓启动/缓停止）[/yellow]")
                self.console.print("[yellow]  Q       = 左夹爪开关[/yellow]")
                self.console.print("[yellow]  E       = 右夹爪开关[/yellow]")
                self.console.print("[cyan]=================================================[/cyan]\n")
                
                self.print_current_joint()
                
                # GUI模式也自动启动ramp up（如果启用且连接真机）
                if self.enable_ramp_up and self.sim2real:
                    self.start_ramp_up()
                
                while viewer.is_running():
                    self.step()
                    mujoco.mj_camlight(self.model, self.data)
                    viewer.sync()
                    self.rate.sleep()
    
    def close(self):
        """关闭控制器"""
        if self.ramp_state.progress > 0 and self.sim2real and self.enable_ramp_down:
            self.start_ramp_down()
            while self.ramp_state.active:
                try:
                    self.step()
                except Exception:
                    break
                time.sleep(0.01)
        
        if self.taks_client:
            try:
                taks.disconnect()
            except Exception:
                pass
        
        if self.sdk_proc:
            try:
                self.sdk_proc.terminate()
                self.sdk_proc.wait(timeout=2)
            except Exception:
                pass
        
        self.console.print("[green][Controller] 已关闭[/green]")


# ==================== 主程序 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo数字孪生控制器 - SIM2REAL")
    parser.add_argument("--model", type=str, default="semi", choices=["full", "semi"], 
                       help="模型类型 (full=全身, semi=半身)")
    parser.add_argument("--headless", action="store_true", default=False, 
                       help="无头模式（无GUI）")
    parser.add_argument("--host", type=str, default="192.168.5.16", 
                       help="TAKS服务器地址")
    parser.add_argument("--port", type=int, default=5555, 
                       help="TAKS服务器端口")
    parser.add_argument("--no-real", action="store_true", default=False, 
                       help="禁用真机（仅仿真）")
    parser.add_argument("--no-ramp-up", action="store_true", default=False, 
                       help="禁用缓启动")
    parser.add_argument("--no-ramp-down", action="store_true", default=False, 
                       help="禁用缓停止")
    
    args = parser.parse_args()
    
    controller = DigitalTwinController(
        model_type=args.model,
        sim2real=not args.no_real,
        auto_start_sdk=True,
        headless=args.headless,
        host=args.host,
        port=args.port,
        enable_ramp_up=not args.no_ramp_up,
        enable_ramp_down=not args.no_ramp_down
    )
    
    def signal_handler(signum, frame):
        print("\n[Info] 收到中断信号，正在关闭...")
        controller.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n[Info] 用户中断")
    finally:
        controller.close()