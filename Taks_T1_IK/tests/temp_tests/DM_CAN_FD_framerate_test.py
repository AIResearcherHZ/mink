#!/usr/bin/env python3
"""
DM_CAN_FD controlMIT 和 getPosition 帧率测试程序

测试 DM_CAN_FD.py 的 controlMIT 发送帧率和 getPosition 读取帧率
使用 rich 库实时显示测试结果

sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on
pip install rich python-can
"""

import sys
import time
import os
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError:
    print("请先安装 rich 库: pip install rich")
    exit(1)

from libs.drivers.DM_CAN_FD import Motor, MotorControlFD, DM_Motor_Type

# ============ 测试配置 ============
CAN_INTERFACE = "can1"  # CAN 接口名称
TEST_MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]  # 测试的电机ID列表 (右臂)
DM_Motor_Type = DM_Motor_Type.DM4340  # 电机类型
TEST_DURATION = 5.0  # 每个频率测试时长(秒)
TARGET_HZ_LIST = [100, 200, 300, 400, 500]  # 目标测试频率列表

# 测试参数 (安全值，不会让电机动)
TEST_KP = 0.0
TEST_KD = 0.0
TEST_Q = 0.0
TEST_DQ = 0.0
TEST_TAU = 0.0


class FrameRateTester:
    """帧率测试器"""
    
    def __init__(self, can_interface: str, motor_ids: list, DM_Motor_Type):
        self.can_interface = can_interface
        self.motor_ids = motor_ids
        self.DM_Motor_Type = DM_Motor_Type
        
        # 统计数据
        self.send_count = 0
        self.recv_count = 0
        self.send_errors = 0
        self.recv_errors = 0
        
        # 帧率计算
        self.send_times = deque(maxlen=1000)
        self.recv_times = deque(maxlen=1000)
        
        # 延迟统计
        self.latencies = deque(maxlen=1000)
        
        # 控制器和电机
        self.controller = None
        self.motors = {}
        
        # 线程控制
        self.running = False
        self.test_running = False
        
    def start(self):
        """启动测试器"""
        try:
            self.controller = MotorControlFD(can_interface=self.can_interface)
            
            # 添加电机
            for motor_id in self.motor_ids:
                motor = Motor(self.DM_Motor_Type, motor_id, motor_id + 0x80)
                self.controller.addMotor(motor)
                self.motors[motor_id] = motor
            
            self.running = True
            return True
        except Exception as e:
            print(f"初始化失败: {e}")
            return False
    
    def stop(self):
        """停止测试器"""
        self.running = False
        self.test_running = False
        if self.controller:
            self.controller.close()
    
    def reset_stats(self):
        """重置统计数据"""
        self.send_count = 0
        self.recv_count = 0
        self.send_errors = 0
        self.recv_errors = 0
        self.send_times.clear()
        self.recv_times.clear()
        self.latencies.clear()
    
    def test_controlMIT_hz(self, target_hz: float, duration: float = 5.0) -> dict:
        """测试 controlMIT 发送帧率
        
        Args:
            target_hz: 目标频率
            duration: 测试时长(秒)
        
        Returns:
            测试结果字典
        """
        self.reset_stats()
        period = 1.0 / target_hz
        motor_count = len(self.motors)
        
        start_time = time.perf_counter()
        cycle_count = 0
        
        while time.perf_counter() - start_time < duration:
            cycle_start = time.perf_counter()
            
            # 发送所有电机的 controlMIT 命令
            for motor_id, motor in self.motors.items():
                try:
                    send_start = time.perf_counter()
                    self.controller.controlMIT(motor, TEST_KP, TEST_KD, TEST_Q, TEST_DQ, TEST_TAU)
                    send_end = time.perf_counter()
                    
                    self.send_count += 1
                    self.send_times.append(send_end)
                    self.latencies.append((send_end - send_start) * 1000)  # ms
                except Exception as e:
                    self.send_errors += 1
            
            cycle_count += 1
            
            # 精确等待
            elapsed = time.perf_counter() - cycle_start
            if elapsed < period:
                remaining = period - elapsed
                if remaining > 0.001:
                    time.sleep(remaining - 0.0005)
                # 忙等待剩余时间
                while time.perf_counter() - cycle_start < period:
                    pass
        
        total_time = time.perf_counter() - start_time
        actual_hz = cycle_count / total_time
        frames_per_sec = self.send_count / total_time
        error_rate = self.send_errors / (self.send_count + self.send_errors) * 100 if (self.send_count + self.send_errors) > 0 else 0
        
        # 延迟统计
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        
        return {
            'target_hz': target_hz,
            'actual_hz': actual_hz,
            'frames_per_sec': frames_per_sec,
            'total_frames': self.send_count,
            'errors': self.send_errors,
            'error_rate': error_rate,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'motor_count': motor_count,
            'duration': total_time,
        }
    
    def test_getPosition_hz(self, target_hz: float, duration: float = 5.0) -> dict:
        """测试 getPosition 读取帧率
        
        Args:
            target_hz: 目标频率
            duration: 测试时长(秒)
        
        Returns:
            测试结果字典
        """
        self.reset_stats()
        period = 1.0 / target_hz
        motor_count = len(self.motors)
        
        start_time = time.perf_counter()
        cycle_count = 0
        
        while time.perf_counter() - start_time < duration:
            cycle_start = time.perf_counter()
            
            # 读取所有电机的位置
            for motor_id, motor in self.motors.items():
                try:
                    read_start = time.perf_counter()
                    pos = motor.getPosition()
                    vel = motor.getVelocity()
                    tau = motor.getTorque()
                    read_end = time.perf_counter()
                    
                    self.recv_count += 1
                    self.recv_times.append(read_end)
                    self.latencies.append((read_end - read_start) * 1000)  # ms
                except Exception as e:
                    self.recv_errors += 1
            
            cycle_count += 1
            
            # 精确等待
            elapsed = time.perf_counter() - cycle_start
            if elapsed < period:
                remaining = period - elapsed
                if remaining > 0.001:
                    time.sleep(remaining - 0.0005)
                while time.perf_counter() - cycle_start < period:
                    pass
        
        total_time = time.perf_counter() - start_time
        actual_hz = cycle_count / total_time
        reads_per_sec = self.recv_count / total_time
        error_rate = self.recv_errors / (self.recv_count + self.recv_errors) * 100 if (self.recv_count + self.recv_errors) > 0 else 0
        
        # 延迟统计
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        max_latency = max(self.latencies) if self.latencies else 0
        min_latency = min(self.latencies) if self.latencies else 0
        
        return {
            'target_hz': target_hz,
            'actual_hz': actual_hz,
            'reads_per_sec': reads_per_sec,
            'total_reads': self.recv_count,
            'errors': self.recv_errors,
            'error_rate': error_rate,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'min_latency_ms': min_latency,
            'motor_count': motor_count,
            'duration': total_time,
        }


def create_result_table(results: list, test_type: str) -> Table:
    """创建结果表格"""
    title = f"📊 {test_type} 帧率测试结果"
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    table.add_column("目标Hz", justify="right", style="cyan")
    table.add_column("实际Hz", justify="right", style="green")
    table.add_column("帧/秒", justify="right", style="yellow")
    table.add_column("总帧数", justify="right")
    table.add_column("错误率", justify="right")
    table.add_column("平均延迟", justify="right", style="blue")
    table.add_column("最大延迟", justify="right", style="red")
    table.add_column("状态", justify="center")
    
    for r in results:
        error_rate = r.get('error_rate', 0)
        if error_rate < 1:
            status = "✅"
        elif error_rate < 5:
            status = "⚠️"
        else:
            status = "❌"
        
        frames_key = 'frames_per_sec' if 'frames_per_sec' in r else 'reads_per_sec'
        total_key = 'total_frames' if 'total_frames' in r else 'total_reads'
        
        table.add_row(
            f"{r['target_hz']}",
            f"{r['actual_hz']:.1f}",
            f"{r[frames_key]:.1f}",
            f"{r[total_key]:,}",
            f"{error_rate:.2f}%",
            f"{r['avg_latency_ms']:.3f}ms",
            f"{r['max_latency_ms']:.3f}ms",
            status
        )
    
    return table


def main():
    console = Console()
    
    console.print(Panel.fit(
        "[bold blue]🔧 DM_CAN_FD 帧率测试程序[/bold blue]\n"
        f"CAN接口: {CAN_INTERFACE}\n"
        f"测试电机: {TEST_MOTOR_IDS}\n"
        f"测试频率: {TARGET_HZ_LIST} Hz\n"
        f"每次测试时长: {TEST_DURATION}s",
        title="配置信息"
    ))
    
    console.print("\n[yellow]正在初始化...[/yellow]")
    
    tester = FrameRateTester(CAN_INTERFACE, TEST_MOTOR_IDS, DM_Motor_Type)
    
    if not tester.start():
        console.print("[red]❌ 初始化失败，请检查 CAN 接口[/red]")
        return
    
    console.print("[green]✅ CAN 接口已打开[/green]\n")
    
    try:
        # ============ controlMIT 测试 ============
        console.print("[bold cyan]━━━ 测试 1: controlMIT 发送帧率 ━━━[/bold cyan]\n")
        
        mit_results = []
        for hz in TARGET_HZ_LIST:
            console.print(f"  测试 {hz} Hz...", end=" ")
            result = tester.test_controlMIT_hz(hz, TEST_DURATION)
            mit_results.append(result)
            
            status = "✅" if result['error_rate'] < 1 else "⚠️" if result['error_rate'] < 5 else "❌"
            console.print(f"实际 {result['actual_hz']:.1f} Hz, 错误 {result['error_rate']:.2f}% {status}")
        
        console.print()
        console.print(create_result_table(mit_results, "controlMIT"))
        
        # ============ getPosition 测试 ============
        console.print("\n[bold cyan]━━━ 测试 2: getPosition 读取帧率 ━━━[/bold cyan]\n")
        
        get_results = []
        for hz in TARGET_HZ_LIST:
            console.print(f"  测试 {hz} Hz...", end=" ")
            result = tester.test_getPosition_hz(hz, TEST_DURATION)
            get_results.append(result)
            
            status = "✅" if result['error_rate'] < 1 else "⚠️" if result['error_rate'] < 5 else "❌"
            console.print(f"实际 {result['actual_hz']:.1f} Hz, 错误 {result['error_rate']:.2f}% {status}")
        
        console.print()
        console.print(create_result_table(get_results, "getPosition"))
        
        # ============ 综合测试 (同时发送和读取) ============
        console.print("\n[bold cyan]━━━ 测试 3: 综合测试 (controlMIT + getPosition) ━━━[/bold cyan]\n")
        
        combined_results = []
        for hz in TARGET_HZ_LIST:
            console.print(f"  测试 {hz} Hz (发送+读取)...", end=" ")
            
            tester.reset_stats()
            period = 1.0 / hz
            start_time = time.perf_counter()
            cycle_count = 0
            
            while time.perf_counter() - start_time < TEST_DURATION:
                cycle_start = time.perf_counter()
                
                for motor_id, motor in tester.motors.items():
                    try:
                        # 发送控制命令
                        tester.controller.controlMIT(motor, TEST_KP, TEST_KD, TEST_Q, TEST_DQ, TEST_TAU)
                        tester.send_count += 1
                        
                        # 读取状态
                        _ = motor.getPosition()
                        tester.recv_count += 1
                    except:
                        tester.send_errors += 1
                
                cycle_count += 1
                
                elapsed = time.perf_counter() - cycle_start
                if elapsed < period:
                    remaining = period - elapsed
                    if remaining > 0.001:
                        time.sleep(remaining - 0.0005)
                    while time.perf_counter() - cycle_start < period:
                        pass
            
            total_time = time.perf_counter() - start_time
            actual_hz = cycle_count / total_time
            error_rate = tester.send_errors / (tester.send_count + tester.send_errors) * 100 if (tester.send_count + tester.send_errors) > 0 else 0
            
            combined_results.append({
                'target_hz': hz,
                'actual_hz': actual_hz,
                'frames_per_sec': tester.send_count / total_time,
                'total_frames': tester.send_count,
                'error_rate': error_rate,
                'avg_latency_ms': 0,
                'max_latency_ms': 0,
            })
            
            status = "✅" if error_rate < 1 else "⚠️" if error_rate < 5 else "❌"
            console.print(f"实际 {actual_hz:.1f} Hz, 错误 {error_rate:.2f}% {status}")
        
        console.print()
        console.print(create_result_table(combined_results, "综合测试"))
        
        # ============ 总结 ============
        console.print("\n" + "="*60)
        console.print("[bold green]📋 测试总结[/bold green]")
        console.print("="*60)
        
        # 找到最高稳定频率
        max_stable_mit = max([r['target_hz'] for r in mit_results if r['error_rate'] < 1], default=0)
        max_stable_get = max([r['target_hz'] for r in get_results if r['error_rate'] < 1], default=0)
        max_stable_combined = max([r['target_hz'] for r in combined_results if r['error_rate'] < 1], default=0)
        
        console.print(f"  controlMIT 最高稳定频率: [green]{max_stable_mit} Hz[/green]")
        console.print(f"  getPosition 最高稳定频率: [green]{max_stable_get} Hz[/green]")
        console.print(f"  综合测试 最高稳定频率: [green]{max_stable_combined} Hz[/green]")
        console.print(f"  测试电机数量: {len(TEST_MOTOR_IDS)}")
        console.print("="*60)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️ 测试被中断[/yellow]")
    finally:
        tester.stop()
        console.print("[green]✅ 测试完成[/green]")


if __name__ == "__main__":
    main()