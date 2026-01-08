"""
VR Pose UDP Receiver
接收来自Unity VR的位姿数据，使用rich库美化打印

协议: JSON over UDP
端口: 5005 (默认)
"""

import socket
import json
import time
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

# 配置
UDP_IP = "0.0.0.0"  # 监听所有网卡
UDP_PORT = 7000
BUFFER_SIZE = 4096

console = Console()

# 用于状态变更检测和显示
last_tracking_state = None  # 上一次的追踪状态
STATE_CHANGE_DISPLAY_DURATION = 2.0  # 状态变更在界面上高亮显示的持续时间（秒）
state_change_time = 0  # 状态变更的时间

# 按钮双击事件显示
BUTTON_EVENT_DISPLAY_DURATION = 1.5  # 按钮事件显示持续时间（秒）
button_event_times = {
    "leftX": 0,
    "leftY": 0,
    "rightA": 0,
    "rightB": 0
}


def format_position(pos):
    """格式化位置数组"""
    if pos is None:
        return "N/A"
    return f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"


def format_quaternion(quat):
    """
    格式化四元数数组 (w, x, y, z)
    
    注意：Unity发送的四元数需要对虚部取反才能在MuJoCo/Isaac Sim中正确使用
    原因：Unity使用"主动旋转"惯例（旋转物体），而MuJoCo/Isaac Sim使用"被动旋转"惯例（旋转坐标系）
    转换公式：(w, x, y, z) -> (w, -x, -y, -z)
    这里显示的是取反后的值，可直接用于MuJoCo/Isaac Sim
    """
    if quat is None:
        return "N/A"
    # 对虚部取反，转换为MuJoCo/Isaac Sim惯例
    w, x, y, z = quat[0], -quat[1], -quat[2], -quat[3]
    return f"(w:{w:+.3f}, x:{x:+.3f}, y:{y:+.3f}, z:{z:+.3f})"


def format_gripper(gripper):
    """格式化夹爪值"""
    if gripper is None:
        return "N/A"
    return f"{gripper:.1f}%"


def get_active_button_events():
    """获取当前活跃的按钮事件列表"""
    current_time = time.time()
    active_events = []
    button_names = {
        "leftX": "🆇 左手X",
        "leftY": "🆈 左手Y",
        "rightA": "🅰️ 右手A",
        "rightB": "🅱️ 右手B"
    }
    for btn, event_time in button_event_times.items():
        if current_time - event_time < BUTTON_EVENT_DISPLAY_DURATION:
            active_events.append(button_names[btn])
    return active_events


def create_display_table(data, packet_count, fps, tracking_enabled=False, state_just_changed=False):
    """创建显示表格"""
    # 获取活跃的按钮事件
    active_events = get_active_button_events()
    
    # 根据追踪状态显示不同的标题和样式
    if tracking_enabled:
        if state_just_changed:
            title = f"[bold green blink]>>> TRACKING ENABLED <<<[/bold green blink] | Packets: {packet_count} | FPS: {fps:.1f}"
            border_style = "bold green"
        else:
            title = f"[bold green]● TRACKING[/bold green] | Packets: {packet_count} | FPS: {fps:.1f}"
            border_style = "green"
    else:
        if state_just_changed:
            title = f"[bold red blink]>>> TRACKING STOPPED <<<[/bold red blink] | Packets: {packet_count} | FPS: {fps:.1f}"
            border_style = "bold red"
        else:
            title = f"[dim]○ STOPPED[/dim] | Packets: {packet_count} | FPS: {fps:.1f}"
            border_style = "dim"
    
    # 如果有活跃的按钮事件，添加到标题
    if active_events:
        events_str = " | ".join(active_events)
        title += f" | [bold yellow blink]双击: {events_str}[/bold yellow blink]"
        border_style = "bold yellow"
    
    table = Table(title=title, show_header=True, header_style="bold magenta", border_style=border_style)
    
    table.add_column("部位", style="cyan", width=12)
    table.add_column("位置 (x, y, z)", style="green", width=32)
    table.add_column("四元数 (w, x, y, z)", style="yellow", width=42)
    table.add_column("夹爪", style="blue", width=10)
    
    # Head
    head = data.get("head", {})
    table.add_row(
        "🎯 Head",
        format_position(head.get("position")),
        format_quaternion(head.get("quaternion")),
        "-"
    )
    
    # Left Hand
    left = data.get("leftHand", {})
    table.add_row(
        "🤚 Left",
        format_position(left.get("position")),
        format_quaternion(left.get("quaternion")),
        format_gripper(left.get("gripper"))
    )
    
    # Right Hand
    right = data.get("rightHand", {})
    table.add_row(
        "✋ Right",
        format_position(right.get("position")),
        format_quaternion(right.get("quaternion")),
        format_gripper(right.get("gripper"))
    )
    
    return table


def check_state_change(tracking_enabled):
    """检查追踪状态是否变更"""
    global last_tracking_state, state_change_time
    
    if last_tracking_state is None:
        # 首次接收数据，记录状态
        last_tracking_state = tracking_enabled
        state_change_time = time.time()
        return True  # 首次也视为状态变更
    
    if tracking_enabled != last_tracking_state:
        # 状态发生变化
        last_tracking_state = tracking_enabled
        state_change_time = time.time()
        return True
    
    return False


def check_button_events(data):
    """检查按钮双击事件"""
    global button_event_times
    
    button_events = data.get("buttonEvents", {})
    current_time = time.time()
    
    # 检查每个按钮的双击事件
    if button_events.get("leftX", False):
        if current_time - button_event_times["leftX"] > 0.5:  # 防止重复触发
            button_event_times["leftX"] = current_time
            console.print("[bold yellow]🆇 左手X按钮双击![/bold yellow]")
    
    if button_events.get("leftY", False):
        if current_time - button_event_times["leftY"] > 0.5:
            button_event_times["leftY"] = current_time
            console.print("[bold yellow]🆈 左手Y按钮双击![/bold yellow]")
    
    if button_events.get("rightA", False):
        if current_time - button_event_times["rightA"] > 0.5:
            button_event_times["rightA"] = current_time
            console.print("[bold yellow]🅰️ 右手A按钮双击![/bold yellow]")
    
    if button_events.get("rightB", False):
        if current_time - button_event_times["rightB"] > 0.5:
            button_event_times["rightB"] = current_time
            console.print("[bold yellow]🅱️ 右手B按钮双击![/bold yellow]")


def main():
    # 创建UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    
    console.print(Panel.fit(
        f"[bold green]VR UDP Receiver 启动[/bold green]\n"
        f"监听地址: [cyan]{UDP_IP}:{UDP_PORT}[/cyan]\n"
        f"等待数据...",
        title="🎮 VR Pose Receiver"
    ))
    
    packet_count = 0
    last_data = {}
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0.0
    current_tracking_state = False  # 当前追踪状态
    
    try:
        with Live(console=console, refresh_per_second=30) as live:
            while True:
                try:
                    data_bytes, addr = sock.recvfrom(BUFFER_SIZE)
                    data = json.loads(data_bytes.decode('utf-8'))
                    
                    packet_count += 1
                    fps_counter += 1
                    last_data = data
                    
                    # 计算FPS
                    elapsed = time.time() - fps_start_time
                    if elapsed >= 1.0:
                        current_fps = fps_counter / elapsed
                        fps_counter = 0
                        fps_start_time = time.time()
                    
                    # 获取追踪状态
                    current_tracking_state = data.get("trackingEnabled", False)
                    
                    # 检查状态是否变更
                    check_state_change(current_tracking_state)
                    
                    # 检查按钮双击事件
                    check_button_events(data)
                    
                    # 判断是否在状态变更高亮显示期间
                    state_just_changed = (time.time() - state_change_time) < STATE_CHANGE_DISPLAY_DURATION
                    
                    # 更新显示
                    layout = Layout()
                    
                    # 构建状态行
                    status_text = f"Timestamp: {data.get('timestamp', 'N/A'):.3f}s | From: {addr[0]}:{addr[1]}"
                    if current_tracking_state:
                        status_text = f"[bold green]▶ TRACKING ACTIVE[/bold green] | " + status_text
                    else:
                        status_text = f"[dim]■ TRACKING STOPPED[/dim] | " + status_text
                    
                    layout.split_column(
                        Layout(create_display_table(data, packet_count, current_fps, current_tracking_state, state_just_changed)),
                        Layout(Text.from_markup(status_text), size=1)
                    )
                    live.update(layout)
                    
                except BlockingIOError:
                    # 没有数据，短暂休眠
                    time.sleep(0.001)
                except json.JSONDecodeError as e:
                    console.print(f"[red]JSON解析错误: {e}[/red]")
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]接收器已停止[/yellow]")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
