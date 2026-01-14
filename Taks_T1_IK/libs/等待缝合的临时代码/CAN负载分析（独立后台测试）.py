#!/usr/bin/env python3
"""
CAN FD 总线负载监控程序
使用 rich 库实时显示 CAN 总线状态

sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
pip install rich python-can
"""

import time
import threading
import can
from collections import deque
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
except ImportError:
    print("请先安装 rich 库: pip install rich")
    exit(1)


class CANMonitor:
    """CAN FD 总线监控器"""
    
    # CAN FD 理论参数
    # 标准CAN帧: 11位ID + 数据 + 开销 ≈ 111 bits (8字节数据)
    # CAN FD帧: 更复杂，这里用近似值
    BITS_PER_FRAME_OVERHEAD = 67  # CAN FD 帧开销 (不含数据)
    BITS_PER_DATA_BYTE = 8
    
    def __init__(self, interface='can1', data_bitrate=5_000_000, nominal_bitrate=1_000_000):
        self.interface = interface
        self.data_bitrate = data_bitrate
        self.nominal_bitrate = nominal_bitrate
        
        # 统计数据
        self.rx_count = 0
        self.tx_count = 0
        self.rx_bytes = 0
        self.tx_bytes = 0
        self.rx_errors = 0
        self.tx_errors = 0
        self.total_bits = 0
        
        # 帧率计算
        self.frame_times = deque(maxlen=1000)  # 最近1000帧的时间戳
        self.byte_history = deque(maxlen=100)  # 最近100次采样的字节数
        
        # 按ID统计
        self.id_stats = {}  # {can_id: {'count': n, 'bytes': b, 'last_data': bytes}}
        
        # 线程控制
        self.running = False
        self.bus = None
        self.recv_thread = None
        self.lock = threading.Lock()
        
        # 启动时间
        self.start_time = None
        
    def start(self):
        """启动监控"""
        try:
            self.bus = can.interface.Bus(
                channel=self.interface,
                bustype='socketcan',
                fd=True,
                bitrate=self.nominal_bitrate,
                data_bitrate=self.data_bitrate,
            )
            self.running = True
            self.start_time = time.time()
            self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
            self.recv_thread.start()
            return True
        except Exception as e:
            print(f"无法打开 CAN 接口: {e}")
            return False
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=1.0)
        if self.bus:
            self.bus.shutdown()
    
    def _recv_loop(self):
        """接收循环"""
        while self.running:
            try:
                msg = self.bus.recv(timeout=0.01)
                if msg is not None:
                    self._process_message(msg)
            except Exception as e:
                with self.lock:
                    self.rx_errors += 1
    
    def _process_message(self, msg):
        """处理接收到的消息"""
        now = time.time()
        data_len = len(msg.data)
        
        # 计算帧的比特数 (近似)
        # CAN FD: 仲裁段用nominal_bitrate, 数据段用data_bitrate
        # 简化计算: 假设开销用nominal, 数据用data bitrate
        frame_bits = self.BITS_PER_FRAME_OVERHEAD + data_len * self.BITS_PER_DATA_BYTE
        
        with self.lock:
            self.rx_count += 1
            self.rx_bytes += data_len
            self.total_bits += frame_bits
            self.frame_times.append(now)
            
            # 按ID统计
            can_id = msg.arbitration_id
            if can_id not in self.id_stats:
                self.id_stats[can_id] = {'count': 0, 'bytes': 0, 'last_data': b'', 'last_time': 0}
            self.id_stats[can_id]['count'] += 1
            self.id_stats[can_id]['bytes'] += data_len
            self.id_stats[can_id]['last_data'] = bytes(msg.data)
            self.id_stats[can_id]['last_time'] = now
    
    def get_stats(self):
        """获取统计数据"""
        now = time.time()
        
        with self.lock:
            # 计算帧率 (最近1秒内的帧数)
            recent_frames = sum(1 for t in self.frame_times if now - t < 1.0)
            
            # 计算负载百分比
            # 理论最大: data_bitrate bps
            # 实际: 最近1秒的比特数
            elapsed = now - self.start_time if self.start_time else 1
            avg_bits_per_sec = self.total_bits / elapsed if elapsed > 0 else 0
            
            # 使用最近1秒的帧来估算当前负载
            recent_bits = recent_frames * (self.BITS_PER_FRAME_OVERHEAD + 8 * self.BITS_PER_DATA_BYTE)
            current_load = (recent_bits / self.data_bitrate) * 100 if self.data_bitrate > 0 else 0
            
            # 平均负载
            avg_load = (avg_bits_per_sec / self.data_bitrate) * 100 if self.data_bitrate > 0 else 0
            
            return {
                'rx_count': self.rx_count,
                'tx_count': self.tx_count,
                'rx_bytes': self.rx_bytes,
                'tx_bytes': self.tx_bytes,
                'rx_errors': self.rx_errors,
                'tx_errors': self.tx_errors,
                'frame_rate': recent_frames,
                'current_load': min(current_load, 100),
                'avg_load': min(avg_load, 100),
                'elapsed': elapsed,
                'id_stats': dict(self.id_stats),
                'total_bits': self.total_bits,
            }


def format_bytes(n):
    """格式化字节数"""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    else:
        return f"{n/(1024*1024):.2f} MB"


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def get_id_name(can_id):
    """根据CAN ID获取名称"""
    # 电机响应ID (0x81-0x96)
    if 0x81 <= can_id <= 0x88:
        motor_id = can_id - 0x80
        if motor_id == 0x08:
            return f"右手夹爪"
        return f"右手电机{motor_id}"
    elif 0x89 <= can_id <= 0x90:
        motor_id = can_id - 0x80
        if motor_id == 0x10:
            return f"左手夹爪"
        return f"左手电机{motor_id-8}"
    elif 0x91 <= can_id <= 0x93:
        return f"腰部电机{can_id - 0x90}"
    elif 0x94 <= can_id <= 0x96:
        return f"脖子电机{can_id - 0x93}"
    elif can_id == 0x7FF:
        return "广播命令"
    else:
        return f"ID: {hex(can_id)}"


def create_dashboard(monitor: CANMonitor, console: Console):
    """创建仪表盘"""
    stats = monitor.get_stats()
    
    # 主布局
    layout = Layout()
    
    # 概览面板
    overview = Table.grid(padding=1)
    overview.add_column(style="cyan", justify="right")
    overview.add_column(style="green")
    overview.add_column(style="cyan", justify="right")
    overview.add_column(style="green")
    
    overview.add_row(
        "接口:", monitor.interface,
        "运行时间:", format_time(stats['elapsed'])
    )
    overview.add_row(
        "数据波特率:", f"{monitor.data_bitrate/1_000_000:.1f} Mbps",
        "仲裁波特率:", f"{monitor.nominal_bitrate/1_000_000:.1f} Mbps"
    )
    
    # 统计面板
    stats_table = Table(title="📊 流量统计", show_header=True, header_style="bold magenta")
    stats_table.add_column("指标", style="cyan")
    stats_table.add_column("接收 (RX)", justify="right", style="green")
    stats_table.add_column("发送 (TX)", justify="right", style="yellow")
    
    stats_table.add_row("帧数", f"{stats['rx_count']:,}", f"{stats['tx_count']:,}")
    stats_table.add_row("字节数", format_bytes(stats['rx_bytes']), format_bytes(stats['tx_bytes']))
    stats_table.add_row("错误", f"{stats['rx_errors']}", f"{stats['tx_errors']}")
    
    # 负载面板
    load_table = Table(title="⚡ 总线负载", show_header=True, header_style="bold magenta")
    load_table.add_column("指标", style="cyan")
    load_table.add_column("数值", justify="right", style="green")
    
    # 负载颜色
    current_load = stats['current_load']
    if current_load < 30:
        load_color = "green"
    elif current_load < 70:
        load_color = "yellow"
    else:
        load_color = "red"
    
    load_bar = "█" * int(current_load / 5) + "░" * (20 - int(current_load / 5))
    
    load_table.add_row("当前帧率", f"{stats['frame_rate']} fps")
    load_table.add_row("当前负载", f"[{load_color}]{current_load:.2f}%[/{load_color}]")
    load_table.add_row("负载条", f"[{load_color}]{load_bar}[/{load_color}]")
    load_table.add_row("平均负载", f"{stats['avg_load']:.4f}%")
    load_table.add_row("总比特数", f"{stats['total_bits']:,} bits")
    
    # ID统计面板 (按帧数排序，显示前15个)
    id_table = Table(title="🔍 CAN ID 统计 (Top 15)", show_header=True, header_style="bold magenta")
    id_table.add_column("CAN ID", style="cyan", justify="center")
    id_table.add_column("名称", style="white")
    id_table.add_column("帧数", justify="right", style="green")
    id_table.add_column("字节", justify="right", style="yellow")
    id_table.add_column("最后数据", style="dim")
    
    sorted_ids = sorted(stats['id_stats'].items(), key=lambda x: x[1]['count'], reverse=True)[:15]
    for can_id, id_data in sorted_ids:
        hex_data = id_data['last_data'].hex().upper()
        hex_formatted = ' '.join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
        id_table.add_row(
            f"0x{can_id:03X}",
            get_id_name(can_id),
            f"{id_data['count']:,}",
            format_bytes(id_data['bytes']),
            hex_formatted[:23] + "..." if len(hex_formatted) > 26 else hex_formatted
        )
    
    # 组合所有面板
    main_table = Table.grid(padding=1)
    main_table.add_column()
    main_table.add_column()
    
    main_table.add_row(
        Panel(overview, title="🖥️ CAN FD 总线监控", border_style="blue"),
    )
    
    sub_table = Table.grid(padding=1)
    sub_table.add_column()
    sub_table.add_column()
    sub_table.add_row(stats_table, load_table)
    
    main_table.add_row(sub_table)
    main_table.add_row(id_table)
    main_table.add_row(
        Text(f"按 Ctrl+C 退出 | 更新时间: {datetime.now().strftime('%H:%M:%S')}", style="dim")
    )
    
    return Panel(main_table, border_style="green")


def main():
    console = Console()
    
    console.print("[bold blue]🚀 CAN FD 总线负载监控器[/bold blue]")
    console.print("正在初始化...\n")
    
    monitor = CANMonitor(interface='can1', data_bitrate=5_000_000, nominal_bitrate=1_000_000)
    
    if not monitor.start():
        console.print("[red]❌ 无法启动监控，请检查 CAN 接口[/red]")
        return
    
    console.print("[green]✅ CAN 接口已打开[/green]\n")
    
    try:
        with Live(create_dashboard(monitor, console), refresh_per_second=4, console=console) as live:
            while True:
                time.sleep(0.25)
                live.update(create_dashboard(monitor, console))
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️ 正在停止监控...[/yellow]")
    finally:
        monitor.stop()
        console.print("[green]✅ 监控已停止[/green]")


if __name__ == "__main__":
    main()
