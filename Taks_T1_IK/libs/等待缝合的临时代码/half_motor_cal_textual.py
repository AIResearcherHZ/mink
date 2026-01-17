#!/usr/bin/env python3
"""
电机零位校准程序 - Textual TUI 版本 (CAN FD)
使用 Textual 库显示所有电机当前的弧度值
选中角度后弹窗确认是否清零校准
基于 DM_CAN_FD.py SocketCAN 接口
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drivers.DM_CAN_FD import Motor, MotorControlFD, DM_Motor_Type

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, DataTable, Button, Label
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text

# 配置参数
CAN_INTERFACE = "can1"  # SocketCAN 接口名
UI_REFRESH_RATE = 10  # UI刷新频率 Hz (降低以保证流畅)
CAN_POLL_INTERVAL = 0.05  # CAN轮询间隔 50ms (20Hz)

# 预设的电机配置 (半身 22个电机: 0x01 ~ 0x16)
PRESET_MOTOR_IDS = list(range(0x01, 0x17))  # 1~22

# 电机ID与关节名称的映射 (半身 22个电机)
MOTOR_NAMES = {
    # 右手 (Arm_R) 0x01~0x07
    0x01: "R_ShldrP",    # right_shoulder_pitch
    0x02: "R_ShldrR",    # right_shoulder_roll
    0x03: "R_ShldrY",    # right_shoulder_yaw
    0x04: "R_Elbow",     # right_elbow
    0x05: "R_WristR",    # right_wrist_roll
    0x06: "R_WristY",    # right_wrist_yaw
    0x07: "R_WristP",    # right_wrist_pitch
    # 右手夹爪 (Gripper_R) 0x08
    0x08: "R_Gripper",   # right_gripper
    
    # 左手 (Arm_L) 0x09~0x0F
    0x09: "L_ShldrP",    # left_shoulder_pitch
    0x0A: "L_ShldrR",    # left_shoulder_roll
    0x0B: "L_ShldrY",    # left_shoulder_yaw
    0x0C: "L_Elbow",     # left_elbow
    0x0D: "L_WristR",    # left_wrist_roll
    0x0E: "L_WristY",    # left_wrist_yaw
    0x0F: "L_WristP",    # left_wrist_pitch
    # 左手夹爪 (Gripper_L) 0x10
    0x10: "L_Gripper",   # left_gripper
    
    # 腰部 (Waist) 0x11~0x13
    0x11: "WaistY",      # waist_yaw
    0x12: "WaistR",      # waist_roll
    0x13: "WaistP",      # waist_pitch
    
    # 脖子 (Neck) 0x14~0x16
    0x14: "NeckY",       # neck_yaw
    0x15: "NeckR",       # neck_roll
    0x16: "NeckP",       # neck_pitch
}

def get_motor_name(motor_id: int) -> str:
    """获取电机名称"""
    return MOTOR_NAMES.get(motor_id, f"J{motor_id}")


# 分组配置: (group_id, 中文名, 英文名)
MOTOR_GROUPS = {
    "arm_r": ("右手", range(0x01, 0x09)),      # 0x01~0x08
    "arm_l": ("左手", range(0x09, 0x11)),      # 0x09~0x10
    "waist": ("腰部", range(0x11, 0x14)),      # 0x11~0x13
    "neck":  ("脖子", range(0x14, 0x17)),      # 0x14~0x16
}

def get_motor_group_name(motor_id: int) -> str:
    """根据电机ID获取分组名称(中文)"""
    if 0x01 <= motor_id <= 0x08:
        return "右手"
    elif 0x09 <= motor_id <= 0x10:
        return "左手"
    elif 0x11 <= motor_id <= 0x13:
        return "腰部"
    elif 0x14 <= motor_id <= 0x16:
        return "脖子"
    return "未知"

def get_motor_group_id(motor_id: int) -> str:
    """根据电机ID获取分组ID(英文, 用于Textual ID)"""
    if 0x01 <= motor_id <= 0x08:
        return "arm_r"
    elif 0x09 <= motor_id <= 0x10:
        return "arm_l"
    elif 0x11 <= motor_id <= 0x13:
        return "waist"
    elif 0x14 <= motor_id <= 0x16:
        return "neck"
    return "unknown"


class MotorDataCache:
    """线程安全的电机数据缓存 - 纯数据层，与UI完全解耦"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}  # {motor_id: (pos, vel, tau, online)}
        self._version = 0  # 数据版本号
    
    def update(self, motor_id: int, pos: float, vel: float, tau: float, online: bool = True):
        """更新电机数据 (由CAN线程调用)"""
        with self._lock:
            self._data[motor_id] = (pos, vel, tau, online)
            self._version += 1
    
    def get(self, motor_id: int):
        """获取单个电机数据"""
        with self._lock:
            return self._data.get(motor_id, (0.0, 0.0, 0.0, False))
    
    def get_all_if_changed(self, last_version: int):
        """获取所有电机数据的快照 (仅当版本变化时返回数据)"""
        with self._lock:
            if self._version == last_version:
                return None, last_version  # 数据未变化
            return dict(self._data), self._version


class CANCommThread(threading.Thread):
    """独立的CAN通讯线程 - 完全与UI解耦"""
    
    def __init__(self, motor_ids: list, data_cache: MotorDataCache, interface: str = "can1"):
        super().__init__(daemon=True)
        self.motor_ids = motor_ids
        self.data_cache = data_cache
        self.interface = interface
        self.running = False
        self.motor_control = None
        self.motors = {}
        self._command_lock = threading.Lock()
        self._pending_commands = []  # [(command_type, motor_id, args), ...]
    
    def run(self):
        """CAN通讯主循环"""
        self.running = True
        
        # 初始化CAN接口
        try:
            self.motor_control = MotorControlFD(can_interface=self.interface)
        except Exception as e:
            print(f"CAN接口初始化失败: {e}")
            self.running = False
            return
        
        # 添加所有电机
        for slave_id in self.motor_ids:
            master_id = slave_id + 0x80
            motor = Motor(DM_Motor_Type.DM4340, slave_id, master_id)
            self.motors[slave_id] = motor
            self.motor_control.addMotor(motor)
        
        # 启动时先 enable 所有电机并写入 kp/kd=0
        self._init_all_motors()
        
        # 主循环
        while self.running:
            # 处理待执行的命令
            self._process_commands()
            
            # 刷新所有电机状态
            self._refresh_all_motors()
            
            time.sleep(CAN_POLL_INTERVAL)
        
        # 退出时失能所有电机
        self._disable_all_motors()
        
        if self.motor_control:
            self.motor_control.close()
    
    def _init_all_motors(self):
        """启动时初始化所有电机: enable + controlMIT(kp=0, kd=0)"""
        if not self.motor_control:
            return
        
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.enable(motor)
                self.motor_control.controlMIT(motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
        
        # 等待响应
        time.sleep(0.05)
        self.motor_control.recv()
    
    def _process_commands(self):
        """处理待执行的命令"""
        with self._command_lock:
            commands = self._pending_commands[:]
            self._pending_commands.clear()
        
        for cmd_type, motor_id, args in commands:
            if motor_id not in self.motors:
                continue
            motor = self.motors[motor_id]
            
            try:
                if cmd_type == "enable":
                    self.motor_control.enable(motor)
                    self.motor_control.controlMIT(motor, 0.0, 0.0, 0.0, 0.0, 0.0)
                elif cmd_type == "disable":
                    self.motor_control.disable(motor)
                elif cmd_type == "set_zero":
                    self.motor_control.set_zero_position(motor)
            except Exception:
                pass
    
    def _refresh_all_motors(self):
        """刷新所有电机状态"""
        if not self.motor_control:
            return
        
        # 发送所有电机的状态请求
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.refresh_motor_status(motor)
            except Exception:
                pass
        
        # 接收响应
        time.sleep(0.01)
        self.motor_control.recv()
        
        # 更新缓存
        for motor_id, motor in self.motors.items():
            pos = motor.getPosition()
            vel = motor.getVelocity()
            tau = motor.getTorque()
            self.data_cache.update(motor_id, pos, vel, tau, True)
    
    def _disable_all_motors(self):
        """失能所有电机"""
        if not self.motor_control:
            return
        
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.disable(motor)
            except Exception:
                pass
    
    def send_command(self, cmd_type: str, motor_id: int, args=None):
        """发送命令到CAN线程 (线程安全)"""
        with self._command_lock:
            self._pending_commands.append((cmd_type, motor_id, args))
    
    def stop(self):
        """停止CAN通讯线程"""
        self.running = False


class ConfirmZeroScreen(ModalScreen[bool]):
    """确认清零校准的弹窗"""
    
    BINDINGS = [
        Binding("y", "confirm", "确认"),
        Binding("n", "cancel", "取消"),
        Binding("escape", "cancel", "取消"),
    ]
    
    CSS = """
    ConfirmZeroScreen {
        align: center middle;
    }
    
    #dialog {
        width: 50;
        height: 12;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    
    #dialog-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    
    #dialog-content {
        text-align: center;
        margin-bottom: 1;
    }
    
    #dialog-buttons {
        align: center middle;
        height: 3;
    }
    
    #dialog-buttons Button {
        margin: 0 2;
    }
    """
    
    def __init__(self, motor_id: int, current_deg: float):
        super().__init__()
        self.motor_id = motor_id
        self.current_deg = current_deg
    
    def compose(self) -> ComposeResult:
        motor_name = get_motor_name(self.motor_id)
        group_name = get_motor_group_name(self.motor_id)
        with Container(id="dialog"):
            yield Label("⚠️ 确认清零校准", id="dialog-title")
            yield Label(
                f"电机 {motor_name} (0x{self.motor_id:02X}) [{group_name}]\n"
                f"当前角度: {self.current_deg:+.1f}°\n"
                f"确定要将此位置设为零位吗?",
                id="dialog-content"
            )
            with Horizontal(id="dialog-buttons"):
                yield Button("确认 (Y)", variant="warning", id="confirm")
                yield Button("取消 (N)", variant="default", id="cancel")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        else:
            self.dismiss(False)
    
    def action_confirm(self) -> None:
        self.dismiss(True)
    
    def action_cancel(self) -> None:
        self.dismiss(False)


class MotorDoubleClicked(Message):
    """电机双击消息"""
    def __init__(self, motor_id: int, pos_deg: float):
        super().__init__()
        self.motor_id = motor_id
        self.pos_deg = pos_deg


class MotorTable(Static):
    """电机表格组件 (CAN FD 版本)"""
    
    def __init__(self, group_name: str, motor_ids: list, **kwargs):
        super().__init__(**kwargs)
        self.group_name = group_name
        self.motor_ids = motor_ids
        self.motor_data = {}  # {motor_id: (pos_rad, pos_deg, vel, tau)}
        self._last_rendered = {}  # 缓存上次渲染的值，避免重复渲染
        self._table_cache = None  # 缓存 DataTable 引用，避免重复 query_one
        self._motor_id_to_row = {mid: idx for idx, mid in enumerate(motor_ids)}  # 缓存 motor_id -> row_idx 映射
        
    def compose(self) -> ComposeResult:
        table = DataTable(id=f"table_{self.group_name}", cursor_type="row")
        yield table
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """双击行时触发清零确认"""
        if event.row_key is not None:
            motor_id = int(event.row_key.value)
            pos_deg = 0.0
            if motor_id in self.motor_data:
                _, pos_deg, _, _ = self.motor_data[motor_id]
            self.post_message(MotorDoubleClicked(motor_id, pos_deg))
    
    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        self._table_cache = table  # 缓存引用
        table.add_columns("名称", "ID", "10进制", "rad", "°", "vel", "τ")
        
        # 添加初始行
        for motor_id in self.motor_ids:
            motor_name = get_motor_name(motor_id)
            table.add_row(
                motor_name,
                f"0x{motor_id:02X}",
                f"{motor_id}",
                "+0.000",
                "+0.0",
                "+0.00",
                "+0.00",
                key=str(motor_id)
            )
    
    def get_degree_color(self, motor_id: int, pos_deg: float) -> str:
        """根据角度值返回渐变颜色
        0~5°: 灰色 (接近零位)
        5~15°: 浅色
        15~30°: 中等色
        30~60°: 较深色
        60~90°: 深色
        |deg|>90°: 反色警告
        正值用绿色系，负值用红色系
        """
        if round(pos_deg, 1) == 0.0:
            return "dim white"
        # 计算相对于参考值的偏差
        if motor_id in (0x0C, 0x04):
            ref_deg = -90.0
        else:
            ref_deg = 0.0
        
        deg = pos_deg - ref_deg
        abs_deg = abs(deg)
        is_positive = deg >= 0
        
        # 渐变颜色映射
        if abs_deg < 5:
            return "dim white"
        elif abs_deg < 15:
            return "green4" if is_positive else "indian_red"
        elif abs_deg < 30:
            return "green3" if is_positive else "red3"
        elif abs_deg < 60:
            return "green1" if is_positive else "red1"
        elif abs_deg <= 90:
            return "bold bright_green" if is_positive else "bold bright_red"
        else:
            # 绝对值超过90度: 反色警告 (包括>90和<-90)
            return "bold reverse bright_green" if is_positive else "bold reverse bright_red"
    
    def render_motor(self, motor_id: int, pos_rad: float, vel: float, tau: float):
        """渲染电机数据 (固定帧率调用，只在数据变化时更新UI)"""
        pos_deg = pos_rad * 180.0 / 3.14159265359
        pos_deg_display = round(pos_deg, 1)
        self.motor_data[motor_id] = (pos_rad, pos_deg_display, vel, tau)
        
        # 格式化字符串
        rad_str = f"{pos_rad:+.3f}"
        deg_str = f"{pos_deg_display:+.1f}"
        vel_str = f"{vel:+.2f}"
        tau_str = f"{tau:+.2f}"
        
        # 检查是否需要更新UI (数据未变化则跳过)
        current_render = (rad_str, deg_str, vel_str, tau_str)
        if motor_id in self._last_rendered and self._last_rendered[motor_id] == current_render:
            return  # 数据未变化，跳过UI更新
        self._last_rendered[motor_id] = current_render
        
        # 找到该电机在列表中的索引 (使用缓存的映射)
        row_idx = self._motor_id_to_row.get(motor_id)
        if row_idx is None:
            return
        
        # 获取颜色
        color = self.get_degree_color(motor_id, pos_deg_display)
        
        # 更新单元格 (使用缓存的 table 引用)
        try:
            table = self._table_cache
            if table is None:
                return
            table.update_cell_at((row_idx, 3), Text(rad_str, style=color))
            table.update_cell_at((row_idx, 4), Text(deg_str, style=color))
            table.update_cell_at((row_idx, 5), vel_str)
            table.update_cell_at((row_idx, 6), tau_str)
        except Exception:
            pass
    
    def get_selected_motor(self) -> tuple:
        """获取当前选中的电机ID和角度"""
        table = self._table_cache if self._table_cache else self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.motor_ids):
            motor_id = self.motor_ids[table.cursor_row]
            if motor_id in self.motor_data:
                _, pos_deg, _, _ = self.motor_data[motor_id]
                return motor_id, pos_deg
            return motor_id, 0.0
        return None, None


class MotorCalibrationApp(App):
    """电机校准 Textual 应用 (CAN FD 版本)"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #main-container {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    
    #tables-container {
        width: 100%;
        height: 1fr;
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
    }
    
    .group-container {
        border: solid $primary;
        padding: 0 1;
        height: 100%;
    }
    
    .group-title {
        text-align: center;
        text-style: bold;
        color: $secondary;
        background: $primary-darken-3;
    }
    
    DataTable {
        height: 1fr;
    }
    
    #status-bar {
        dock: bottom;
        height: 3;
        background: $primary-darken-2;
        padding: 0 1;
    }
    
    #status-text {
        text-align: center;
        color: $text;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "退出", priority=True),
        Binding("q", "quit", "退出"),
        Binding("r", "refresh", "刷新"),
        Binding("tab", "next_table", "下一表格"),
        Binding("shift+tab", "prev_table", "上一表格"),
    ]
    
    status_message = reactive("就绪")
    
    def __init__(self):
        super().__init__()
        self.motor_ids = PRESET_MOTOR_IDS  # 直接使用预设电机列表
        self.motor_tables = {}  # {group_name: MotorTable}
        self.current_table_index = 0
        self.data_cache = MotorDataCache()  # 线程安全的数据缓存
        self.can_thread = None  # CAN通讯线程
        self._last_data_version = 0  # 上次渲染时的数据版本
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Grid(id="tables-container"):
                pass  # 动态添加表格
            with Horizontal(id="status-bar"):
                yield Label(self.status_message, id="status-text")
        yield Footer()
    
    async def on_mount(self) -> None:
        """应用启动 - 直接创建表格并启动CAN通讯"""
        await self.create_tables()
        self.start_can_thread()
        # 启动UI刷新定时器 (每秒10次)
        self.set_interval(1.0 / UI_REFRESH_RATE, self._refresh_ui)
        self.status_message = f"✅ 已连接 {CAN_INTERFACE}，监控 {len(self.motor_ids)} 个电机"
    
    def start_can_thread(self):
        """启动独立的CAN通讯线程"""
        self.can_thread = CANCommThread(self.motor_ids, self.data_cache, CAN_INTERFACE)
        self.can_thread.start()
    
    async def create_tables(self):
        """创建电机表格 - 按半身布局排序"""
        container = self.query_one("#tables-container")
        
        # 按分组整理电机 (使用英文ID作为key)
        groups = {
            "arm_r": [],   # 右手
            "arm_l": [],   # 左手
            "waist": [],   # 腰部
            "neck": [],    # 脖子
        }
        
        for motor_id in self.motor_ids:
            group_id = get_motor_group_id(motor_id)
            if group_id in groups:
                groups[group_id].append(motor_id)
        
        # 定义显示顺序: (英文ID, 中文显示名)
        display_order = [
            ("arm_l", "左手"),
            ("waist", "腰部"),
            ("arm_r", "右手"),
            ("neck", "脖子"),
        ]
        
        for group_id, display_name in display_order:
            motor_ids = groups.get(group_id, [])
            if not motor_ids:
                continue
            
            group_container = Vertical(classes="group-container", id=f"group_{group_id}")
            await container.mount(group_container)
            
            title_label = Label(display_name, classes="group-title")
            await group_container.mount(title_label)
            
            motor_table = MotorTable(group_id, motor_ids)
            self.motor_tables[group_id] = motor_table
            await group_container.mount(motor_table)
    
    def _refresh_ui(self):
        """固定20fps渲染 - 仅当数据版本变化时才处理"""
        # 检查数据是否有变化
        all_data, new_version = self.data_cache.get_all_if_changed(self._last_data_version)
        
        if all_data is None:
            return  # 数据未变化，跳过本帧
        
        self._last_data_version = new_version
        
        # 遍历所有电机，用最新数据渲染
        for motor_id, (pos, vel, tau, online) in all_data.items():
            group_id = get_motor_group_id(motor_id)
            if group_id in self.motor_tables:
                self.motor_tables[group_id].render_motor(motor_id, pos, vel, tau)
    
    def set_motor_zero(self, motor_id: int) -> bool:
        """设置指定电机的零位 (通过CAN线程发送命令)"""
        if not self.can_thread or not self.can_thread.running:
            self.notify(f"❌ CAN通讯未连接", severity="error")
            return False
        
        self.can_thread.send_command("set_zero", motor_id)
        self.notify(f"✅ 电机 0x{motor_id:02X} 零位设置命令已发送!", severity="information")
        return True
    
    def on_motor_double_clicked(self, event: MotorDoubleClicked) -> None:
        """处理电机双击事件"""
        motor_id = event.motor_id
        pos_deg = event.pos_deg
        
        def handle_confirm(result: bool) -> None:
            if result:
                self.set_motor_zero(motor_id)
        
        self.push_screen(
            ConfirmZeroScreen(motor_id, pos_deg),
            handle_confirm
        )
    
    def action_next_table(self) -> None:
        """切换到下一个表格"""
        if not self.motor_tables:
            return
        
        tables = list(self.motor_tables.values())
        self.current_table_index = (self.current_table_index + 1) % len(tables)
        tables[self.current_table_index].query_one(DataTable).focus()
    
    def action_prev_table(self) -> None:
        """切换到上一个表格"""
        if not self.motor_tables:
            return
        
        tables = list(self.motor_tables.values())
        self.current_table_index = (self.current_table_index - 1) % len(tables)
        tables[self.current_table_index].query_one(DataTable).focus()
    
    def action_refresh(self) -> None:
        """手动刷新"""
        self.notify("已刷新", severity="information")
    
    def on_unmount(self) -> None:
        """关闭时停止CAN线程 (线程会自动失能所有电机)"""
        if self.can_thread:
            self.can_thread.stop()
            self.can_thread.join(timeout=2.0)  # 等待线程结束
    
    def watch_status_message(self, message: str) -> None:
        """更新状态栏"""
        try:
            label = self.query_one("#status-text", Label)
            label.update(message)
        except Exception:
            pass


def main():
    app = MotorCalibrationApp()
    app.title = "电机零位校准程序"
    app.sub_title = "CAN FD 版本"
    app.run()


if __name__ == "__main__":
    main()