#!/usr/bin/env python3
"""
电机零位校准程序 - Textual TUI 版本 (CAN FD) - 全身版本
使用 Textual 库显示所有电机当前的弧度值
选中角度后弹窗确认是否清零校准
基于 DM_CAN_FD.py SocketCAN 接口
支持双CAN总线: can0(上半身) + can1(双腿)

sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on

"""
import sys
import os
import time
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from libs.drivers.DM_CAN_FD import Motor, MotorControlFD, DM_Motor_Type

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Static, DataTable, Button, Label
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from rich.text import Text
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
import math

CAN0_INTERFACE = "can0"
CAN1_INTERFACE = "can1"
UI_REFRESH_RATE = 30
CAN_POLL_INTERVAL = 0.03

PRESET_MOTOR_IDS_CAN0 = list(range(0x01, 0x17))
PRESET_MOTOR_IDS_CAN1 = list(range(0x17, 0x23))

MOTOR_NAMES = {
    0x01: "R_ShldrP",
    0x02: "R_ShldrR",
    0x03: "R_ShldrY",
    0x04: "R_Elbow",
    0x05: "R_WristR",
    0x06: "R_WristY",
    0x07: "R_WristP",
    0x08: "R_Gripper",
    
    0x09: "L_ShldrP",
    0x0A: "L_ShldrR",
    0x0B: "L_ShldrY",
    0x0C: "L_Elbow",
    0x0D: "L_WristR",
    0x0E: "L_WristY",
    0x0F: "L_WristP",
    0x10: "L_Gripper",
    
    0x11: "WaistY",
    0x12: "WaistR",
    0x13: "WaistP",
    
    0x14: "NeckY",
    0x15: "NeckR",
    0x16: "NeckP",
    
    0x17: "L_HipP",
    0x18: "L_HipR",
    0x19: "L_HipY",
    0x1A: "L_Knee",
    0x1B: "L_AnkleP",
    0x1C: "L_AnkleR",
    
    0x1D: "R_HipP",
    0x1E: "R_HipR",
    0x1F: "R_HipY",
    0x20: "R_Knee",
    0x21: "R_AnkleP",
    0x22: "R_AnkleR",
}

def get_motor_name(motor_id: int) -> str:
    return MOTOR_NAMES.get(motor_id, f"J{motor_id}")


MOTOR_GROUPS = {
    "arm_r": ("右手", range(0x01, 0x09)),
    "arm_l": ("左手", range(0x09, 0x11)),
    "waist": ("腰部", range(0x11, 0x14)),
    "neck":  ("脖子", range(0x14, 0x17)),
    "leg_l": ("左腿", range(0x17, 0x1D)),
    "leg_r": ("右腿", range(0x1D, 0x23)),
}

def get_motor_group_name(motor_id: int) -> str:
    if 0x01 <= motor_id <= 0x08:
        return "右手"
    elif 0x09 <= motor_id <= 0x10:
        return "左手"
    elif 0x11 <= motor_id <= 0x13:
        return "腰部"
    elif 0x14 <= motor_id <= 0x16:
        return "脖子"
    elif 0x17 <= motor_id <= 0x1C:
        return "左腿"
    elif 0x1D <= motor_id <= 0x22:
        return "右腿"
    return "未知"

def get_motor_group_id(motor_id: int) -> str:
    if 0x01 <= motor_id <= 0x08:
        return "arm_r"
    elif 0x09 <= motor_id <= 0x10:
        return "arm_l"
    elif 0x11 <= motor_id <= 0x13:
        return "waist"
    elif 0x14 <= motor_id <= 0x16:
        return "neck"
    elif 0x17 <= motor_id <= 0x1C:
        return "leg_l"
    elif 0x1D <= motor_id <= 0x22:
        return "leg_r"
    return "unknown"

def get_motor_can_interface(motor_id: int) -> str:
    if 0x01 <= motor_id <= 0x16:
        return CAN0_INTERFACE
    elif 0x17 <= motor_id <= 0x22:
        return CAN1_INTERFACE
    return CAN0_INTERFACE


def create_angle_indicator(angle_deg: float) -> str:
    angle = angle_deg % 360
    if angle < 0:
        angle += 360
    
    angle_rad = math.radians(90 - angle)
    
    width_chars = 5
    width_dots = width_chars * 2
    height_dots = 4
    
    cx, cy = 5.0, 2.0
    radius = 3.5
    
    px = cx + radius * math.cos(angle_rad)
    py = cy - radius * math.sin(angle_rad)
    
    canvas = [[False] * width_dots for _ in range(height_dots)]
    
    for x in range(width_dots):
        for y in range(height_dots):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if abs(dist - radius) < 0.8:
                canvas[y][x] = True
    
    steps = 30
    for i in range(steps):
        t = i / steps
        lx = int(cx + t * (px - cx) + 0.5)
        ly = int(cy + t * (py - cy) + 0.5)
        if 0 <= lx < width_dots and 0 <= ly < height_dots:
            canvas[ly][lx] = True
    
    result = ""
    for char_x in range(width_chars):
        dot_x = char_x * 2
        
        dots = [
            canvas[0][dot_x] if dot_x < width_dots else False,
            canvas[1][dot_x] if dot_x < width_dots else False,
            canvas[2][dot_x] if dot_x < width_dots else False,
            canvas[0][dot_x+1] if dot_x+1 < width_dots else False,
            canvas[1][dot_x+1] if dot_x+1 < width_dots else False,
            canvas[2][dot_x+1] if dot_x+1 < width_dots else False,
            canvas[3][dot_x] if dot_x < width_dots else False,
            canvas[3][dot_x+1] if dot_x+1 < width_dots else False,
        ]
        
        braille_value = 0
        if dots[0]: braille_value |= 0x01
        if dots[1]: braille_value |= 0x02
        if dots[2]: braille_value |= 0x04
        if dots[3]: braille_value |= 0x08
        if dots[4]: braille_value |= 0x10
        if dots[5]: braille_value |= 0x20
        if dots[6]: braille_value |= 0x40
        if dots[7]: braille_value |= 0x80
        
        result += chr(0x2800 + braille_value)
    
    return result


class MotorDataCache:
    
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}
        self._version = 0
    
    def update(self, motor_id: int, pos: float, vel: float, tau: float, online: bool = True):
        with self._lock:
            self._data[motor_id] = (pos, vel, tau, online)
            self._version += 1
    
    def get(self, motor_id: int):
        with self._lock:
            return self._data.get(motor_id, (0.0, 0.0, 0.0, False))
    
    def get_all_if_changed(self, last_version: int):
        with self._lock:
            if self._version == last_version:
                return None, last_version
            return dict(self._data), self._version


class CANCommThread(threading.Thread):
    
    def __init__(self, motor_ids: list, data_cache: MotorDataCache, interface: str = "can0"):
        super().__init__(daemon=True)
        self.motor_ids = motor_ids
        self.data_cache = data_cache
        self.interface = interface
        self.running = False
        self.motor_control = None
        self.motors = {}
        self._command_lock = threading.Lock()
        self._pending_commands = []
    
    def run(self):
        self.running = True
        
        try:
            self.motor_control = MotorControlFD(can_interface=self.interface)
        except Exception as e:
            print(f"CAN接口 {self.interface} 初始化失败: {e}")
            self.running = False
            return
        
        for slave_id in self.motor_ids:
            master_id = slave_id + 0x80
            motor = Motor(DM_Motor_Type.DM4340, slave_id, master_id)
            self.motors[slave_id] = motor
            self.motor_control.addMotor(motor)
        
        self._init_all_motors()
        
        while self.running:
            self._process_commands()
            self._refresh_all_motors()
            time.sleep(CAN_POLL_INTERVAL)
        
        self._disable_all_motors()
        
        if self.motor_control:
            self.motor_control.close()
    
    def _init_all_motors(self):
        if not self.motor_control:
            return
        
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.enable(motor)
                self.motor_control.controlMIT(motor, 0.0, 0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass
        
        time.sleep(0.05)
        self.motor_control.recv()
    
    def _process_commands(self):
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
        if not self.motor_control:
            return
        
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.refresh_motor_status(motor)
            except Exception:
                pass
        
        time.sleep(0.01)
        self.motor_control.recv()
        
        for motor_id, motor in self.motors.items():
            pos = motor.getPosition()
            vel = motor.getVelocity()
            tau = motor.getTorque()
            self.data_cache.update(motor_id, pos, vel, tau, True)
    
    def _disable_all_motors(self):
        if not self.motor_control:
            return
        
        for motor_id, motor in self.motors.items():
            try:
                self.motor_control.disable(motor)
            except Exception:
                pass
    
    def send_command(self, cmd_type: str, motor_id: int, args=None):
        with self._command_lock:
            self._pending_commands.append((cmd_type, motor_id, args))
    
    def stop(self):
        self.running = False


class ConfirmZeroScreen(ModalScreen[bool]):
    
    BINDINGS = [
        Binding("y", "confirm", "确认"),
        Binding("n", "cancel", "取消"),
        Binding("escape", "cancel", "取消"),
    ]
    
    CSS = """
    ConfirmZeroScreen {
        align: center middle;
        background: $boost;
        tint: black 70%;
    }
    
    #dialog {
        width: 50;
        height: 12;
        border: thick $warning;
        background: $panel;
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
        color: $text;
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
        can_interface = get_motor_can_interface(self.motor_id)
        with Container(id="dialog"):
            yield Label("⚠️ 确认清零校准", id="dialog-title")
            yield Label(
                f"电机 {motor_name} (0x{self.motor_id:02X}) [{group_name}] [{can_interface}]\n"
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
    def __init__(self, motor_id: int, pos_deg: float):
        super().__init__()
        self.motor_id = motor_id
        self.pos_deg = pos_deg


class MotorTable(Static):
    
    def __init__(self, group_name: str, motor_ids: list, **kwargs):
        super().__init__(**kwargs)
        self.group_name = group_name
        self.motor_ids = motor_ids
        self.motor_data = {}
        self._last_rendered = {}
        self._table_cache = None
        self._motor_id_to_row = {mid: idx for idx, mid in enumerate(motor_ids)}
        
    def compose(self) -> ComposeResult:
        table = DataTable(id=f"table_{self.group_name}", cursor_type="row")
        yield table
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is not None:
            motor_id = int(event.row_key.value)
            pos_deg = 0.0
            if motor_id in self.motor_data:
                _, pos_deg, _, _ = self.motor_data[motor_id]
            self.post_message(MotorDoubleClicked(motor_id, pos_deg))
    
    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        self._table_cache = table
        table.add_columns("名称", "ID", "rad", "°", "⊙", "vel", "τ")
        
        for motor_id in self.motor_ids:
            motor_name = get_motor_name(motor_id)
            table.add_row(
                motor_name,
                f"0x{motor_id:02X}",
                "+0.0",
                "+0.0",
                create_angle_indicator(0.0),
                "+0.0",
                "+0.0",
                key=str(motor_id)
            )
    
    def get_degree_color(self, motor_id: int, pos_deg: float) -> str:
        abs_deg = abs(pos_deg)
        
        if abs_deg < 5:
            return "dim white"
        elif abs_deg < 15:
            return "green4"
        elif abs_deg < 30:
            return "green3"
        elif abs_deg < 45:
            return "green1"
        elif abs_deg < 90:
            return "indian_red"
        elif abs_deg < 135:
            return "red3"
        elif abs_deg <= 180:
            return "red1"
        else:
            return "bold reverse bright_red"
    
    def render_motor(self, motor_id: int, pos_rad: float, vel: float, tau: float):
        pos_deg = pos_rad * 180.0 / 3.14159265359
        pos_deg_display = round(pos_deg, 1)
        self.motor_data[motor_id] = (pos_rad, pos_deg_display, vel, tau)
        
        rad_str = f"{pos_rad:+.1f}"
        deg_str = f"{pos_deg_display:+.1f}"
        vel_str = f"{vel:+.1f}"
        tau_str = f"{tau:+.1f}"
        angle_indicator = create_angle_indicator(pos_deg_display)
        
        current_render = (rad_str, deg_str, angle_indicator, vel_str, tau_str)
        if motor_id in self._last_rendered and self._last_rendered[motor_id] == current_render:
            return
        self._last_rendered[motor_id] = current_render
        
        row_idx = self._motor_id_to_row.get(motor_id)
        if row_idx is None:
            return
        
        color = self.get_degree_color(motor_id, pos_deg_display)
        
        try:
            table = self._table_cache
            if table is None:
                return
            table.update_cell_at((row_idx, 2), Text(rad_str, style=color))
            table.update_cell_at((row_idx, 3), Text(deg_str, style=color))
            table.update_cell_at((row_idx, 4), Text(angle_indicator, style=color))
            table.update_cell_at((row_idx, 5), vel_str)
            table.update_cell_at((row_idx, 6), tau_str)
        except Exception:
            pass
    
    def get_selected_motor(self) -> tuple:
        table = self._table_cache if self._table_cache else self.query_one(DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.motor_ids):
            motor_id = self.motor_ids[table.cursor_row]
            if motor_id in self.motor_data:
                _, pos_deg, _, _ = self.motor_data[motor_id]
                return motor_id, pos_deg
            return motor_id, 0.0
        return None, None


class MotorCalibrationApp(App):
    
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
        grid-size: 3 2;
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
        self.motor_ids_can0 = PRESET_MOTOR_IDS_CAN0
        self.motor_ids_can1 = PRESET_MOTOR_IDS_CAN1
        self.motor_tables = {}
        self.current_table_index = 0
        self.data_cache = MotorDataCache()
        self.can0_thread = None
        self.can1_thread = None
        self._last_data_version = 0
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Grid(id="tables-container"):
                pass
            with Horizontal(id="status-bar"):
                yield Label(self.status_message, id="status-text")
        yield Footer()
    
    async def on_mount(self) -> None:
        await self.create_tables()
        self.start_can_threads()
        self.set_interval(1.0 / UI_REFRESH_RATE, self._refresh_ui)
        total_motors = len(self.motor_ids_can0) + len(self.motor_ids_can1)
        self.status_message = f"✅ 已连接 {CAN0_INTERFACE}({len(self.motor_ids_can0)}个) + {CAN1_INTERFACE}({len(self.motor_ids_can1)}个)，共 {total_motors} 个电机"
    
    def start_can_threads(self):
        self.can0_thread = CANCommThread(self.motor_ids_can0, self.data_cache, CAN0_INTERFACE)
        self.can0_thread.start()
        
        self.can1_thread = CANCommThread(self.motor_ids_can1, self.data_cache, CAN1_INTERFACE)
        self.can1_thread.start()
    
    async def create_tables(self):
        container = self.query_one("#tables-container")
        
        groups = {
            "arm_r": [],
            "arm_l": [],
            "waist": [],
            "neck": [],
            "leg_l": [],
            "leg_r": [],
        }
        
        all_motor_ids = self.motor_ids_can0 + self.motor_ids_can1
        for motor_id in all_motor_ids:
            group_id = get_motor_group_id(motor_id)
            if group_id in groups:
                groups[group_id].append(motor_id)
        
        display_order = [
            ("arm_l", "左手"),
            ("waist", "腰部"),
            ("arm_r", "右手"),
            ("leg_l", "左腿"),
            ("neck", "脖子"),
            ("leg_r", "右腿"),
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
        all_data, new_version = self.data_cache.get_all_if_changed(self._last_data_version)
        
        if all_data is None:
            return
        
        self._last_data_version = new_version
        
        for motor_id, (pos, vel, tau, online) in all_data.items():
            group_id = get_motor_group_id(motor_id)
            if group_id in self.motor_tables:
                self.motor_tables[group_id].render_motor(motor_id, pos, vel, tau)
    
    def set_motor_zero(self, motor_id: int) -> bool:
        can_interface = get_motor_can_interface(motor_id)
        
        if can_interface == CAN0_INTERFACE:
            can_thread = self.can0_thread
        elif can_interface == CAN1_INTERFACE:
            can_thread = self.can1_thread
        else:
            self.notify(f"❌ 未知CAN接口", severity="error")
            return False
        
        if not can_thread or not can_thread.running:
            self.notify(f"❌ {can_interface} 通讯未连接", severity="error")
            return False
        
        can_thread.send_command("set_zero", motor_id)
        self.notify(f"✅ 电机 0x{motor_id:02X} ({can_interface}) 零位设置命令已发送!", severity="information")
        return True
    
    def on_motor_double_clicked(self, event: MotorDoubleClicked) -> None:
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
        if not self.motor_tables:
            return
        
        tables = list(self.motor_tables.values())
        self.current_table_index = (self.current_table_index + 1) % len(tables)
        tables[self.current_table_index].query_one(DataTable).focus()
    
    def action_prev_table(self) -> None:
        if not self.motor_tables:
            return
        
        tables = list(self.motor_tables.values())
        self.current_table_index = (self.current_table_index - 1) % len(tables)
        tables[self.current_table_index].query_one(DataTable).focus()
    
    def action_refresh(self) -> None:
        self.notify("已刷新", severity="information")
    
    def on_unmount(self) -> None:
        if self.can0_thread:
            self.can0_thread.stop()
            self.can0_thread.join(timeout=2.0)
        
        if self.can1_thread:
            self.can1_thread.stop()
            self.can1_thread.join(timeout=2.0)
    
    def watch_status_message(self, message: str) -> None:
        try:
            label = self.query_one("#status-text", Label)
            label.update(message)
        except Exception:
            pass


def main():
    app = MotorCalibrationApp()
    app.title = "电机零位校准程序 - 全身版本"
    app.sub_title = "CAN FD 双总线版本 (can0 + can1)"
    app.run()


if __name__ == "__main__":
    main()
