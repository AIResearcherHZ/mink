# CAN 接口启动说明

## 硬件配置

| 接口 | 硬件 | 驱动 | 22电机稳定频率 |
|------|------|------|---------------|
| can0 | 板载 TCAN4x5x (SPI) | m_can/tcan4x5x | ~40 Hz |
| can1 | USB cando_can | slcan | **600 Hz** |

---

## can0 - 板载 SPI-CAN (TCAN4x5x)

### 启动命令

```bash
# 关闭接口（如果已启动）
sudo ip link set can0 down

# 启动 CAN FD 模式：仲裁段 1Mbps，数据段 5Mbps，采样点 0.75
sudo ip link set can0 up type can bitrate 1000000 sample-point 0.75 dbitrate 5000000 fd on

# 验证
ip -details link show can0
```

### 仅启动经典 CAN 模式

```bash
sudo ip link set can0 up type can bitrate 1000000
```

### 关闭接口

```bash
sudo ip link set can0 down
```

---

## can1 - USB CAN 模块 (cando_can + slcan)

### 启动命令

```bash
# 1. 确认 USB 设备已连接
lsusb | grep -i can
# 应显示: GDMicroelectronics cando_can 或类似

# 2. 确认串口设备
ls /dev/ttyACM*
# 应显示: /dev/ttyACM0

# 3. 启动 slcan 守护进程 (4Mbps 串口速率)
sudo slcand -o -c -s8 -S 4000000 /dev/ttyACM0 can1

# 4. 启动接口
sudo ip link set can1 up

# 5. 验证
ip link show can1
```

### 一键启动脚本

```bash
sudo pkill slcand 2>/dev/null
sudo ip link delete can1 2>/dev/null
sudo slcand -o -c -s8 -S 4000000 /dev/ttyACM0 can1 && sudo ip link set can1 up
```

### 关闭接口

```bash
sudo ip link set can1 down
sudo ip link delete can1
sudo pkill slcand
```

### slcand 参数说明

| 参数 | 说明 |
|------|------|
| `-o` | 发送打开命令 |
| `-c` | 发送关闭命令 |
| `-s8` | CAN 波特率 1Mbps (s0=10k, s1=20k, s2=50k, s3=100k, s4=125k, s5=250k, s6=500k, s7=800k, s8=1M) |
| `-S 4000000` | 串口波特率 4Mbps |

---

## 性能测试结果

测试日期: 2026-01-10

### can0 (SPI-TCAN4x5x)
- 瓶颈: SPI 驱动使用同步模式 (spi_sync)
- 22电机稳定频率: **~40 Hz**
- 实际吞吐: ~1000 msg/s

### can1 (USB cando_can)
- 22电机稳定频率: **600 Hz**
- 极限频率: ~720 Hz (开始丢包)
- 实际吞吐: ~6600 msg/s
- 提升: **15 倍**

---

## 故障排除

### USB 设备未识别
```bash
# 检查 USB 设备
lsusb

# 检查内核日志
dmesg | tail -20 | grep -i can
```

### ttyACM0 不存在
```bash
# 重置 USB 设备
sudo usbreset <vendor_id>:<product_id>

# 或拔插 USB 线
```

### can1 启动失败
```bash
# 先清理旧进程
sudo pkill slcand
sudo ip link delete can1 2>/dev/null

# 重新启动
sudo slcand -o -c -s8 -S 4000000 /dev/ttyACM0 can1
sudo ip link set can1 up
```
