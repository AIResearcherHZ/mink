#!/usr/bin/env python3
"""
测试更高频率，找到模块极限
"""

import time
import can

def test_hz(interface, hz, motor_count, duration=3):
    bus = can.interface.Bus(channel=interface, interface='socketcan')
    
    messages = []
    for i in range(1, motor_count + 1):
        msg = can.Message(
            arbitration_id=i,
            data=bytes([0x7F, 0xFF, 0x7F, 0xF0, 0x00, 0x00, 0x07, 0xFF]),
            is_extended_id=False
        )
        messages.append(msg)
    
    period = 1.0 / hz
    send_count = 0
    send_errors = 0
    cycle_count = 0
    
    start = time.perf_counter()
    
    while time.perf_counter() - start < duration:
        cycle_start = time.perf_counter()
        
        for msg in messages:
            try:
                bus.send(msg)
                send_count += 1
            except can.CanError:
                send_errors += 1
        cycle_count += 1
        
        elapsed = time.perf_counter() - cycle_start
        if elapsed < period:
            if period - elapsed > 0.001:
                time.sleep(period - elapsed - 0.0005)
            while time.perf_counter() - cycle_start < period:
                pass
    
    total_time = time.perf_counter() - start
    actual_hz = cycle_count / total_time
    error_pct = send_errors / (send_count + send_errors) * 100 if send_count + send_errors > 0 else 0
    
    bus.shutdown()
    return actual_hz, error_pct

def main():
    print(f"\n{'='*70}")
    print(f"📊 cando_can 高频率极限测试 - 22 个电机")
    print(f"{'='*70}\n")
    
    for hz in [500, 600, 700, 800, 900, 1000]:
        actual_hz, error = test_hz('can1', hz, 22, duration=3)
        status = "✅" if error < 1 else "⚠️" if error < 5 else "❌"
        print(f"  {hz:4d} Hz → 实际 {actual_hz:6.1f} Hz | 错误 {error:5.1f}% {status}")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
