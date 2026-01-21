import math
from DM_CAN_FD import *
import time
import threading

Motor1 = Motor(DM_Motor_Type.DM4340, 0x01, 0x81)
Motor2 = Motor(DM_Motor_Type.DM6248P, 0x0A, 0x8A)

MotorControl1 = MotorControlFD(can_interface='can0', bitrate=5000000)
MotorControl2 = MotorControlFD(can_interface='can1', bitrate=5000000)

MotorControl1.addMotor(Motor1)
MotorControl2.addMotor(Motor2)

if MotorControl1.switchControlMode(Motor1, Control_Type.MIT):
    print("Motor1切换MIT模式成功")
if MotorControl2.switchControlMode(Motor2, Control_Type.MIT):
    print("Motor2切换MIT模式成功")

print("sub_ver:", MotorControl1.read_motor_param(Motor1, DM_variable.sub_ver))
print("Gr:", MotorControl1.read_motor_param(Motor1, DM_variable.Gr))

print("PMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.PMAX))
print("MST_ID:", MotorControl1.read_motor_param(Motor1, DM_variable.MST_ID))
print("VMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.VMAX))
print("TMAX:", MotorControl1.read_motor_param(Motor1, DM_variable.TMAX))
print("Motor2:")
print("PMAX:", MotorControl2.read_motor_param(Motor2, DM_variable.PMAX))
print("MST_ID:", MotorControl2.read_motor_param(Motor2, DM_variable.MST_ID))
print("VMAX:", MotorControl2.read_motor_param(Motor2, DM_variable.VMAX))
print("TMAX:", MotorControl2.read_motor_param(Motor2, DM_variable.TMAX))

MotorControl1.save_motor_param(Motor1)
MotorControl2.save_motor_param(Motor2)
MotorControl1.enable(Motor1)
MotorControl2.enable(Motor2)

def control_motor1():
    i = 0
    while i < 10000:
        q = math.sin(time.time())
        i = i + 1
        print(Motor1.getPosition())
        MotorControl1.controlMIT(Motor1, 10, 0.1, q*8, 0, 0)
        time.sleep(0.001)

def control_motor2():
    i = 0
    while i < 10000:
        q = math.sin(time.time())
        i = i + 1
        print(Motor2.getPosition())
        time.sleep(0.001)

thread1 = threading.Thread(target=control_motor1)
thread2 = threading.Thread(target=control_motor2)
thread1.start()
thread2.start()

thread1.join()
thread2.join()

MotorControl1.close()
MotorControl2.close()