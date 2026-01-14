import math
from DM_Motor import *
import serial
import time
import threading

Motor1=Motor(DM_Motor_Type.DM4340,0x01,0x81)
Motor2=Motor(DM_Motor_Type.DM6248P,0x0A,0x8A)
# 右手串口控制Motor1
serial_device_right = serial.Serial('/dev/ttyACM0', 921600, timeout=0.5)
# 左手串口控制Motor2
serial_device_left = serial.Serial('/dev/ttyACM1', 921600, timeout=0.5)
MotorControl1=MotorControl(serial_device_right)
MotorControl2=MotorControl(serial_device_left)
MotorControl1.addMotor(Motor1)
MotorControl2.addMotor(Motor2)

if MotorControl1.switchControlMode(Motor1,Control_Type.MIT):
    print("Motor1切换MIT模式成功")
if MotorControl2.switchControlMode(Motor2,Control_Type.MIT):
    print("Motor2切换MIT模式成功")
print("sub_ver:",MotorControl1.read_motor_param(Motor1,DM_variable.sub_ver))
print("Gr:",MotorControl1.read_motor_param(Motor1,DM_variable.Gr))

# 修改电机参数示例
# if MotorControl1.change_motor_param(Motor1,DM_variable.KP_APR,54):
#     print("写入成功")
print("PMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.PMAX))
print("MST_ID:",MotorControl1.read_motor_param(Motor1,DM_variable.MST_ID))
print("VMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.VMAX))
print("TMAX:",MotorControl1.read_motor_param(Motor1,DM_variable.TMAX))
print("Motor2:")
print("PMAX:",MotorControl2.read_motor_param(Motor2,DM_variable.PMAX))
print("MST_ID:",MotorControl2.read_motor_param(Motor2,DM_variable.MST_ID))
print("VMAX:",MotorControl2.read_motor_param(Motor2,DM_variable.VMAX))
print("TMAX:",MotorControl2.read_motor_param(Motor2,DM_variable.TMAX))
MotorControl1.save_motor_param(Motor1)
MotorControl2.save_motor_param(Motor2)
MotorControl1.enable(Motor1)
MotorControl2.enable(Motor2)
# Motor1控制线程
def control_motor1():
    i=0
    while i<10000:
        q=math.sin(time.time())
        i=i+1
        # MotorControl1.control_pos_force(Motor1, 10, 1000,100)
        # MotorControl1.control_Vel(Motor1, q*8)
        # MotorControl1.control_Pos_Vel(Motor1,q*8,30)
        # print("Motor1:","POS:",Motor1.getPosition(),"VEL:",Motor1.getVelocity(),"TORQUE:",Motor1.getTorque())
        print(Motor1.getPosition())
        MotorControl1.controlMIT(Motor1, 10, 0.1, q*8, 0, 0)
        time.sleep(0.001)

# Motor2控制线程
def control_motor2():
    i=0
    while i<10000:
        q=math.sin(time.time())
        i=i+1
        # MotorControl2.control_pos_force(Motor2, 10, 1000,100)
        # MotorControl2.control_Vel(Motor2, q*8)
        # MotorControl2.control_Pos_Vel(Motor2,q*8,30)
        # print("Motor2:","POS:",Motor2.getPosition(),"VEL:",Motor2.getVelocity(),"TORQUE:",Motor2.getTorque())
        print(Motor2.getPosition())
        # MotorControl2.controlMIT(Motor2, 10, 0.1, q*8, 0, 0)
        time.sleep(0.001)

# 启动两个线程
thread1 = threading.Thread(target=control_motor1)
thread2 = threading.Thread(target=control_motor2)
thread1.start()
thread2.start()

# 等待线程结束
thread1.join()
thread2.join()

# 关闭串口
serial_device_right.close()
serial_device_left.close()