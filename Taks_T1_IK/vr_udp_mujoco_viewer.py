"""
VR Pose MuJoCo 3D Viewer
基于UDP接收VR双手位姿数据，使用MuJoCo进行3D可视化显示

协议: JSON over UDP
端口: 7000 (默认)
"""

import socket
import json
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer

# UDP配置
UDP_IP = "0.0.0.0"
UDP_PORT = 7000
BUFFER_SIZE = 4096

# 全局数据存储
pose_data = {
    "head": {"position": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
    "leftHand": {"position": [-0.3, 0, 0.5], "quaternion": [1, 0, 0, 0], "gripper": 0},
    "rightHand": {"position": [0.3, 0, 0.5], "quaternion": [1, 0, 0, 0], "gripper": 0},
    "trackingEnabled": False
}
data_lock = threading.Lock()
running = True

# MuJoCo模型XML - 包含双手坐标系可视化
MUJOCO_XML = """
<mujoco model="vr_hands_viewer">
    <option gravity="0 0 0"/>
    
    <visual>
        <global offwidth="1920" offheight="1080"/>
        <quality shadowsize="4096"/>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    </visual>
    
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2" 
                 width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
        <material name="red" rgba="1 0.2 0.2 1"/>
        <material name="green" rgba="0.2 1 0.2 1"/>
        <material name="blue" rgba="0.2 0.2 1 1"/>
        <material name="left_hand" rgba="0.2 0.6 1 0.8"/>
        <material name="right_hand" rgba="1 0.6 0.2 0.8"/>
        <material name="head" rgba="0.8 0.8 0.8 0.8"/>
    </asset>
    
    <worldbody>
        <!-- 地面网格 -->
        <geom name="floor" type="plane" size="2 2 0.1" material="grid_mat" pos="0 0 0"/>
        
        <!-- 世界坐标系参考 -->
        <site name="world_x" type="cylinder" size="0.01 0.5" pos="0.5 0 0" euler="0 90 0" rgba="1 0 0 0.5"/>
        <site name="world_y" type="cylinder" size="0.01 0.5" pos="0 0.5 0" euler="90 0 0" rgba="0 1 0 0.5"/>
        <site name="world_z" type="cylinder" size="0.01 0.5" pos="0 0 0.5" rgba="0 0 1 0.5"/>
        <site name="world_origin" type="sphere" size="0.03" pos="0 0 0" rgba="1 1 1 0.8"/>
        
        <!-- 头部 -->
        <body name="head" pos="0 0 1.5">
            <freejoint name="head_joint"/>
            <geom name="head_geom" type="sphere" size="0.1" material="head"/>
            <!-- 头部坐标系 -->
            <site name="head_x" type="cylinder" size="0.008 0.15" pos="0.15 0 0" euler="0 90 0" rgba="1 0 0 1"/>
            <site name="head_y" type="cylinder" size="0.008 0.15" pos="0 0.15 0" euler="90 0 0" rgba="0 1 0 1"/>
            <site name="head_z" type="cylinder" size="0.008 0.15" pos="0 0 0.15" rgba="0 0 1 1"/>
        </body>
        
        <!-- 左手 -->
        <body name="left_hand" pos="-0.3 0 1">
            <freejoint name="left_hand_joint"/>
            <!-- 手掌 -->
            <geom name="left_palm" type="box" size="0.04 0.06 0.015" material="left_hand"/>
            <!-- 左手坐标系 - 更粗更明显 -->
            <site name="left_x" type="cylinder" size="0.01 0.12" pos="0.12 0 0" euler="0 90 0" rgba="1 0 0 1"/>
            <site name="left_y" type="cylinder" size="0.01 0.12" pos="0 0.12 0" euler="90 0 0" rgba="0 1 0 1"/>
            <site name="left_z" type="cylinder" size="0.01 0.12" pos="0 0 0.12" rgba="0 0 1 1"/>
            <site name="left_origin" type="sphere" size="0.02" pos="0 0 0" rgba="0.2 0.6 1 1"/>
            <!-- 坐标轴箭头 -->
            <site name="left_x_tip" type="sphere" size="0.015" pos="0.24 0 0" rgba="1 0 0 1"/>
            <site name="left_y_tip" type="sphere" size="0.015" pos="0 0.24 0" rgba="0 1 0 1"/>
            <site name="left_z_tip" type="sphere" size="0.015" pos="0 0 0.24" rgba="0 0 1 1"/>
        </body>
        
        <!-- 右手 -->
        <body name="right_hand" pos="0.3 0 1">
            <freejoint name="right_hand_joint"/>
            <!-- 手掌 -->
            <geom name="right_palm" type="box" size="0.04 0.06 0.015" material="right_hand"/>
            <!-- 右手坐标系 - 更粗更明显 -->
            <site name="right_x" type="cylinder" size="0.01 0.12" pos="0.12 0 0" euler="0 90 0" rgba="1 0 0 1"/>
            <site name="right_y" type="cylinder" size="0.01 0.12" pos="0 0.12 0" euler="90 0 0" rgba="0 1 0 1"/>
            <site name="right_z" type="cylinder" size="0.01 0.12" pos="0 0 0.12" rgba="0 0 1 1"/>
            <site name="right_origin" type="sphere" size="0.02" pos="0 0 0" rgba="1 0.6 0.2 1"/>
            <!-- 坐标轴箭头 -->
            <site name="right_x_tip" type="sphere" size="0.015" pos="0.24 0 0" rgba="1 0 0 1"/>
            <site name="right_y_tip" type="sphere" size="0.015" pos="0 0.24 0" rgba="0 1 0 1"/>
            <site name="right_z_tip" type="sphere" size="0.015" pos="0 0 0.24" rgba="0 0 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""


def unity_to_mujoco_position(pos):
    """
    Unity坐标系转MuJoCo坐标系
    原始数据已经是正确的，直接使用
    """
    if pos is None:
        return [0, 0, 0]
    # 直接使用原始数据
    return [pos[0], pos[1], pos[2]]


def unity_to_mujoco_quaternion(quat):
    """
    Unity四元数转MuJoCo四元数
    位置正确但旋转3个方向都反了，需要对虚部取反
    """
    if quat is None:
        return [1, 0, 0, 0]
    w, x, y, z = quat
    # 对虚部取反来修正旋转方向
    return [w, -x, -y, -z]


def udp_receiver_thread():
    """UDP数据接收线程"""
    global pose_data, running
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(False)
    sock.settimeout(0.1)
    
    print(f"[UDP] 监听 {UDP_IP}:{UDP_PORT}")
    
    while running:
        try:
            data_bytes, addr = sock.recvfrom(BUFFER_SIZE)
            data = json.loads(data_bytes.decode('utf-8'))
            
            with data_lock:
                # 更新头部数据
                if "head" in data:
                    head = data["head"]
                    if "position" in head:
                        pose_data["head"]["position"] = head["position"]
                    if "quaternion" in head:
                        pose_data["head"]["quaternion"] = head["quaternion"]
                
                # 更新左手数据
                if "leftHand" in data:
                    left = data["leftHand"]
                    if "position" in left:
                        pose_data["leftHand"]["position"] = left["position"]
                    if "quaternion" in left:
                        pose_data["leftHand"]["quaternion"] = left["quaternion"]
                    if "gripper" in left:
                        pose_data["leftHand"]["gripper"] = left["gripper"]
                
                # 更新右手数据
                if "rightHand" in data:
                    right = data["rightHand"]
                    if "position" in right:
                        pose_data["rightHand"]["position"] = right["position"]
                    if "quaternion" in right:
                        pose_data["rightHand"]["quaternion"] = right["quaternion"]
                    if "gripper" in right:
                        pose_data["rightHand"]["gripper"] = right["gripper"]
                
                # 更新追踪状态
                pose_data["trackingEnabled"] = data.get("trackingEnabled", False)
                
        except socket.timeout:
            continue
        except BlockingIOError:
            time.sleep(0.001)
        except json.JSONDecodeError as e:
            print(f"[UDP] JSON解析错误: {e}")
        except Exception as e:
            if running:
                print(f"[UDP] 错误: {e}")
    
    sock.close()
    print("[UDP] 接收器已停止")


def update_mujoco_pose(model, data):
    """更新MuJoCo模型中的位姿"""
    with data_lock:
        # 更新头部位姿
        head_pos = unity_to_mujoco_position(pose_data["head"]["position"])
        head_quat = unity_to_mujoco_quaternion(pose_data["head"]["quaternion"])
        
        head_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_joint")
        head_qpos_addr = model.jnt_qposadr[head_joint_id]
        data.qpos[head_qpos_addr:head_qpos_addr+3] = head_pos
        data.qpos[head_qpos_addr+3:head_qpos_addr+7] = head_quat
        
        # 更新左手位姿
        left_pos = unity_to_mujoco_position(pose_data["leftHand"]["position"])
        left_quat = unity_to_mujoco_quaternion(pose_data["leftHand"]["quaternion"])
        
        left_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_joint")
        left_qpos_addr = model.jnt_qposadr[left_joint_id]
        data.qpos[left_qpos_addr:left_qpos_addr+3] = left_pos
        data.qpos[left_qpos_addr+3:left_qpos_addr+7] = left_quat
        
        # 更新右手位姿
        right_pos = unity_to_mujoco_position(pose_data["rightHand"]["position"])
        right_quat = unity_to_mujoco_quaternion(pose_data["rightHand"]["quaternion"])
        
        right_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_joint")
        right_qpos_addr = model.jnt_qposadr[right_joint_id]
        data.qpos[right_qpos_addr:right_qpos_addr+3] = right_pos
        data.qpos[right_qpos_addr+3:right_qpos_addr+7] = right_quat
    
    # 前向运动学
    mujoco.mj_forward(model, data)


def main():
    global running
    
    print("=" * 50)
    print("VR Pose MuJoCo 3D Viewer")
    print("=" * 50)
    print(f"UDP端口: {UDP_PORT}")
    print("坐标系说明:")
    print("  - 红色(X轴), 绿色(Y轴), 蓝色(Z轴)")
    print("  - 蓝色手掌: 左手")
    print("  - 橙色手掌: 右手")
    print("  - 灰色球体: 头部")
    print("=" * 50)
    
    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)
    
    # 启动UDP接收线程
    udp_thread = threading.Thread(target=udp_receiver_thread, daemon=True)
    udp_thread.start()
    
    # 启动MuJoCo可视化
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机视角
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0, 0, 0.8]
            
            print("[Viewer] MuJoCo可视化窗口已启动")
            print("[Viewer] 等待VR数据...")
            
            while viewer.is_running():
                # 更新位姿
                update_mujoco_pose(model, data)
                
                # 同步可视化
                viewer.sync()
                
                # 控制刷新率 - 匹配 Unity 发送频率 (约100Hz)
                time.sleep(0.005)
                
    except Exception as e:
        print(f"[Viewer] 错误: {e}")
    finally:
        running = False
        print("[Viewer] 程序结束")


if __name__ == "__main__":
    main()
