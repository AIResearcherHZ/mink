"""半身VR控制IK"""

import sys
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

sys.path.insert(0, str(Path(__file__).parent))
from vr_interface import VRReceiver

_XML = Path(__file__).parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"

# 关节分组
JOINT_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

# 末端执行器
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target", ["left_arm"]),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target", ["right_arm"]),
    "waist": ("torso_link", "waist_target", ["waist"]),
}

# 碰撞对
COLLISION_PAIRS = [
    (["left_hand_collision"], ["torso_collision"]),
    (["right_hand_collision"], ["torso_collision"]),
    (["left_elbow_collision"], ["torso_collision"]),
    (["right_elbow_collision"], ["torso_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_elbow_collision"], ["right_elbow_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["head_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_elbow_collision"]),
    (["right_hand_collision"], ["left_elbow_collision"]),
]

# 速度限制参数(过滤抖动)
MAX_VEL = 100.0  # rad/s

# 动作EMA平滑参数
# alpha越大响应越快，越小越平滑但有延迟
# 0.3-0.5: 较平滑, 0.6-0.8: 快速响应, 1.0: 无平滑
ACTION_EMA_ALPHA = 0.7

# 全局状态
reset_state = {"active": False, "alpha": 0.0, "start_pos": {}, "start_quat": {}, "start_q": None}


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """四元数球面插值"""
    q0, q1 = q0.copy(), q1.copy()
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = (1 - alpha) * q0 + alpha * q1
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        result = (np.sin((1-alpha)*theta)*q0 + np.sin(alpha*theta)*q1) / np.sin(theta)
    return result / np.linalg.norm(result)


def compute_lookat_quat(head_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """计算look-at四元数(MuJoCo wxyz格式)"""
    direction = target_pos - head_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction = direction / dist
    forward = np.array([1.0, 0.0, 0.0])
    dot = np.clip(np.dot(forward, direction), -1.0, 1.0)
    if dot > 0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = np.cross(forward, direction)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap, _) in END_EFFECTORS.items()}
    ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
    neck_pitch_mid = model.body("neck_pitch_target").mocapid[0]
    
    # 创建任务
    tasks = [
        mink.FrameTask("base_link", "body", position_cost=0.0, orientation_cost=0.0),
        mink.PostureTask(model, cost=1e-2),
    ]
    for name, (link, _, _) in END_EFFECTORS.items():
        if name == "waist":
            tasks.append(mink.FrameTask(link, "body", position_cost=5.0, orientation_cost=5.0))
        else:
            tasks.append(mink.FrameTask(link, "body", position_cost=5.0, orientation_cost=5.0))
    neck_task = mink.FrameTask("neck_pitch_link", "body", position_cost=0.0, orientation_cost=5.0)
    tasks.append(neck_task)
    
    ee_tasks = {name: tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
    limits = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, gain=0.5, 
                                     minimum_distance_from_collisions=0.02, 
                                     collision_detection_distance=0.1)
    ]
    
    # VR接收器
    vr = VRReceiver()
    vr.start()
    
    # VR校准状态: 包含手部偏移和头部到腰部的偏移
    vr_state = {
        "calibrated": False,
        "left_offset": np.zeros(3),
        "right_offset": np.zeros(3),
        "head_to_waist": np.zeros(3),  # VR头部到MuJoCo腰部的偏移
    }
    
    def key_callback(keycode: int) -> None:
        if keycode == 259:  # BACKSPACE: 复位
            reset_state["active"] = True
            reset_state["alpha"] = 0.0
            reset_state["start_q"] = cfg.q.copy()
            for name, mid in mocap_ids.items():
                reset_state["start_pos"][name] = data.mocap_pos[mid].copy()
                reset_state["start_quat"][name] = data.mocap_quat[mid].copy()
            print("[Reset] 全局复位开始...")
        elif keycode == 67:  # C: VR校准
            vr_data = vr.data
            if vr_data.tracking_enabled:
                # 手部偏移: MuJoCo mocap位置 - VR手部位置
                left_mocap = data.mocap_pos[mocap_ids["left_hand"]]
                right_mocap = data.mocap_pos[mocap_ids["right_hand"]]
                vr_state["left_offset"] = left_mocap - vr_data.left_hand.position
                vr_state["right_offset"] = right_mocap - vr_data.right_hand.position
                # 头部到腰部偏移: MuJoCo腰部位置 - VR头部位置
                waist_pos = data.mocap_pos[mocap_ids["waist"]]
                vr_state["head_to_waist"] = waist_pos - vr_data.head.position
                vr_state["calibrated"] = True
                print(f"[VR] 校准完成:")
                print(f"  L={vr_state['left_offset']}")
                print(f"  R={vr_state['right_offset']}")
                print(f"  Head->Waist={vr_state['head_to_waist']}")
            else:
                print("[VR] 未启用追踪，无法校准")
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False, 
                                       key_callback=key_callback) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 初始化
        cfg.update_from_keyframe("home")
        tasks[0].set_target_from_configuration(cfg)
        tasks[1].set_target_from_configuration(cfg)
        for name, (link, mocap, _) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(model, data, mocap, link, "body")
            ee_tasks[name].set_target_from_configuration(cfg)
        mink.move_mocap_to_frame(model, data, "neck_pitch_target", "neck_pitch_link", "body")
        neck_task.set_target_from_configuration(cfg)
        
        # 保存初始位置
        init_q = cfg.q.copy()
        init_mocap_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        init_mocap_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        
        prev_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        prev_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        print_counter = 0
        threshold_sq = 1e-6
        quat_threshold = 0.01
        reset_duration = 1.5
        
        left_hand_mid = mocap_ids["left_hand"]
        right_hand_mid = mocap_ids["right_hand"]
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        filtered_vel = np.zeros(model.nv)  # EMA平滑状态
        
        print("[Info] 键盘: C=校准, Backspace=复位 | VR手柄: B双击=校准, A双击=复位")
        
        while viewer.is_running():
            # 获取VR数据并更新mocap
            vr_data = vr.data
            
            # 检测 VR按键双击事件: B双击=校准, A双击=复位
            if vr_data.button_events.right_b:  # B双击: 校准
                if vr_data.tracking_enabled:
                    left_mocap = data.mocap_pos[mocap_ids["left_hand"]]
                    right_mocap = data.mocap_pos[mocap_ids["right_hand"]]
                    vr_state["left_offset"] = left_mocap - vr_data.left_hand.position
                    vr_state["right_offset"] = right_mocap - vr_data.right_hand.position
                    waist_pos = data.mocap_pos[mocap_ids["waist"]]
                    vr_state["head_to_waist"] = waist_pos - vr_data.head.position
                    vr_state["calibrated"] = True
                    print(f"[VR] B双击校准完成: L={vr_state['left_offset']}, R={vr_state['right_offset']}")
                else:
                    print("[VR] 未启用追踪，无法校准")
            
            if vr_data.button_events.right_a:  # A双击: 复位
                reset_state["active"] = True
                reset_state["alpha"] = 0.0
                reset_state["start_q"] = cfg.q.copy()
                for name, mid in mocap_ids.items():
                    reset_state["start_pos"][name] = data.mocap_pos[mid].copy()
                    reset_state["start_quat"][name] = data.mocap_quat[mid].copy()
                print("[VR] A双击复位开始...")
            
            if vr_state["calibrated"] and vr_data.tracking_enabled:
                # 手部位置 = VR手部 + 偏移
                data.mocap_pos[left_hand_mid] = vr_data.left_hand.position + vr_state["left_offset"]
                data.mocap_quat[left_hand_mid] = vr_data.left_hand.quaternion
                data.mocap_pos[right_hand_mid] = vr_data.right_hand.position + vr_state["right_offset"]
                data.mocap_quat[right_hand_mid] = vr_data.right_hand.quaternion
                # 腰部: 位置 = VR头部 + 头到腰偏移, 姿态 = VR头部姿态
                waist_mid = mocap_ids["waist"]
                data.mocap_pos[waist_mid] = vr_data.head.position + vr_state["head_to_waist"]
                data.mocap_quat[waist_mid] = vr_data.head.quaternion
            
            # look-at目标
            hands_center = (data.mocap_pos[left_hand_mid] + data.mocap_pos[right_hand_mid]) / 2.0
            head_pos = data.xpos[model.body("neck_pitch_link").id]
            lookat_quat = compute_lookat_quat(head_pos, hands_center)
            data.mocap_quat[neck_pitch_mid] = lookat_quat
            neck_task.set_target(mink.SE3.from_mocap_id(data, neck_pitch_mid))
            
            # 处理reset
            if reset_state["active"]:
                reset_state["alpha"] += dt / reset_duration
                alpha = min(1.0, reset_state["alpha"])
                
                for name, mid in mocap_ids.items():
                    data.mocap_pos[mid] = (1 - alpha) * reset_state["start_pos"][name] + alpha * init_mocap_pos[name]
                    data.mocap_quat[mid] = slerp(reset_state["start_quat"][name], init_mocap_quat[name], alpha)
                    prev_pos[name] = data.mocap_pos[mid].copy()
                    prev_quat[name] = data.mocap_quat[mid].copy()
                
                cfg.update(reset_state["start_q"] * (1 - alpha) + init_q * alpha)
                for name in END_EFFECTORS:
                    ee_tasks[name].set_target_from_configuration(cfg)
                
                active_limbs = ["neck"]
                mask = np.zeros(model.nv, dtype=bool)
                for idx in joint_idx["neck"]:
                    mask[idx] = True
                tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.5, limits=limits)
                vel[~mask] = 0.0
                cfg.integrate_inplace(vel, dt)
                
                if alpha >= 1.0:
                    reset_state["active"] = False
                    print("[Reset] 全局复位完成")
                
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = data.qfrc_bias[:]
                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()
                continue
            
            # 检测移动的mocap
            active_limbs = []
            for name, mid in mocap_ids.items():
                pos_diff = data.mocap_pos[mid] - prev_pos[name]
                quat_diff = np.abs(data.mocap_quat[mid] - prev_quat[name])
                pos_changed = np.dot(pos_diff, pos_diff) > threshold_sq
                quat_changed = np.max(quat_diff) > quat_threshold
                
                ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
                
                if pos_changed or quat_changed:
                    active_limbs.extend(ee_limbs[name])
                
                prev_pos[name] = data.mocap_pos[mid].copy()
                prev_quat[name] = data.mocap_quat[mid].copy()
            
            active_limbs.append("neck")
            active_limbs = list(set(active_limbs))
            
            # 动态调整posture cost
            if active_limbs:
                mask = np.zeros(model.nv, dtype=bool)
                for limb in active_limbs:
                    if limb in joint_idx:
                        for idx in joint_idx[limb]:
                            mask[idx] = True
                tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
            else:
                tasks[1].cost[:] = 1e-2
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.5, limits=limits)
            
            if active_limbs:
                mask = np.zeros(model.nv, dtype=bool)
                for limb in active_limbs:
                    if limb in joint_idx:
                        for idx in joint_idx[limb]:
                            mask[idx] = True
                vel[~mask] = 0.0
            else:
                vel[:] = 0.0
            
            # 速度限制(过滤抖动)
            vel = np.clip(vel, -MAX_VEL, MAX_VEL)
            # 动作EMA平滑(neck关节不平滑，保持look-at响应)
            neck_mask = np.zeros(model.nv, dtype=bool)
            for idx in joint_idx["neck"]:
                neck_mask[idx] = True
            filtered_vel[~neck_mask] = ACTION_EMA_ALPHA * vel[~neck_mask] + (1 - ACTION_EMA_ALPHA) * filtered_vel[~neck_mask]
            filtered_vel[neck_mask] = vel[neck_mask]  # neck直接使用原始速度
            cfg.integrate_inplace(filtered_vel, dt)
            
            # 前馈扭矩补偿
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                tracking_str = "ON" if vr_data.tracking_enabled else "OFF"
                calib_str = "YES" if vr_state["calibrated"] else "NO"
                print(f"[VR] Tracking={tracking_str}, Calibrated={calib_str}")
                print(f"[Compensation]:\n  {np.array2string(data.qfrc_applied[6:], precision=3, suppress_small=True)}")
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
    
    vr.stop()