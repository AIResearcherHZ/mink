"""半身独立肢体IK控制 - 稳定版

解决多路IK跳变: 使用DofFreezingTask作为equality constraint冻结非活动关节
"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_XML = Path(__file__).parent.parent / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"

# 关节分组
JOINT_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

# 末端执行器：手和waist
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target", ["left_arm"]),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target", ["right_arm"]),
    "waist": ("neck_yaw_link", "waist_target", ["waist"]),
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

# 全局状态
reset_state = {"active": False, "alpha": 0.0, "start_pos": {}, "start_quat": {}, "start_q": None}


def slerp(q0, q1, alpha):
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


def compute_lookat_quat(head_pos, target_pos):
    """计算look-at四元数(MuJoCo wxyz格式)，头部X轴朝向目标"""
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
    
    # 预计算DOF索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    # 所有可控DOF索引
    all_dof_indices = []
    for limb in JOINT_GROUPS:
        all_dof_indices.extend(joint_idx[limb])
    all_dof_indices = sorted(set(all_dof_indices))
    
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap, _) in END_EFFECTORS.items()}
    ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
    neck_pitch_mid = model.body("neck_pitch_target").mocapid[0]
    left_hand_mid, right_hand_mid = mocap_ids["left_hand"], mocap_ids["right_hand"]
    
    # 创建任务(固定cost)
    tasks = [
        mink.FrameTask("base_link", "body", position_cost=1e6, orientation_cost=1e6),  # 固定base_link
        mink.PostureTask(model, cost=1e-3),
    ]
    for name, (link, _, _) in END_EFFECTORS.items():
        cost = (0.0, 5.0) if name == "waist" else (5.0, 5.0)
        tasks.append(mink.FrameTask(link, "body", position_cost=cost[0], orientation_cost=cost[1]))
    neck_task = mink.FrameTask("neck_pitch_link", "body", position_cost=0.0, orientation_cost=3.0)
    tasks.append(neck_task)
    ee_tasks = {name: tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
    
    limits = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, gain=0.5, 
                                     minimum_distance_from_collisions=0.02, 
                                     collision_detection_distance=0.1)
    ]
    
    def key_callback(keycode):
        if keycode == 259:  # GLFW_KEY_BACKSPACE
            reset_state["active"] = True
            reset_state["alpha"] = 0.0
            reset_state["start_q"] = cfg.q.copy()
            for name, mid in mocap_ids.items():
                reset_state["start_pos"][name] = data.mocap_pos[mid].copy()
                reset_state["start_quat"][name] = data.mocap_quat[mid].copy()
            print("[Reset] 全局复位开始...")
    
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
        # 初始化neck_pitch mocap位置
        mink.move_mocap_to_frame(model, data, "neck_pitch_target", "neck_pitch_link", "body")
        neck_task.set_target_from_configuration(cfg)
        
        # 保存初始位置
        init_q = cfg.q.copy()
        init_mocap_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        init_mocap_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        
        prev_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        prev_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        print_counter = 0
        reset_duration = 1.5
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 计算双手中心点，更新neck look-at目标
            hands_center = (data.mocap_pos[left_hand_mid] + data.mocap_pos[right_hand_mid]) / 2.0
            head_pos = data.xpos[model.body("neck_pitch_link").id]
            lookat_quat = compute_lookat_quat(head_pos, hands_center)
            data.mocap_quat[neck_pitch_mid] = lookat_quat
            neck_task.set_target(mink.SE3.from_mocap_id(data, neck_pitch_mid))
            
            # 处理reset (neck保持look-at)
            if reset_state["active"]:
                reset_state["alpha"] += dt / reset_duration
                alpha = min(1.0, reset_state["alpha"])
                
                # 插值mocap到初始位置
                for name, mid in mocap_ids.items():
                    data.mocap_pos[mid] = (1 - alpha) * reset_state["start_pos"][name] + alpha * init_mocap_pos[name]
                    data.mocap_quat[mid] = slerp(reset_state["start_quat"][name], init_mocap_quat[name], alpha)
                    prev_pos[name] = data.mocap_pos[mid].copy()
                    prev_quat[name] = data.mocap_quat[mid].copy()
                
                # 插值configuration
                cfg.update(reset_state["start_q"] * (1 - alpha) + init_q * alpha)
                for name in END_EFFECTORS:
                    ee_tasks[name].set_target_from_configuration(cfg)
                
                # neck始终激活并执行IK
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
                    tasks[1].cost[:] = 1e-3  # 恢复posture cost
                    print("[Reset] 全局复位完成")
                
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = data.qfrc_bias[:]
                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()
                continue
            
            # 更新所有末端任务目标
            for name, mid in mocap_ids.items():
                ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
            
            # 检测活动肢体
            active_dofs = set(joint_idx["neck"])
            for name, mid in mocap_ids.items():
                pos_diff = data.mocap_pos[mid] - prev_pos[name]
                quat_diff = np.abs(data.mocap_quat[mid] - prev_quat[name])
                if name == "waist":
                    if np.max(quat_diff) > 0.005:
                        for limb in ee_limbs[name]:
                            active_dofs.update(joint_idx[limb])
                else:
                    if np.dot(pos_diff, pos_diff) > 1e-7 or np.max(quat_diff) > 0.005:
                        for limb in ee_limbs[name]:
                            active_dofs.update(joint_idx[limb])
                prev_pos[name], prev_quat[name] = data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy()
            
            # 构建冻结约束
            frozen_dofs = [i for i in all_dof_indices if i not in active_dofs]
            constraints = []
            if frozen_dofs:
                constraints.append(mink.DofFreezingTask(model, dof_indices=frozen_dofs))
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=1e-3, limits=limits, constraints=constraints)
            cfg.integrate_inplace(vel, dt)
            
            # 前馈扭矩补偿
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                print(f"[Compensation] :\n  {np.array2string(data.qfrc_applied[6:], precision=3, suppress_small=True)}")
                print("\n[Mocap状态]:")
                for name, mid in mocap_ids.items():
                    pos = data.mocap_pos[mid]
                    quat = data.mocap_quat[mid]
                    print(f"  {name}: pos={np.array2string(pos, precision=3, suppress_small=True)}, quat={np.array2string(quat, precision=3, suppress_small=True)}")
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()