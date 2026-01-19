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

WAIST_AUTO_CONFIG = {
    'arm_reach': 0.55,          # 手臂有效作业半径(m)
    'deadzone': 0.05,           # 死区，避免微小移动触发腰部补偿
    'compensation_gain': 10.0,   # 腰部位置补偿增益
    'yaw_smooth': 0.1,          # yaw平滑系数 (0-1, 越小越平滑)
    'pos_smooth': 0.05,         # 位置补偿平滑系数
}

# 手臂向外向下偏置配置
ARM_BIAS_CONFIG = {
    'outward_bias': 0.15,       # 向外偏置强度 (shoulder_roll)
    'downward_bias': 0.1,       # 向下偏置强度 (elbow)
    'bias_cost': 1e-3,          # 偏置任务权重
}

# 腰部平滑状态
waist_smooth_state = {'quat': None, 'compensation': None}


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


def compute_waist_yaw_quat(hands_center, waist_pos):
    """计算腰部yaw四元数，让腰部朝向双手中心 (roll=0, pitch=0)"""
    direction = hands_center - waist_pos
    direction[2] = 0
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    yaw = np.arctan2(direction[1], direction[0])
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy, 0.0, 0.0, sy])


def compute_waist_compensation(hands_center, waist_init_pos, arm_reach, deadzone, gain):
    """计算腰部位置补偿"""
    diff = hands_center - waist_init_pos
    diff[2] = 0
    dist = np.linalg.norm(diff)
    if dist <= arm_reach - deadzone:
        return np.zeros(3)
    excess = dist - (arm_reach - deadzone)
    direction = diff / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    compensation = direction * excess * gain
    compensation[2] = 0
    return compensation


def compute_lookat_quat(head_pos, target_pos):
    """计算look-at四元数(MuJoCo wxyz格式)，头部X轴朝向目标，只使用yaw和pitch，roll冻结为0"""
    direction = target_pos - head_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction = direction / dist
    # 计算yaw(绕Z轴): 在XY平面上的投影方向
    yaw = np.arctan2(direction[1], direction[0])
    # 计算pitch(绕Y轴): 垂直方向的角度
    horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
    pitch = -np.arctan2(direction[2], horizontal_dist)  # 负号是因为pitch向下为正
    # 构建四元数: 先yaw后pitch (roll=0)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    # q = q_yaw * q_pitch (MuJoCo wxyz格式)
    w = cy * cp
    x = cy * sp
    y = sy * sp
    z = sy * cp
    return np.array([w, x, y, z])


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
        mink.PostureTask(model, cost=1e-2),
    ]
    for name, (link, _, _) in END_EFFECTORS.items():
        cost = (5.0, 5.0)  # 腰部位置姿态都关注
        tasks.append(mink.FrameTask(link, "body", position_cost=cost[0], orientation_cost=cost[1]))
    neck_task = mink.FrameTask("neck_pitch_link", "body", position_cost=1.0, orientation_cost=1.0)  # 头只关注姿态
    tasks.append(neck_task)
    ee_tasks = {name: tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
    
    # 手臂偏置任务 (向外向下)
    arm_posture_task = mink.PostureTask(model, cost=ARM_BIAS_CONFIG['bias_cost'])
    # 设置偏置目标: 左手向外(roll正), 右手向外(roll负), 胘弯曲
    bias_q = cfg.q.copy()
    # 右臂: shoulder_roll向外(负), elbow弯曲(负)
    bias_q[model.jnt_qposadr[model.joint("right_shoulder_roll_joint").id]] = -ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("right_elbow_joint").id]] = -ARM_BIAS_CONFIG['downward_bias']
    # 左臂: shoulder_roll向外(正), elbow弯曲(正)
    bias_q[model.jnt_qposadr[model.joint("left_shoulder_roll_joint").id]] = ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("left_elbow_joint").id]] = ARM_BIAS_CONFIG['downward_bias']
    arm_posture_task.set_target(bias_q)
    tasks.append(arm_posture_task)
    
    # neck_roll的DOF索引 (用于冻结)
    neck_roll_dof = int(model.jnt_dofadr[model.joint("neck_roll_joint").id])
    
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model),  # 限制关节速度
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
        
        # 初始化腰部平滑状态
        waist_smooth_state['quat'] = init_mocap_quat["waist"].copy()
        waist_smooth_state['compensation'] = np.zeros(3)
        
        prev_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        prev_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        print_counter = 0
        reset_duration = 1.5
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 计算双手中心点
            hands_center = (data.mocap_pos[left_hand_mid] + data.mocap_pos[right_hand_mid]) / 2.0
            
            # 腰部自动计算: yaw跟随双手, 位置超出范围时补偿
            waist_mid = mocap_ids["waist"]
            waist_init_pos = init_mocap_pos["waist"]
            cfg_w = WAIST_AUTO_CONFIG
            target_yaw_quat = compute_waist_yaw_quat(hands_center, waist_init_pos)
            waist_smooth_state['quat'] = slerp(waist_smooth_state['quat'], target_yaw_quat, cfg_w['yaw_smooth'])
            data.mocap_quat[waist_mid] = waist_smooth_state['quat']
            target_comp = compute_waist_compensation(
                hands_center, waist_init_pos, cfg_w['arm_reach'], cfg_w['deadzone'], cfg_w['compensation_gain'])
            waist_smooth_state['compensation'] = (
                (1 - cfg_w['pos_smooth']) * waist_smooth_state['compensation'] + cfg_w['pos_smooth'] * target_comp)
            data.mocap_pos[waist_mid] = waist_init_pos + waist_smooth_state['compensation']
            
            # 更新neck look-at目标 (只关注姿态)
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
                
                # neck始终激活并执行IK (neck_roll冻结)
                active_limbs = ["neck"]
                mask = np.zeros(model.nv, dtype=bool)
                for idx in joint_idx["neck"]:
                    if idx != neck_roll_dof:  # 跳过neck_roll
                        mask[idx] = True
                # 冻结neck_roll
                frozen_dofs_reset = [neck_roll_dof]
                constraints_reset = [mink.DofFreezingTask(model, dof_indices=frozen_dofs_reset)]
                tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.5, limits=limits, constraints=constraints_reset)
                vel[~mask] = 0.0
                cfg.integrate_inplace(vel, dt)
                
                if alpha >= 1.0:
                    reset_state["active"] = False
                    tasks[1].cost[:] = 1e-2  # 恢复posture cost
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
            
            # 检测活动肢体 (neck_roll始终冻结)
            active_dofs = set(int(i) for i in joint_idx["neck"]) - {neck_roll_dof}  # 移除neck_roll
            for name, mid in mocap_ids.items():
                pos_diff = data.mocap_pos[mid] - prev_pos[name]
                quat_diff = np.abs(data.mocap_quat[mid] - prev_quat[name])
                # waist也检测位置变化
                if np.dot(pos_diff, pos_diff) > 1e-7 or np.max(quat_diff) > 0.005:
                    for limb in ee_limbs[name]:
                        active_dofs.update(joint_idx[limb])
                prev_pos[name], prev_quat[name] = data.mocap_pos[mid].copy(), data.mocap_quat[mid].copy()
            
            # 构建冻结约束
            frozen_dofs = [i for i in all_dof_indices if i not in active_dofs]
            constraints = []
            if frozen_dofs:
                constraints.append(mink.DofFreezingTask(model, dof_indices=frozen_dofs))
            
            # 求解IK(damping提高奇异点稳定性)
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=2e-1, limits=limits, constraints=constraints)
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