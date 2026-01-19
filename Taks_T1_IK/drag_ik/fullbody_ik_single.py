"""全身单路IK控制 - Y字形解算

头部look-at独立计算，不参与IK解算
腰部低权重：大部分时间直立，只有yaw跟随双手，超距时才位置补偿
"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_XML = Path(__file__).parent.parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml"

# 关节分组
JOINT_GROUPS = {
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                 "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
}

# 末端执行器
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target"),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target"),
    "left_foot": ("left_ankle_roll_link", "left_foot_target"),
    "right_foot": ("right_ankle_roll_link", "right_foot_target"),
    "waist": ("neck_yaw_link", "waist_target"),
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
    (["left_foot_collision"], ["right_foot_collision"]),
    (["left_hip_collision"], ["right_hip_collision"]),
    (["left_hand_collision"], ["left_hip_collision", "left_foot_collision"]),
    (["right_hand_collision"], ["right_hip_collision", "right_foot_collision"]),
    (["left_hand_collision"], ["right_hip_collision", "right_foot_collision"]),
    (["right_hand_collision"], ["left_hip_collision", "left_foot_collision"]),
    (["left_foot_collision"], ["torso_collision"]),
    (["right_foot_collision"], ["torso_collision"]),
    (["left_foot_collision", "right_foot_collision"], ["floor"]),
]

# 复位状态
reset_state = {"active": False, "alpha": 0.0, "start_pos": {}, "start_quat": {}, "start_q": None}

# 腰部自动计算配置 - 大死区，低权重，只有超距才补偿
WAIST_AUTO_CONFIG = {
    'arm_reach': 0.55,          # 手臂有效作业半径(m)
    'deadzone': 0.40,           # 大死区，双手在这个范围内腰部不动
    'compensation_gain': 0.8,   # 超距补偿增益
    'yaw_smooth': 0.08,         # yaw平滑系数
    'pos_smooth': 0.03,         # 位置补偿平滑系数
}

# 手臂向外向下偏置配置
ARM_BIAS_CONFIG = {
    'outward_bias': 0.15,
    'downward_bias': 0.1,
    'bias_cost': 1e-3,
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
    """计算腰部yaw四元数，让腰部朝向双手中心"""
    direction = hands_center - waist_pos
    direction[2] = 0
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    yaw = np.arctan2(direction[1], direction[0])
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cy, 0.0, 0.0, sy])


def compute_waist_compensation(hands_center, waist_init_pos, arm_reach, deadzone, gain):
    """计算腰部位置补偿 - 只有超出作业范围才补偿"""
    diff = hands_center - waist_init_pos
    diff[2] = 0
    dist = np.linalg.norm(diff)
    # 在死区范围内不补偿
    if dist <= arm_reach - deadzone:
        return np.zeros(3)
    excess = dist - (arm_reach - deadzone)
    direction = diff / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    compensation = direction * excess * gain
    compensation[2] = 0
    return compensation


def compute_lookat_quat(head_pos, target_pos):
    """计算look-at四元数(MuJoCo wxyz格式)，只使用yaw和pitch，roll=0"""
    direction = target_pos - head_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    direction = direction / dist
    yaw = np.arctan2(direction[1], direction[0])
    horizontal_dist = np.sqrt(direction[0]**2 + direction[1]**2)
    pitch = -np.arctan2(direction[2], horizontal_dist)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    w = cy * cp
    x = cy * sp
    y = sy * sp
    z = sy * cp
    return np.array([w, x, y, z])


def set_neck_from_lookat(cfg, model, lookat_quat):
    """直接设置neck关节角度实现look-at，不参与IK解算"""
    # 从四元数提取yaw和pitch
    w, x, y, z = lookat_quat
    # 计算yaw (绕Z轴)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    # 计算pitch (绕Y轴)
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    # 设置neck关节
    neck_yaw_idx = model.jnt_qposadr[model.joint("neck_yaw_joint").id]
    neck_pitch_idx = model.jnt_qposadr[model.joint("neck_pitch_joint").id]
    neck_roll_idx = model.jnt_qposadr[model.joint("neck_roll_joint").id]
    q = cfg.q.copy()
    q[neck_yaw_idx] = yaw
    q[neck_pitch_idx] = pitch
    q[neck_roll_idx] = 0.0  # roll冻结为0
    return q


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算DOF索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    # 非neck的所有DOF索引 (neck独立控制)
    ik_dof_indices = []
    for limb in JOINT_GROUPS:
        if limb != "neck":
            ik_dof_indices.extend(joint_idx[limb])
    ik_dof_indices = sorted(set(ik_dof_indices))
    
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap) in END_EFFECTORS.items()}
    left_hand_mid, right_hand_mid = mocap_ids["left_hand"], mocap_ids["right_hand"]
    
    # 创建IK任务 - 不包含neck任务
    tasks = [
        mink.FrameTask("pelvis", "body", position_cost=1e6, orientation_cost=1e6),  # 固定pelvis
        mink.PostureTask(model, cost=1e-2),
    ]
    # 末端执行器任务
    ee_tasks = {}
    for name, (link, _) in END_EFFECTORS.items():
        if name == "waist":
            # 腰部低权重 - 大部分时间保持直立
            task = mink.FrameTask(link, "body", position_cost=0.5, orientation_cost=0.5)
        else:
            task = mink.FrameTask(link, "body", position_cost=2.0, orientation_cost=2.0)
        tasks.append(task)
        ee_tasks[name] = task
    
    # 手臂偏置任务
    arm_posture_task = mink.PostureTask(model, cost=ARM_BIAS_CONFIG['bias_cost'])
    bias_q = cfg.q.copy()
    bias_q[model.jnt_qposadr[model.joint("right_shoulder_roll_joint").id]] = -ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("right_elbow_joint").id]] = -ARM_BIAS_CONFIG['downward_bias']
    bias_q[model.jnt_qposadr[model.joint("left_shoulder_roll_joint").id]] = ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("left_elbow_joint").id]] = ARM_BIAS_CONFIG['downward_bias']
    arm_posture_task.set_target(bias_q)
    tasks.append(arm_posture_task)
    
    # neck DOF索引 (用于冻结) - 转换为int
    neck_dof_indices = [int(i) for i in joint_idx["neck"]]
    
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model),
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
        cfg.update_from_keyframe("stand")
        tasks[0].set_target_from_configuration(cfg)
        tasks[1].set_target_from_configuration(cfg)
        for name, (link, mocap) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(model, data, mocap, link, "body")
            ee_tasks[name].set_target_from_configuration(cfg)
        
        # 保存初始位置
        init_q = cfg.q.copy()
        init_mocap_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        init_mocap_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        
        # 初始化腰部平滑状态
        waist_smooth_state['quat'] = init_mocap_quat["waist"].copy()
        waist_smooth_state['compensation'] = np.zeros(3)
        
        print_counter = 0
        reset_duration = 1.5
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 计算双手中心点
            hands_center = (data.mocap_pos[left_hand_mid] + data.mocap_pos[right_hand_mid]) / 2.0
            
            # 腰部自动计算 - yaw跟随，位置超距补偿
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
            
            # 头部look-at独立计算
            head_pos = data.xpos[model.body("neck_pitch_link").id]
            lookat_quat = compute_lookat_quat(head_pos, hands_center)
            
            # 处理reset
            if reset_state["active"]:
                reset_state["alpha"] += dt / reset_duration
                alpha = min(1.0, reset_state["alpha"])
                
                for name, mid in mocap_ids.items():
                    data.mocap_pos[mid] = (1 - alpha) * reset_state["start_pos"][name] + alpha * init_mocap_pos[name]
                    data.mocap_quat[mid] = slerp(reset_state["start_quat"][name], init_mocap_quat[name], alpha)
                
                cfg.update(reset_state["start_q"] * (1 - alpha) + init_q * alpha)
                for name in END_EFFECTORS:
                    ee_tasks[name].set_target_from_configuration(cfg)
                
                if alpha >= 1.0:
                    reset_state["active"] = False
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
            
            # 冻结neck关节，让头部独立控制
            constraints = [mink.DofFreezingTask(model, dof_indices=neck_dof_indices)]
            
            # 单路IK解算
            try:
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=1e-1, limits=limits, constraints=constraints)
            except mink.exceptions.NoSolutionFound:
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=1.0, limits=limits, constraints=[])
            cfg.integrate_inplace(vel, dt)
            
            # 独立设置neck关节实现look-at
            q_with_neck = set_neck_from_lookat(cfg, model, lookat_quat)
            cfg.update(q_with_neck)
            
            # 前馈扭矩补偿
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                print(f"[Compensation]:\n  {np.array2string(data.qfrc_applied[6:], precision=3, suppress_small=True)}")
                print(f"[Waist] compensation: {np.array2string(waist_smooth_state['compensation'], precision=3)}")
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
