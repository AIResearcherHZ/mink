"""全身单路IK控制 - 重构版
头部look-at独立计算，腰部纯hardcode补偿（大部分时间直立，超距才补偿）
"""

import numpy as np
import mujoco
import mujoco.viewer
import mink
from pathlib import Path
from loop_rate_limiters import RateLimiter

# ==================== 配置 ====================

_XML = Path(__file__).parent.parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml"

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

# 末端执行器（腰部不参与IK）
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target"),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target"),
    "left_foot": ("left_ankle_roll_link", "left_foot_target"),
    "right_foot": ("right_ankle_roll_link", "right_foot_target"),
}

# 碰撞对
COLLISION_PAIRS = [
    (["torso_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["torso_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_elbow_collision"], ["right_elbow_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["head_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_hand_collision"], ["right_elbow_collision"]),
    (["right_hand_collision"], ["left_elbow_collision"]),
    (["left_hand_collision"], ["left_hip_collision", "left_foot_collision"]),
    (["left_hand_collision"], ["right_hip_collision", "right_foot_collision"]),
    (["right_hand_collision"], ["left_hip_collision", "left_foot_collision"]),
    (["left_foot_collision"], ["torso_collision"]),
    (["right_foot_collision"], ["torso_collision"]),
    (["left_foot_collision", "right_foot_collision"], ["floor"]),
    (["pelvis_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["pelvis_collision"], ["left_elbow_collision", "right_elbow_collision"]),
    (["left_shoulder_roll_collision"], ["right_hand_collision", "right_elbow_collision"]),
    (["right_shoulder_roll_collision"], ["left_hand_collision", "left_elbow_collision"]),
]

# 腰部配置
WAIST_CONFIG = {
    'arm_reach': 0.55,
    'deadzone': 0.10,
    'compensation_gain': 1.0,
    'yaw_smooth': 0.03,
}

# 手臂偏置配置
ARM_BIAS_CONFIG = {'outward_bias': 0.35, 'downward_bias': 0.2, 'bias_cost': 1e-1}


# ==================== 工具函数 ====================

def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值"""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def compute_waist_yaw(hands_center: np.ndarray, waist_pos: np.ndarray, 
                      inner_radius: float = 0.20, outer_radius: float = 0.40) -> float:
    """计算腰部yaw，内外圈逻辑：内圈保持0位，外圈跟随目标，过渡区平滑插值"""
    diff = hands_center - waist_pos
    diff[2] = 0
    dist = float(np.linalg.norm(diff))
    
    if dist <= inner_radius:
        return 0.0
    
    target_yaw = float(np.arctan2(diff[1], diff[0]))
    
    if dist >= outer_radius:
        return target_yaw
    
    blend = (dist - inner_radius) / (outer_radius - inner_radius)
    return target_yaw * blend


def compute_waist_compensation(forward_dist: float, arm_reach: float, deadzone: float, gain: float) -> float:
    """计算腰部pitch补偿，使用单手最大前伸距离（正值前倾，负值后仰）"""
    # 前向补偿：超过arm_reach-deadzone才开始
    forward_threshold = arm_reach - deadzone
    # 后向补偿：降低阈值，更早触发
    backward_threshold = 0.1
    
    if forward_dist > 0:
        # 前倾补偿
        if forward_dist <= forward_threshold:
            return 0.0
        excess = forward_dist - forward_threshold
        return float(np.clip(excess * gain, 0.0, 0.4))
    else:
        # 后仰补偿（手往后背时）
        backward_dist = -forward_dist
        if backward_dist <= backward_threshold:
            return 0.0
        excess = backward_dist - backward_threshold
        # 增强后仰补偿，使用与前倾相同的gain
        return float(np.clip(-excess * gain * 0.8, -0.35, 0.0))


def compute_local_forward_dist(hand_pos: np.ndarray, waist_pos: np.ndarray, waist_yaw: float) -> float:
    """计算手在腰部局部坐标系下的前向距离（考虑yaw旋转）"""
    diff = hand_pos - waist_pos
    diff[2] = 0
    # 将世界坐标系的diff旋转到腰部局部坐标系
    cos_yaw = np.cos(-waist_yaw)
    sin_yaw = np.sin(-waist_yaw)
    local_x = diff[0] * cos_yaw - diff[1] * sin_yaw
    return float(local_x)


def compute_neck_angles(head_pos: np.ndarray, target_pos: np.ndarray, prev_yaw: float, prev_pitch: float, 
                        waist_yaw: float = 0.0, inner_radius: float = 0.3, outer_radius: float = 0.6) -> tuple:
    """计算neck关节角度（yaw, pitch），内外圈逻辑：内圈保持0位，外圈跟随目标"""
    direction = target_pos - head_pos
    dist = float(np.linalg.norm(direction))
    if dist < 1e-6:
        return prev_yaw, prev_pitch
    direction /= dist
    
    # 内圈：完全不动，保持0位
    if dist <= inner_radius:
        target_yaw = 0.0
        target_pitch = 0.0
    # 外圈：跟随目标
    elif dist >= outer_radius:
        world_yaw = float(np.arctan2(direction[1], direction[0]))
        raw_yaw = world_yaw - waist_yaw
        while raw_yaw > np.pi:
            raw_yaw -= 2 * np.pi
        while raw_yaw < -np.pi:
            raw_yaw += 2 * np.pi
        horizontal_dist = float(np.sqrt(direction[0]**2 + direction[1]**2))
        pitch = float(-np.arctan2(direction[2], horizontal_dist))
        target_yaw = float(np.clip(raw_yaw, -1.2, 1.2))
        target_pitch = float(np.clip(pitch, -0.5, 0.8))
    # 过渡区：线性插值
    else:
        blend = (dist - inner_radius) / (outer_radius - inner_radius)
        world_yaw = float(np.arctan2(direction[1], direction[0]))
        raw_yaw = world_yaw - waist_yaw
        while raw_yaw > np.pi:
            raw_yaw -= 2 * np.pi
        while raw_yaw < -np.pi:
            raw_yaw += 2 * np.pi
        horizontal_dist = float(np.sqrt(direction[0]**2 + direction[1]**2))
        pitch = float(-np.arctan2(direction[2], horizontal_dist))
        full_yaw = float(np.clip(raw_yaw, -1.2, 1.2))
        full_pitch = float(np.clip(pitch, -0.5, 0.8))
        target_yaw = full_yaw * blend
        target_pitch = full_pitch * blend
    
    # 平滑过渡
    yaw_diff = target_yaw - prev_yaw
    while yaw_diff > np.pi:
        yaw_diff -= 2 * np.pi
    while yaw_diff < -np.pi:
        yaw_diff += 2 * np.pi
    yaw = prev_yaw + yaw_diff * 0.1
    pitch = prev_pitch + (target_pitch - prev_pitch) * 0.1
    
    return yaw, pitch


# ==================== 主程序 ====================

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算关节索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    neck_dof = [int(i) for i in joint_idx["neck"]]
    waist_dof = [int(i) for i in joint_idx["waist"]]
    
    # 腰部和neck关节qpos索引
    waist_yaw_qpos = model.jnt_qposadr[model.joint("waist_yaw_joint").id]
    waist_roll_qpos = model.jnt_qposadr[model.joint("waist_roll_joint").id]
    waist_pitch_qpos = model.jnt_qposadr[model.joint("waist_pitch_joint").id]
    neck_yaw_qpos = model.jnt_qposadr[model.joint("neck_yaw_joint").id]
    neck_pitch_qpos = model.jnt_qposadr[model.joint("neck_pitch_joint").id]
    neck_roll_qpos = model.jnt_qposadr[model.joint("neck_roll_joint").id]
    
    # 创建IK任务
    tasks = [
        mink.FrameTask("pelvis", "body", position_cost=1e6, orientation_cost=1e6),
        mink.PostureTask(model, cost=1e-2),
    ]
    ee_tasks = {}
    for name, (link, _) in END_EFFECTORS.items():
        task = mink.FrameTask(link, "body", position_cost=2.0, orientation_cost=2.0)
        tasks.append(task)
        ee_tasks[name] = task
    
    # 手臂偏置任务
    arm_posture = mink.PostureTask(model, cost=ARM_BIAS_CONFIG['bias_cost'])
    bias_q = cfg.q.copy()
    bias_q[model.jnt_qposadr[model.joint("right_shoulder_roll_joint").id]] = -ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("right_elbow_joint").id]] = -ARM_BIAS_CONFIG['downward_bias']
    bias_q[model.jnt_qposadr[model.joint("left_shoulder_roll_joint").id]] = ARM_BIAS_CONFIG['outward_bias']
    bias_q[model.jnt_qposadr[model.joint("left_elbow_joint").id]] = ARM_BIAS_CONFIG['downward_bias']
    arm_posture.set_target(bias_q)
    tasks.append(arm_posture)
    
    # 碰撞限制（优化参数减少抖动）
    limits = [
        mink.ConfigurationLimit(model),
        mink.VelocityLimit(model),
        # mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, gain=0.1,
        #                              minimum_distance_from_collisions=0.05,
        #                              collision_detection_distance=0.15)
    ]
    
    # 复位状态
    reset_state = {"active": False, "alpha": 0.0, "start_q": None, "start_pos": {}, "start_quat": {}}
    
    def key_callback(keycode):
        if keycode == 259:
            reset_state["active"] = True
            reset_state["alpha"] = 0.0
            reset_state["start_q"] = cfg.q.copy()
            for name, mid in mocap_ids.items():
                reset_state["start_pos"][name] = data.mocap_pos[mid].copy()
                reset_state["start_quat"][name] = data.mocap_quat[mid].copy()
            print("[Reset] 开始复位...")
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False,
                                       key_callback=key_callback) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        cfg.update_from_keyframe("stand")
        tasks[0].set_target_from_configuration(cfg)
        tasks[1].set_target_from_configuration(cfg)
        
        mocap_ids = {}
        init_pos, init_quat = {}, {}
        for name, (link, mocap) in END_EFFECTORS.items():
            mid = model.body(mocap).mocapid[0]
            mocap_ids[name] = mid
            mink.move_mocap_to_frame(model, data, mocap, link, "body")
            init_pos[name] = data.mocap_pos[mid].copy()
            init_quat[name] = data.mocap_quat[mid].copy()
        
        left_mid, right_mid = mocap_ids["left_hand"], mocap_ids["right_hand"]
        init_q = cfg.q.copy()
        
        waist_init_pos = data.xpos[model.body("pelvis").id].copy()
        prev_neck_yaw = 0.0
        prev_neck_pitch = 0.0
        prev_waist_yaw = 0.0
        prev_target_yaw = 0.0
        
        rate = RateLimiter(frequency=100.0, warn=False)
        dt = rate.dt
        reset_duration = 1.5
        print_counter = 0
        
        while viewer.is_running():
            # 双手中心（用于yaw跟随）
            hands_center = (data.mocap_pos[left_mid] + data.mocap_pos[right_mid]) / 2.0
            cfg_w = WAIST_CONFIG
            
            # 计算左右手分别到躯干的水平距离
            left_diff = data.mocap_pos[left_mid] - waist_init_pos
            left_diff[2] = 0
            left_dist = float(np.linalg.norm(left_diff))
            
            right_diff = data.mocap_pos[right_mid] - waist_init_pos
            right_diff[2] = 0
            right_dist = float(np.linalg.norm(right_diff))
            
            # 用单手最大距离计算blend_factor（不叠加，避免双手权重叠加）
            max_hand_dist = max(left_dist, right_dist)
            
            # 安全范围配置
            waist_safe_zone_inner = 0.15
            waist_safe_zone_outer = 0.25
            
            # 计算blend_factor
            if max_hand_dist <= waist_safe_zone_inner:
                blend_factor = 0.0
            elif max_hand_dist >= waist_safe_zone_outer:
                blend_factor = 1.0
            else:
                blend_factor = (max_hand_dist - waist_safe_zone_inner) / (waist_safe_zone_outer - waist_safe_zone_inner)
                blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
            
            # 先计算YAW（用于后续局部坐标系计算）
            target_waist_yaw = compute_waist_yaw(hands_center, waist_init_pos, 
                                                 inner_radius=waist_safe_zone_inner, 
                                                 outer_radius=waist_safe_zone_outer)
            yaw_diff = target_waist_yaw - prev_waist_yaw
            while yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            while yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            waist_yaw = prev_waist_yaw + yaw_diff * 0.15
            
            # 用世界坐标系判断是否在身后（用于禁用yaw）
            left_forward_world = compute_local_forward_dist(data.mocap_pos[left_mid], waist_init_pos, 0.0)
            right_forward_world = compute_local_forward_dist(data.mocap_pos[right_mid], waist_init_pos, 0.0)
            is_backward = (left_forward_world < -0.2 and right_forward_world < -0.2)
            
            if is_backward:
                # 后仰模式：禁用yaw跟随，保持0位
                waist_yaw = 0.0
                prev_waist_yaw = 0.0
            else:
                # 正常模式：应用计算的yaw
                prev_waist_yaw = waist_yaw
            
            # 用当前yaw的局部坐标系计算前向距离（用于pitch补偿）
            left_forward = compute_local_forward_dist(data.mocap_pos[left_mid], waist_init_pos, waist_yaw)
            right_forward = compute_local_forward_dist(data.mocap_pos[right_mid], waist_init_pos, waist_yaw)
            if abs(left_forward) > abs(right_forward):
                max_forward_dist = left_forward
            else:
                max_forward_dist = right_forward
            
            # pitch补偿：使用局部坐标系前向距离
            target_pitch = compute_waist_compensation(max_forward_dist, cfg_w['arm_reach'], 
                                                      cfg_w['deadzone'], cfg_w['compensation_gain'])
            waist_pitch = target_pitch * blend_factor
            
            # neck look-at（使用局部坐标系，内外圈逻辑）
            head_pos = data.xpos[model.body("neck_pitch_link").id]
            neck_yaw, neck_pitch = compute_neck_angles(head_pos, hands_center, prev_neck_yaw, 
                                                        prev_neck_pitch, waist_yaw)
            prev_neck_yaw = neck_yaw
            prev_neck_pitch = neck_pitch
            
            # 复位
            if reset_state["active"]:
                reset_state["alpha"] += dt / reset_duration
                alpha = min(1.0, reset_state["alpha"])
                for name, mid in mocap_ids.items():
                    data.mocap_pos[mid] = (1 - alpha) * reset_state["start_pos"][name] + alpha * init_pos[name]
                    data.mocap_quat[mid] = slerp(reset_state["start_quat"][name], init_quat[name], alpha)
                cfg.update(reset_state["start_q"] * (1 - alpha) + init_q * alpha)
                for name in END_EFFECTORS:
                    ee_tasks[name].set_target_from_configuration(cfg)
                if alpha >= 1.0:
                    reset_state["active"] = False
                    prev_waist_yaw = 0.0
                    prev_neck_yaw = 0.0
                    prev_neck_pitch = 0.0
                    print("[Reset] 复位完成")
                mujoco.mj_forward(model, data)
                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()
                continue
            
            for name, mid in mocap_ids.items():
                ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
            
            constraints = [mink.DofFreezingTask(model, dof_indices=neck_dof + waist_dof)]
            try:
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=2e-1, limits=limits, constraints=constraints)
            except mink.exceptions.NoSolutionFound:
                vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.5, limits=limits, constraints=[])
            cfg.integrate_inplace(vel, dt)
            
            q = cfg.q.copy()
            q[waist_yaw_qpos] = waist_yaw
            q[waist_pitch_qpos] = waist_pitch
            q[waist_roll_qpos] = 0.0
            q[neck_yaw_qpos] = neck_yaw
            q[neck_pitch_qpos] = neck_pitch
            q[neck_roll_qpos] = 0.0
            cfg.update(q)
            
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            mujoco.mj_camlight(model, data)
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                print(f"[Waist] yaw={waist_yaw:.3f}, pitch={waist_pitch:.3f} | [Neck] yaw={neck_yaw:.3f}, pitch={neck_pitch:.3f}")
            
            viewer.sync()
            rate.sleep()