"""全身独立肢体IK控制 - 含重力补偿、肩膀奇异点检测、平滑reset"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_XML = Path(__file__).parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml"

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

END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target", ["left_arm"]),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target", ["right_arm"]),
    "left_foot": ("left_ankle_roll_link", "left_foot_target", ["left_leg"]),
    "right_foot": ("right_ankle_roll_link", "right_foot_target", ["right_leg"]),
    "neck": ("neck_pitch_link", "neck_target", ["waist", "neck"]),
}

CRITICAL_JOINTS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint"],
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"],
}

COLLISION_PAIRS = [
    (["left_hand_collision"], ["torso_collision"]),
    (["right_hand_collision"], ["torso_collision"]),
    (["left_forearm_collision"], ["torso_collision"]),
    (["right_forearm_collision"], ["torso_collision"]),
    (["left_upper_arm_collision"], ["torso_collision"]),
    (["right_upper_arm_collision"], ["torso_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_forearm_collision"], ["right_forearm_collision"]),
    (["left_upper_arm_collision"], ["right_upper_arm_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["head_collision"], ["left_forearm_collision", "right_forearm_collision"]),
    (["left_foot_collision"], ["right_foot_collision"]),
    (["left_shin_collision"], ["right_shin_collision"]),
    (["left_hip_collision"], ["right_hip_collision"]),
    (["left_hand_collision"], ["left_hip_collision", "left_shin_collision", "left_foot_collision"]),
    (["right_hand_collision"], ["right_hip_collision", "right_shin_collision", "right_foot_collision"]),
    (["left_hand_collision"], ["right_hip_collision", "right_shin_collision", "right_foot_collision"]),
    (["right_hand_collision"], ["left_hip_collision", "left_shin_collision", "left_foot_collision"]),
    (["head_collision"], ["left_foot_collision", "right_foot_collision"]),
    (["head_collision"], ["left_shin_collision", "right_shin_collision"]),
    (["left_shin_collision"], ["torso_collision"]),
    (["right_shin_collision"], ["torso_collision"]),
    (["left_foot_collision"], ["torso_collision"]),
    (["right_foot_collision"], ["torso_collision"]),
    (["left_foot_collision", "right_foot_collision"], ["floor"]),
    (["left_shin_collision", "right_shin_collision"], ["floor"]),
]

POS_JUMP_THRESHOLD = 0.5
RESET_INTERP_STEPS = 50


def check_limb_singularity(q, prev_q, critical_indices_dict, threshold=POS_JUMP_THRESHOLD):
    """检测关键关节奇异点，返回发生奇异点的肢体名称"""
    if prev_q is None:
        return None, None
    for limb_name, indices in critical_indices_dict.items():
        joint_jump = np.abs(q[indices] - prev_q[indices])
        max_idx = np.argmax(joint_jump)
        if joint_jump[max_idx] > threshold:
            return limb_name, (indices[max_idx], joint_jump[max_idx])
    return None, None


def interpolate_limb_reset(current_q, target_q, limb_indices, steps=RESET_INTERP_STEPS):
    """只对指定肢体从当前位置插值到目标位置"""
    interp_qs = []
    for i in range(1, steps + 1):
        q = current_q.copy()
        alpha = i / steps
        q[limb_indices] = current_q[limb_indices] + alpha * (target_q[limb_indices] - current_q[limb_indices])
        interp_qs.append(q)
    return interp_qs


def interpolate_full_reset(current_q, target_q, steps=RESET_INTERP_STEPS):
    """全身从当前位置插值到目标位置"""
    return [current_q + (i / steps) * (target_q - current_q) for i in range(1, steps + 1)]


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    critical_indices = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in CRITICAL_JOINTS.items()}
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap, _) in END_EFFECTORS.items()}
    ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
    reset_target_q = model.key_qpos[model.keyframe("stand").id].copy()
    
    # 创建任务和限制
    tasks = [
        mink.FrameTask("pelvis", "body", position_cost=5.0, orientation_cost=5.0),
        mink.PostureTask(model, cost=1e-2),
    ] + [mink.FrameTask(link, "body", position_cost=5.0, orientation_cost=5.0) 
         for link, _, _ in END_EFFECTORS.values()]
    
    ee_tasks = {name: tasks[i+2] for i, name in enumerate(END_EFFECTORS.keys())}
    limits = [mink.ConfigurationLimit(model), 
              mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, 0.01, 0.1)]
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 初始化
        cfg.update_from_keyframe("stand")
        tasks[0].set_target_from_configuration(cfg)
        tasks[1].set_target_from_configuration(cfg)
        for name, (link, mocap, _) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(model, data, mocap, link, "body")
            ee_tasks[name].set_target_from_configuration(cfg)
        
        prev_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        prev_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        init_mocap_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        init_mocap_quat = {name: data.mocap_quat[mid].copy() for name, mid in mocap_ids.items()}
        prev_q, reset_queue, reset_limb, print_counter = None, [], None, 0
        is_full_reset = False  # 标记是否为全局reset
        threshold_sq = 1e-6
        quat_threshold = 0.01
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 检测退格键触发全局reset（检测配置突变）
            if prev_q is not None and not reset_queue:
                q_diff = np.abs(cfg.q - prev_q)
                # 如果多个关节同时突变，说明是退格键reset
                large_jumps = np.sum(q_diff > POS_JUMP_THRESHOLD)
                if large_jumps > 3:  # 超过3个关节突变，触发全局reset
                    print(f"[INFO] 检测到全局reset（{large_jumps}个关节突变），执行平滑全局reset")
                    reset_queue = interpolate_full_reset(prev_q.copy(), reset_target_q)
                    is_full_reset = True
                    prev_q = None
                    continue
            
            # reset插值处理
            if reset_queue:
                cfg.update(reset_queue.pop(0))
                # 全局reset：更新所有mocap
                if is_full_reset:
                    for name, (link, mocap, _) in END_EFFECTORS.items():
                        mink.move_mocap_to_frame(model, data, mocap, link, "body")
                        ee_tasks[name].set_target_from_configuration(cfg)
                        mid = model.body(mocap).mocapid[0]
                        prev_pos[name] = data.mocap_pos[mid].copy()
                        prev_quat[name] = data.mocap_quat[mid].copy()
                # 单路reset：只更新受影响肢体的mocap
                else:
                    for name, (link, mocap, limbs) in END_EFFECTORS.items():
                        if reset_limb in limbs:
                            mink.move_mocap_to_frame(model, data, mocap, link, "body")
                            ee_tasks[name].set_target_from_configuration(cfg)
                            mid = model.body(mocap).mocapid[0]
                            prev_pos[name] = data.mocap_pos[mid].copy()
                            prev_quat[name] = data.mocap_quat[mid].copy()
                if not reset_queue:
                    reset_limb = None
                    is_full_reset = False
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:] = data.qfrc_bias[:]
                mujoco.mj_camlight(model, data)
                viewer.sync()
                rate.sleep()
                continue
            
            # 检测移动的mocap并设置任务目标（同时检测位置和姿态）
            active_limbs = []
            for name, mid in mocap_ids.items():
                pos_diff = data.mocap_pos[mid] - prev_pos[name]
                quat_diff = np.abs(data.mocap_quat[mid] - prev_quat[name])
                pos_changed = np.dot(pos_diff, pos_diff) > threshold_sq
                quat_changed = np.max(quat_diff) > quat_threshold
                
                if pos_changed or quat_changed:
                    active_limbs = ee_limbs[name]
                    ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
                else:
                    ee_tasks[name].set_target_from_configuration(cfg)
                
                prev_pos[name] = data.mocap_pos[mid].copy()
                prev_quat[name] = data.mocap_quat[mid].copy()
            
            # 动态调整posture cost
            if active_limbs:
                mask = np.zeros(model.nv, dtype=bool)
                for limb in active_limbs:
                    for idx in joint_idx[limb]:
                        mask[idx] = True
                tasks[1].cost[:] = np.where(mask, 1e-2, 1e4)
            else:
                tasks[1].cost[:] = 1e-2
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.1, limits=limits)
            if not active_limbs:
                vel[:] = 0.0
            elif active_limbs:
                mask = np.zeros(model.nv, dtype=bool)
                for limb in active_limbs:
                    for idx in joint_idx[limb]:
                        mask[idx] = True
                vel[~mask] = 0.0
            
            # 关键关节奇异点检测（肩膀+hip）
            singular_limb, info = check_limb_singularity(cfg.q, prev_q, critical_indices)
            if singular_limb is not None and info is not None:
                print(f"[WARN] {singular_limb}奇异点: joint_idx={info[0]} jump={info[1]:.3f}, 执行该路 reset")
                reset_queue = interpolate_limb_reset(cfg.q.copy(), reset_target_q, joint_idx[singular_limb])
                reset_limb = singular_limb
                prev_q = None
                continue
            
            prev_q = cfg.q.copy()
            cfg.integrate_inplace(vel, dt)
            
            # 重力补偿前馈
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                print(f"[Gravity Compensation] qfrc_applied:\n  {np.array2string(data.qfrc_bias[6:], precision=3, suppress_small=True)}")
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
