"""全身独立肢体IK控制 - 含重力补偿、零空间保持"""

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

# shoulder和hip关节用于零空间保持
NULLSPACE_JOINTS = {
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

RESET_INTERP_STEPS = 50


def interpolate_full_reset(current_q, target_q, steps=RESET_INTERP_STEPS):
    """全身从当前位置插值到目标位置"""
    return [current_q + (i / steps) * (target_q - current_q) for i in range(1, steps + 1)]


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    model, data = cfg.model, cfg.data
    
    # 预计算索引
    joint_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in JOINT_GROUPS.items()}
    nullspace_idx = {k: [model.jnt_dofadr[model.joint(j).id] for j in v] for k, v in NULLSPACE_JOINTS.items()}
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
        prev_q, reset_queue, print_counter = None, [], 0
        threshold_sq, quat_threshold = 1e-6, 0.01
        POS_JUMP_THRESHOLD = 0.5
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 检测退格键触发全局reset
            if prev_q is not None and not reset_queue:
                if np.sum(np.abs(cfg.q - prev_q) > POS_JUMP_THRESHOLD) > 3:
                    print(f"[INFO] 检测到全局reset，执行平滑reset")
                    reset_queue = interpolate_full_reset(prev_q.copy(), reset_target_q)
                    prev_q = None
                    continue
            
            # reset插值处理
            if reset_queue:
                cfg.update(reset_queue.pop(0))
                for name, (link, mocap, _) in END_EFFECTORS.items():
                    mink.move_mocap_to_frame(model, data, mocap, link, "body")
                    ee_tasks[name].set_target_from_configuration(cfg)
                    mid = model.body(mocap).mocapid[0]
                    prev_pos[name] = data.mocap_pos[mid].copy()
                    prev_quat[name] = data.mocap_quat[mid].copy()
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
            
            # 构建零空间约束：非活动肢体的shoulder/hip关节冻结
            freeze_dofs = []
            for limb, indices in nullspace_idx.items():
                if limb not in active_limbs:
                    freeze_dofs.extend(indices)
            constraints = [mink.DofFreezingTask(model, freeze_dofs)] if freeze_dofs else None
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.1, limits=limits, constraints=constraints)
            if not active_limbs:
                vel[:] = 0.0
            else:
                mask = np.zeros(model.nv, dtype=bool)
                for limb in active_limbs:
                    for idx in joint_idx[limb]:
                        mask[idx] = True
                vel[~mask] = 0.0
            
            prev_q = cfg.q.copy()
            cfg.integrate_inplace(vel, dt)
            
            # 重力补偿前馈
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]
            
            print_counter += 1
            if print_counter >= 200:
                print_counter = 0
                print(f"[Gravity Compensation] :\n  {np.array2string(data.qfrc_applied[6:], precision=3, suppress_small=True)}")
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()