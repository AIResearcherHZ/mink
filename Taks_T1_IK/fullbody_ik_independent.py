"""全身独立肢体IK控制 - 含重力补偿"""

from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

_XML = Path(__file__).parent / "assets" / "Taks_T1" / "scene_Taks_T1.xml"

# 关节分组 (肢体名 -> 关节名列表)
JOINT_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint"],
    "left_leg": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                 "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
    "neck": ["neck_yaw_joint", "neck_roll_joint", "neck_pitch_joint"],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
}

# 末端配置 (末端名 -> (link, mocap, 控制的肢体列表))
END_EFFECTORS = {
    "left_hand": ("left_wrist_pitch_link", "left_hand_target", ["left_arm"]),
    "right_hand": ("right_wrist_pitch_link", "right_hand_target", ["right_arm"]),
    "left_foot": ("left_ankle_roll_link", "left_foot_target", ["left_leg"]),
    "right_foot": ("right_ankle_roll_link", "right_foot_target", ["right_leg"]),
    "neck": ("neck_pitch_link", "neck_target", ["waist", "neck"]),
}

# 碰撞对
COLLISION_PAIRS = [
    (["left_hand_collision"], ["torso_collision"]),
    (["right_hand_collision"], ["torso_collision"]),
    (["left_hand_collision"], ["right_hand_collision"]),
    (["left_forearm_collision"], ["torso_collision"]),
    (["right_forearm_collision"], ["torso_collision"]),
    (["left_upper_arm_collision"], ["torso_collision"]),
    (["right_upper_arm_collision"], ["torso_collision"]),
    (["head_collision"], ["left_hand_collision", "right_hand_collision"]),
    (["left_foot_collision", "right_foot_collision"], ["floor"]),
    (["left_shin_collision", "right_shin_collision"], ["floor"]),
    (["left_hip_collision"], ["left_hand_collision"]),
    (["right_hip_collision"], ["right_hand_collision"]),
]


def get_dof_indices(model, joint_names):
    """获取关节DOF索引"""
    return [model.jnt_dofadr[model.joint(j).id] for j in joint_names]


def build_limb_mask(nv, joint_idx, active_limbs):
    """构建肢体掩码数组，活动肢体为True"""
    mask = np.zeros(nv, dtype=bool)
    for limb in active_limbs:
        for idx in joint_idx.get(limb, []):
            mask[idx] = True
    return mask


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    cfg = mink.Configuration(model)
    
    # 预计算关节索引
    joint_idx = {k: get_dof_indices(model, v) for k, v in JOINT_GROUPS.items()}
    all_limb_indices = set()
    for indices in joint_idx.values():
        all_limb_indices.update(indices)
    
    # 预分配cost数组
    high_cost = np.ones(model.nv) * 1e4
    
    # 创建任务
    pelvis_task = mink.FrameTask("pelvis", "body", position_cost=1e4, orientation_cost=1e4, lm_damping=1.0)
    posture_task = mink.PostureTask(model, cost=1e3)
    ee_tasks = {name: mink.FrameTask(link, "body", 
                position_cost=300 if "neck" in name else 500,
                orientation_cost=30 if "neck" in name else 50, lm_damping=1.0)
                for name, (link, _, _) in END_EFFECTORS.items()}
    
    tasks = [pelvis_task, posture_task] + list(ee_tasks.values())
    limits = [mink.ConfigurationLimit(model),
              mink.CollisionAvoidanceLimit(model, COLLISION_PAIRS, 0.01, 0.1)]
    
    # 预计算mocap ID
    mocap_ids = {name: model.body(mocap).mocapid[0] for name, (_, mocap, _) in END_EFFECTORS.items()}
    ee_limbs = {name: limbs for name, (_, _, limbs) in END_EFFECTORS.items()}
    
    model, data = cfg.model, cfg.data
    nv = model.nv
    
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        
        # 初始化姿态
        cfg.update_from_keyframe("stand")
        pelvis_task.set_target_from_configuration(cfg)
        posture_task.set_target_from_configuration(cfg)
        for name, (link, mocap, _) in END_EFFECTORS.items():
            mink.move_mocap_to_frame(model, data, mocap, link, "body")
            ee_tasks[name].set_target_from_configuration(cfg)
        
        # 缓存上一帧mocap位置
        prev_pos = {name: data.mocap_pos[mid].copy() for name, mid in mocap_ids.items()}
        threshold_sq = 0.001 ** 2  # 使用平方避免sqrt
        
        rate = RateLimiter(frequency=200.0, warn=False)
        dt = rate.dt
        
        while viewer.is_running():
            # 检测移动的mocap (用平方距离避免sqrt)
            active_limbs = []
            for name, mid in mocap_ids.items():
                diff = data.mocap_pos[mid] - prev_pos[name]
                if np.dot(diff, diff) > threshold_sq:
                    active_limbs = ee_limbs[name]
                    ee_tasks[name].set_target(mink.SE3.from_mocap_id(data, mid))
                else:
                    ee_tasks[name].set_target_from_configuration(cfg)
                prev_pos[name] = data.mocap_pos[mid].copy()
            
            # 更新posture cost
            if active_limbs:
                cost = high_cost.copy()
                for limb in active_limbs:
                    for idx in joint_idx[limb]:
                        cost[idx] = 0.001
                posture_task.set_cost(cost)
            else:
                posture_task.set_cost(high_cost)
            
            # 求解IK
            vel = mink.solve_ik(cfg, tasks, dt, "daqp", damping=0.1, limits=limits)
            
            # 非活动肢体速度清零
            if active_limbs:
                mask = build_limb_mask(nv, joint_idx, active_limbs)
                vel[~mask] = 0.0
            else:
                vel[:] = 0.0
            
            # 积分更新配置
            cfg.integrate_inplace(vel, dt)
            
            # 重力补偿: 计算当前姿态下的重力力矩并应用
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:] = data.qfrc_bias[:]  # qfrc_bias包含重力和科氏力
            
            mujoco.mj_camlight(model, data)
            viewer.sync()
            rate.sleep()
