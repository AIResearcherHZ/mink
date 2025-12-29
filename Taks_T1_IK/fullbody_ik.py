"""
Full-body IK for Taks_T1 humanoid robot.
Controls: 2 arm end-effectors, 2 leg end-effectors, 1 neck end-effector.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "assets" / "Taks_T1" / "scene_Taks_T1.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    # End-effector names
    hands = ["left_wrist_pitch_link", "right_wrist_pitch_link"]
    feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
    neck = "neck_pitch_link"

    # Define tasks
    tasks = [
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=100.0),
    ]

    # Hand tasks
    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="body",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    # Foot tasks
    foot_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="body",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        foot_tasks.append(task)
    tasks.extend(foot_tasks)

    # Neck task
    neck_task = mink.FrameTask(
        frame_name=neck,
        frame_type="body",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(neck_task)

    # Get mocap body IDs
    com_mid = model.body("com_target").mocapid[0]
    hand_mids = [model.body("left_hand_target").mocapid[0], model.body("right_hand_target").mocapid[0]]
    foot_mids = [model.body("left_foot_target").mocapid[0], model.body("right_foot_target").mocapid[0]]
    neck_mid = model.body("neck_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the stand keyframe
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective frames
        for i, hand in enumerate(hands):
            target_name = "left_hand_target" if i == 0 else "right_hand_target"
            mink.move_mocap_to_frame(model, data, target_name, hand, "body")
        for i, foot in enumerate(feet):
            target_name = "left_foot_target" if i == 0 else "right_foot_target"
            mink.move_mocap_to_frame(model, data, target_name, foot, "body")
        mink.move_mocap_to_frame(model, data, "neck_target", neck, "body")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets from mocap bodies
            com_task.set_target(data.mocap_pos[com_mid])
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(data, hand_mids[i]))
            for i, foot_task in enumerate(foot_tasks):
                foot_task.set_target(mink.SE3.from_mocap_id(data, foot_mids[i]))
            neck_task.set_target(mink.SE3.from_mocap_id(data, neck_mid))

            # Solve IK
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
