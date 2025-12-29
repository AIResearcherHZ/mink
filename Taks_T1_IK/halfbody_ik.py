"""
Half-body IK for Semi_Taks_T1 robot.
Controls: 2 arm end-effectors, neck yaw_link, neck pitch_link.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "assets" / "Semi_Taks_T1" / "scene_Semi_Taks_T1.xml"


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    # End-effector names
    hands = ["left_wrist_pitch_link", "right_wrist_pitch_link"]
    neck_yaw = "neck_yaw_link"
    neck_pitch = "neck_pitch_link"

    # Define tasks
    tasks = [
        posture_task := mink.PostureTask(model, cost=1.0),
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

    # Neck yaw task
    neck_yaw_task = mink.FrameTask(
        frame_name=neck_yaw,
        frame_type="body",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(neck_yaw_task)

    # Neck pitch task
    neck_pitch_task = mink.FrameTask(
        frame_name=neck_pitch,
        frame_type="body",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(neck_pitch_task)

    # Get mocap body IDs
    hand_mids = [model.body("left_hand_target").mocapid[0], model.body("right_hand_target").mocapid[0]]
    neck_yaw_mid = model.body("neck_yaw_target").mocapid[0]
    neck_pitch_mid = model.body("neck_pitch_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe
        configuration.update_from_keyframe("home")
        posture_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective frames
        for i, hand in enumerate(hands):
            target_name = "left_hand_target" if i == 0 else "right_hand_target"
            mink.move_mocap_to_frame(model, data, target_name, hand, "body")
        mink.move_mocap_to_frame(model, data, "neck_yaw_target", neck_yaw, "body")
        mink.move_mocap_to_frame(model, data, "neck_pitch_target", neck_pitch, "body")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task targets from mocap bodies
            for i, hand_task in enumerate(hand_tasks):
                hand_task.set_target(mink.SE3.from_mocap_id(data, hand_mids[i]))
            neck_yaw_task.set_target(mink.SE3.from_mocap_id(data, neck_yaw_mid))
            neck_pitch_task.set_target(mink.SE3.from_mocap_id(data, neck_pitch_mid))

            # Solve IK
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            rate.sleep()
