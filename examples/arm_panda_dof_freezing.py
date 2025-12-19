"""Example demonstrating DOF freezing with equality constraints.

This example alternates between frozen and unfrozen modes every 5 seconds.
When frozen (red target), joints 0-1 are locked. When unfrozen (green target),
all joints are free to move. The end-effector follows a circular trajectory
that requires base rotation, illustrating the impact of the constraints.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "franka_emika_panda" / "mjx_scene.xml"

# IK parameters.
SOLVER = "daqp"
POS_THRESHOLD = 5e-4
ORI_THRESHOLD = 5e-4
MAX_ITERS = 20

_RED_COLOR = (0.6, 0.3, 0.3, 0.2)
_GREEN_COLOR = (0.3, 0.6, 0.3, 0.2)


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model=model, cost=1e-1),
    ]

    # Constraint to freeze base rotation and shoulder lift.
    freeze_constraint = mink.DofFreezingTask(model=model, dof_indices=[0, 1])

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        center = data.mocap_pos[0].copy()

        # Wide circular trajectory that requires base rotation.
        radius = 0.25
        freq = 0.1
        toggle_period = 5.0  # Toggle constraints every 5 seconds.
        target_geom_id = model.geom("target").id

        local_time = 0.0
        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            dt = rate.dt
            local_time += dt

            # Toggle constraints every toggle_period seconds.
            phase = int(local_time / toggle_period) % 2
            constraints = [freeze_constraint] if phase == 0 else []

            # Circular trajectory around the robot.
            angle = 2 * np.pi * freq * local_time
            data.mocap_pos[0] = center + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), 0.0]
            )

            # Color the target based on constraint state.
            # Red = frozen, Green = free.
            with viewer.lock():
                model.geom_rgba[target_geom_id] = (
                    _RED_COLOR if constraints else _GREEN_COLOR
                )

            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            for _ in range(MAX_ITERS):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    dt,
                    SOLVER,
                    damping=1e-3,
                    constraints=constraints,
                )
                configuration.integrate_inplace(vel, dt)

                err = end_effector_task.compute_error(configuration)
                if (
                    np.linalg.norm(err[:3]) <= POS_THRESHOLD
                    and np.linalg.norm(err[3:]) <= ORI_THRESHOLD
                ):
                    break

            data.ctrl = configuration.q[:8]
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()
