Configuration

Configuration space of a robot model.

The Configuration class encapsulates a MuJoCo model and data, offering easy access to frame transforms and frame Jacobians. A frame refers to a coordinate system that can be attached to various parts of the robot, such as a body, geom, or site.

class mink.configuration.Configuration

Encapsulates a model and data for convenient access to kinematic quantities.

This class provides methods to access and update the kinematic quantities of a robot model, such as frame transforms and Jacobians. It performs forward kinematics at every time step, ensuring up-to-date information about the robot’s state.

Key functionalities include:

    Running forward kinematics to update the state.

    Checking configuration limits.

    Computing Jacobians for different frames.

    Computing the joint-space inertia matrix.

    Retrieving frame transforms relative to the world frame.

    Integrating velocities to update configurations.

update(q: ndarray | None = None) → None

    Run forward kinematics.

    Parameters:

        q (ndarray | None) – Optional configuration vector to override internal data.qpos with.
    Return type:

        None

update_from_keyframe(key_name: str) → None

    Update the configuration from a keyframe.

    Parameters:

        key_name (str) – The name of the keyframe.
    Return type:

        None

check_limits(tol: float = 1e-06, safety_break: bool = True) → None

    Check that the current configuration is within bounds.

    Parameters:

            tol (float) – Tolerance in [rad].

            safety_break (bool) – If True, stop execution and raise an exception if the current configuration is outside limits. If False, print a warning and continue execution.

    Raises:

        NotWithinConfigurationLimits – If the current configuration is outside the joint limits.
    Return type:

        None

get_frame_jacobian(frame_name: str, frame_type: str) → ndarray

Compute the Jacobian matrix of a frame velocity.

Denoting our frame by
and the world frame by , the Jacobian matrix is related to the body velocity

by:
Parameters:

        frame_name (str) – Name of the frame in the MJCF.

        frame_type (str) – Type of frame. Can be a geom, a body or a site.

Returns:

    Jacobian 

        of the frame.
    Return type:

        ndarray

get_transform_frame_to_world(frame_name: str, frame_type: str) → SE3

    Get the pose of a frame at the current configuration.

    Parameters:

            frame_name (str) – Name of the frame in the MJCF.

            frame_type (str) – Type of frame. Can be a geom, a body or a site.

    Returns:

        The pose of the frame in the world frame.
    Return type:

        SE3

get_transform(source_name: str, source_type: str, dest_name: str, dest_type: str) → SE3

    Get the pose of a frame with respect to another frame at the current configuration.

    Parameters:

            source_name (str) – Name of the frame in the MJCF.

            source_type (str) – Source type of frame. Can be a geom, a body or a site.

            dest_name (str) – Name of the frame to get the pose in.

            dest_type (str) – Dest type of frame. Can be a geom, a body or a site.

    Returns:

        The pose of source_name in dest_name.
    Return type:

        SE3

integrate(velocity: ndarray, dt: float) → ndarray

    Integrate a velocity starting from the current configuration.

    Parameters:

            velocity (ndarray) – The velocity in tangent space.

            dt (float) – Integration duration in [s].

    Returns:

        The new configuration after integration.
    Return type:

        ndarray

integrate_inplace(velocity: ndarray, dt: float) → None

    Integrate a velocity and update the current configuration inplace.

    Parameters:

            velocity (ndarray) – The velocity in tangent space.

            dt (float) – Integration duration in [s].

    Return type:

        None

get_inertia_matrix() → ndarray

Return the joint-space inertia matrix at the current configuration.

Returns:

    The joint-space inertia matrix 

        .
    Return type:

        ndarray

property q: ndarray

    The current configuration vector.

property nv: int

    The dimension of the tangent space.

property nq: int

    The dimension of the configuration space.

Tasks

All kinematic tasks derive from the Task base class.

class mink.tasks.task.Objective

Quadratic objective of the form

.

H: ndarray

Hessian matrix, of shape (
,

    ).

c: ndarray

Linear vector, of shape (

    ,).

value(x: ndarray) → float

        Returns the value of the objective at the input vector.

        Parameters:

            x (ndarray)
        Return type:

            float

class mink.tasks.task.BaseTask

Base class for all tasks.

abstract compute_qp_objective(configuration: Configuration) → Objective

Compute the matrix-vector pair

of the QP objective.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Pair 

            .
        Return type:

            Objective

class mink.tasks.task.Task

Abstract base class for kinematic tasks.

Subclasses must implement the configuration-dependent task error compute_error() and Jacobian compute_jacobian() functions.

The error function
is the quantity that the task aims to drive to zero (

is the dimension of the task). It appears in the first-order task dynamics:
The Jacobian matrix , with the dimension of the robot’s tangent space, is the derivative of the task error with respect to the configuration . The configuration displacement

is the output of inverse kinematics; we divide it by dt to get a commanded velocity.

In the first-order task dynamics, the error
is multiplied by the task gain

. This gain can be 1.0 for dead-beat control (i.e. converge as fast as possible), but might be unstable as it neglects our first-order approximation. Lower values cause slow down the task, similar to low-pass filtering.

abstract compute_error(configuration: Configuration) → ndarray

Compute the task error at the current configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task error vector 

        .
    Return type:

        ndarray

abstract compute_jacobian(configuration: Configuration) → ndarray

Compute the task Jacobian at the current configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task jacobian 

        .
    Return type:

        ndarray

compute_qp_objective(configuration: Configuration) → Objective

Compute the matrix-vector pair

of the QP objective.

This pair is such that the contribution of the task to the QP objective is:
The weight matrix

weights and normalizes task coordinates to the same unit. The unit of the overall contribution is [cost]^2.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Pair 

            .
        Return type:

            Objective

Kinematic Tasks
Frame task

Frame task implementation.

class mink.tasks.frame_task.FrameTask

Regulate the position and orientation of a frame of interest on the robot.

frame_name

    Name of the frame to regulate, typically the name of body, geom or site in the robot model.

frame_type

    The frame type: body, geom or site.

transform_target_to_world

    Target pose of the frame in the world frame.

    Type:

        mink.lie.se3.SE3 | None

Example:

frame_task = FrameTask(
    frame_name="target",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
)

# Update the target pose directly.
transform_target_to_world = SE3.from_translation(np.random.rand(3))
frame_task.set_target(transform_target_to_world)

# Or from the current configuration. This will automatically compute the
# target pose from the current configuration and update the task target.
frame_task.set_target_from_configuration(configuration)

set_target(transform_target_to_world: SE3) → None

    Set the target pose.

    Parameters:

        transform_target_to_world (SE3) – Transform from the task target frame to the world frame.
    Return type:

        None

set_target_from_configuration(configuration: Configuration) → None

Set the target pose from a given robot configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

        .
    Return type:

        None

compute_error(configuration: Configuration) → ndarray

Compute the frame task error.

This error is a twist
expressed in the local frame, i.e., it is a body twist. It is computed by taking the right-minus difference between the target pose and current frame pose

:
where denotes our frame, the target frame and

the inertial frame.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Frame task error vector 

        .
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the frame task Jacobian.

The derivation of the formula for this Jacobian is detailed in [FrameTaskJacobian].

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Frame task jacobian 

            .
        Return type:

            ndarray

Relative frame task

Relative frame task implementation.

class mink.tasks.relative_frame_task.RelativeFrameTask

Regulate the pose of a frame relative to another frame.

frame_name

    Name of the frame to regulate, typically the name of body, geom or site in the robot model.

frame_type

    The frame type: body, geom or site.

root_name

    Name of the frame the task is relative to.

root_type

    The root frame type: body, geom or site.

transform_target_to_root

    Target pose in the root frame.

    Type:

        mink.lie.se3.SE3 | None

set_target(transform_target_to_root: SE3) → None

    Set the target pose in the root frame.

    Parameters:

        transform_target_to_root (SE3) – Transform from the task target frame to the root frame.
    Return type:

        None

set_target_from_configuration(configuration: Configuration) → None

Set the target pose from a given robot configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

        .
    Return type:

        None

compute_error(configuration: Configuration) → ndarray

Compute the task error at the current configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task error vector 

        .
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the task Jacobian at the current configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task jacobian 

            .
        Return type:

            ndarray

Center of mass task

Center-of-mass task implementation.

class mink.tasks.com_task.ComTask

Regulate the center-of-mass (CoM) of a robot.

target_com

    Target position of the CoM.

    Type:

        numpy.ndarray | None

Example:

com_task = ComTask(model, cost=1.0)

# Update the target CoM directly.
com_desired = np.zeros(3)
com_task.set_target(com_desired)

# Or from a keyframe defined in the model.
configuration.update_from_keyframe("home")
com_task.set_target_from_configuration(configuration)

set_cost(cost: ArrayLike) → None

    Set the cost of the CoM task.

    Parameters:

        cost (ArrayLike) – A vector of shape (1,) (aka identical cost for all coordinates), or (3,) (aka different costs for each coordinate).
    Return type:

        None

set_target(target_com: ArrayLike) → None

    Set the target CoM position in the world frame.

    Parameters:

        target_com (ArrayLike) – A vector of shape (3,) representing the desired center-of-mass position in the world frame.
    Return type:

        None

set_target_from_configuration(configuration: Configuration) → None

Set the target CoM from a given robot configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

        .
    Return type:

        None

compute_error(configuration: Configuration) → ndarray

Compute the CoM task error.

The center of mass
for a collection of bodies

is the mass-weighted average of their individual centers of mass. After running forward kinematics, in particular after calling mj_comPos, MuJoCo stores the CoM of each subtree in data.subtree_com. This task uses the CoM of the subtree starting from body 1, which is the entire robot excluding the world body (body 0).

The task error
is the difference between the current CoM and the target CoM

:
Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Center-of-mass task error vector 

        .
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the Jacobian of the CoM task error

.

The Jacobian is the derivative of this error with respect to the generalized coordinates
. Since the target is constant, the Jacobian of the error simplifies to the Jacobian of the CoM position

:

MuJoCo’s mj_jacSubtreeCom function computes this Jacobian using the formula:
where is the total mass of the subtree, is the mass of body , is the position of the origin of body frame in the world frame,
is the Jacobian mapping joint velocities to the linear velocity of the origin of body frame , and the sum is over all bodies

in the specified subtree (body 1 and its descendants).

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Jacobian of the center-of-mass task error 

            .
        Return type:

            ndarray

Equality constraint task

Equality constraint task implementation.

class mink.tasks.equality_constraint_task.EqualityConstraintTask

Regulate equality constraints in a model.

Equality constraints are useful, among other things, for modeling “loop joints” such as four-bar linkages. In MuJoCo, there are several types of equality constraints, including:

    mjEQ_CONNECT: Connect two bodies at a point (ball joint).

    mjEQ_WELD: Fix relative pose of two bodies.

    mjEQ_JOINT: Couple the values of two scalar joints.

    mjEQ_TENDON: Couple the values of two tendons.

This task can regulate all equality constraints in the model or a specific subset identified by name or ID.

Note

MuJoCo computes the constraint residual and its Jacobian and stores them in data.efc_pos and data.efc_J (potentially in sparse format), respectively. The compute_error() and compute_jacobian() methods simply extract the rows corresponding to the active equality constraints specified for this task from data.efc_pos and data.efc_J. More information on MuJoCo’s constraint model can be found in [MuJoCoEqualityConstraints].

equalities

    ID or name of the equality constraints to regulate. If not provided, the task will regulate all equality constraints in the model.

cost

    Cost vector for the equality constraint task. Either a scalar, in which case the same cost is applied to all constraints, or a vector of shape (neq,), where neq is the number of equality constraints in the model.

Raises:

        InvalidConstraint – If a specified equality constraint name or ID is not found, or if the constraint is not active at the initial configuration.

        TaskDefinitionError – If no equality constraints are found or if cost parameters have invalid shape or values.

Example:

# Regulate all equality constraints with the same cost.
eq_task = EqualityConstraintTask(model, cost=1.0)

# Regulate specific equality constraints with different costs.
eq_task = EqualityConstraintTask(
    model,
    cost=[1.0, 0.5],
    equalities=["connect_right", "connect_left"]
)

set_cost(cost: ArrayLike) → None

    Set the cost vector for the equality constraint task.

    Parameters:

        cost (ArrayLike) – Cost vector for the equality constraint task.
    Return type:

        None

compute_error(configuration: Configuration) → ndarray

Compute the task error (constraint residual)

.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task error vector 

        for the active equality constraints.
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the task Jacobian (constraint Jacobian)

.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Task jacobian 

            for the active equality constraints.
        Return type:

            ndarray

DOF freezing task

DOF freezing task implementation.

class mink.tasks.dof_freezing_task.DofFreezingTask

Freeze specific degrees of freedom to zero velocity.

This task is typically used as an equality constraint to prevent specific joints from moving. It enforces zero velocity on the selected DOFs.

dof_indices

    List of DOF indices to freeze (zero velocity).

Example:

# Freeze specific DOFs by index.
dof_freezing_task = DofFreezingTask(
    model=model,
    dof_indices=[0, 1, 2]  # Freeze first 3 DOFs
)

# Use as equality constraint in IK solver.
v = solve_ik(
    configuration=configuration,
    tasks=[frame_task, com_task],
    constraints=[dof_freezing_task],  # Enforce exactly
    dt=dt,
    solver="proxqp",
)

# Freeze specific joints by name.
joint_names = ["shoulder_pan", "shoulder_lift"]
dof_indices = []
for joint_name in joint_names:
    joint_id = model.joint(joint_name).id
    dof_adr = model.jnt_dofadr[joint_id]
    dof_indices.append(dof_adr)

dof_freezing_task = DofFreezingTask(model=model, dof_indices=dof_indices)

compute_error(configuration: Configuration) → ndarray

Compute the DOF freezing task error.

The error is always zero since we’re constraining velocity, not position. When used as an equality constraint with zero error, this enforces Δq[dof] = 0 for each frozen DOF.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Zero vector of shape 
where

        is the number of frozen DOFs.
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the DOF freezing task Jacobian.

The Jacobian is a matrix with one row per frozen DOF, where each row is a standard basis vector (row of the identity matrix) selecting the corresponding DOF.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Jacobian matrix of shape 
where is the number of frozen DOFs and

            is the number of velocity DOFs.
        Return type:

            ndarray

Regularization Tasks
Posture task

Posture task implementation.

class mink.tasks.posture_task.PostureTask

Regulate joint angles towards a target posture.

Often used with a low priority in the task stack, this task acts like a regularizer, biasing the solution toward a specific joint configuration.

target_q

Target configuration
, of shape

    . Units are radians for revolute joints and meters for prismatic joints. Note that floating-base coordinates are not affected by this task but should be included in the target configuration.

    Type:

        numpy.ndarray | None

Example:

posture_task = PostureTask(model, cost=1e-3)

# Update the target posture directly.
q_desired = ...
posture_task.set_target(q_desired)

# Or from a keyframe defined in the model.
configuration.update_from_keyframe("home")
posture_task.set_target_from_configuration(configuration)

set_target(target_q: ArrayLike) → None

    Set the target posture.

    Parameters:

        target_q (ArrayLike) – A vector of shape (nq,) representing the desired joint configuration.
    Return type:

        None

set_target_from_configuration(configuration: Configuration) → None

Set the target posture by extracting it from the current configuration.

Parameters:

    configuration (Configuration) – Robot configuration 

        .
    Return type:

        None

compute_error(configuration: Configuration) → ndarray

Compute the posture task error.

The error is defined as:
Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Posture task error vector 

        .
    Return type:

        ndarray

compute_jacobian(configuration: Configuration) → ndarray

Compute the posture task Jacobian.

The task Jacobian is defined as:
Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Posture task jacobian 

            .
        Return type:

            ndarray

Damping task

Damping task implementation.

class mink.tasks.damping_task.DampingTask

L2-regularization on joint displacements (a.k.a. velocity damping).

This task, typically used with a low priority in the task stack, adds a Levenberg-Marquardt term to the quadratic program, favoring minimum-norm joint velocities in redundant or near-singular situations. Formally, it contributes the following term to the quadratic program:
where is the vector of joint displacements and is the i-th element of the cost parameter. The quadratic form uses
. A larger reduces motion in DoF

. With no other active tasks, the robot remains at rest. Unlike the damping parameter in solve_ik(), which is uniformly applied to all DoFs, this task does not affect the floating-base coordinates.

Note

This task does not favor a particular posture, only small instantaneous motion. If you need a posture bias, use PostureTask instead.

Example:

# Uniform damping across all degrees of freedom.
damping_task = DampingTask(model, cost=1.0)

# Custom damping.
cost = np.zeros(model.nv)
cost[:3] = 1.0  # High damping for the first 3 joints.
cost[3:] = 0.1  # Lower damping for the remaining joints.
damping_task = DampingTask(model, cost)

compute_error(configuration: Configuration) → ndarray

Compute the damping task error.

The damping task does not chase a reference; its desired joint velocity is identically zero, so the task error is always:
Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Zero vector of length 

            .
        Return type:

            ndarray

Kinetic energy regularization task

Kinetic energy regularization task implementation.

class mink.tasks.kinetic_energy_regularization_task.KineticEnergyRegularizationTask

Kinetic-energy regularization.

This task, often used with a low priority in the task stack, penalizes the system’s kinetic energy. Formally, it contributes the following term to the quadratic program:
where is the vector of joint displacements, is the joint-space inertia matrix, is the scalar strength of the regularization, and

is the integration timestep.

Note

This task can be seen as an inertia-weighted version of the DampingTask. Degrees of freedom with higher inertia will move less for the same cost.

Warning

The integration timestep

must be set via set_dt() before use. This ensures the cost is expressed in units of energy (Joules).

Example:

task = KineticEnergyRegularizationTask(cost=1e-4)
task.set_dt(0.02)

set_dt(dt: float) → None

    Set the integration timestep.

    Parameters:

        dt (float) – Integration timestep in [s].
    Return type:

        None

compute_qp_objective(configuration: Configuration) → Objective

Compute the matrix-vector pair

of the QP objective.

Parameters:

    configuration (Configuration) – Robot configuration 

    .
Returns:

    Pair 

    .
Return type:

    Objective

Limits
Definition

All kinematic limits derive from the Limit base class.

class mink.limits.limit.Constraint

Linear inequality constraint of the form

.

Inactive if G and h are None.

G: ndarray | None

    Alias for field number 0

h: ndarray | None

    Alias for field number 1

property inactive: bool

        Returns True if the constraint is inactive.

class mink.limits.limit.Limit

Abstract base class for kinematic limits.

Subclasses must implement the compute_qp_inequalities() method which takes in the current robot configuration and integration time step and returns an instance of Constraint.

abstract compute_qp_inequalities(configuration: Configuration, dt: float) → Constraint

Compute limit as linearized QP inequalities of the form:
where is the robot’s configuration and is the displacement in the tangent space at

.

Parameters:

        configuration (Configuration) – Robot configuration 

        .

        dt (float) – Integration time step in [s].

Returns:

    Pair 

            .
        Return type:

            Constraint

Configuration limits

Joint position limit.

class mink.limits.configuration_limit.ConfigurationLimit

Inequality constraint on joint positions in a robot model.

Floating base joints are ignored.

compute_qp_inequalities(configuration: Configuration, dt: float) → Constraint

Compute the configuration-dependent joint position limits.

The limits are defined as:
where is the robot’s configuration and is the displacement in the tangent space at

.

Parameters:

        configuration (Configuration) – Robot configuration 

        .

        dt (float) – Integration timestep in [s].

Returns:

    Pair 
representing the inequality constraint as

            , or None if there is no limit.
        Return type:

            Constraint

Velocity limits

Joint velocity limit.

class mink.limits.velocity_limit.VelocityLimit

Inequality constraint on joint velocities in a robot model.

Floating base joints are ignored.

indices

    Tangent indices corresponding to velocity-limited joints. Shape (nb,).

    Type:

        numpy.ndarray

limit

    Maximum allowed velocity magnitude for velocity-limited joints, in [m]/[s] for slide joints and [rad]/[s] for hinge joints. Shape (nb,).

    Type:

        numpy.ndarray

projection_matrix

    Projection from tangent space to subspace with velocity-limited joints. Shape (nb, nv) where nb is the dimension of the velocity-limited subspace and nv is the dimension of the tangent space.

    Type:

        numpy.ndarray | None

compute_qp_inequalities(configuration: Configuration, dt: float) → Constraint

Compute the configuration-dependent joint velocity limits.

The limits are defined as:
where is the robot’s velocity limit vector and is the displacement in the tangent space at

.

Parameters:

        configuration (Configuration) – Robot configuration 

        .

        dt (float) – Integration timestep in [s].

Returns:

    Pair 
representing the inequality constraint as

            , or None if there is no limit. G has shape (2nb, nv) and h has shape (2nb,).
        Return type:

            Constraint

Collision avoidance limits

Collision avoidance limit.

class mink.limits.collision_avoidance_limit.Contact

Struct to store contact information between two geoms.

dist

    Smallest signed distance between geom1 and geom2. If no collision of distance smaller than distmax is found, this value is equal to distmax [1].

    Type:

        float

fromto

    Segment connecting the closest points on geom1 and geom2. The first three elements are the coordinates of the closest point on geom1, and the last three elements are the coordinates of the closest point on geom2.

    Type:

        numpy.ndarray

geom1

    ID of geom1.

    Type:

        int

geom2

    ID of geom2.

    Type:

        int

distmax

    Maximum distance between geom1 and geom2.

    Type:

        float

References

[1] MuJoCo API documentation. mj_geomDistance function.

    https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html

property normal: ndarray

    Contact normal pointing from geom1 to geom2.

property inactive: bool

        Returns True if no distance smaller than distmax is detected between geom1 and geom2.

class mink.limits.collision_avoidance_limit.CollisionAvoidanceLimit

Inequality constraint on the normal velocity between geom pairs.

This constraint prevents collisions by limiting how fast geometries can approach each other along their contact normal. The velocity bound is distance-dependent:
where is the normal approach velocity, is the current distance, is the gain, is the bound relaxation, is the minimum distance, and

is the detection distance.

The gain parameter controls how conservatively geoms approach each other, with smaller values being safer but potentially slower.

References

    F. Kanehiro, F. Lamiraux, O. Kanoun, E. Yoshida, J.P. Laumond, “A Local Collision Avoidance Method for Non-strictly Convex Polyhedra”, Robotics: Science and Systems (2008).

    C. Fang, A. Rocchi, E. M. Hoffman, N. G. Tsagarakis and D. G. Caldwell, “Efficient self-collision avoidance based on focus of interest for humanoid robots”, IEEE-RAS International Conference on Humanoid Robots (2015).

model

    MuJoCo model.

geom_pairs

    Set of collision pairs in which to perform active collision avoidance. A collision pair is defined as a pair of geom groups. A geom group is a set of geom names. For each geom pair, the solver will attempt to compute joint velocities that avoid collisions between every geom in the first geom group with every geom in the second geom group. Self collision is achieved by adding a collision pair with the same geom group in both pair fields.

gain

    Gain factor in (0, 1] that determines how fast the geoms are allowed to move towards each other at each iteration. Smaller values are safer but may make the geoms move slower towards each other.

minimum_distance_from_collisions

    The minimum distance to leave between any two geoms. A negative distance allows the geoms to penetrate by the specified amount.

collision_detection_distance

    The distance between two geoms at which the active collision avoidance limit will be active. A large value will cause collisions to be detected early, but may incur high computational cost. A negative value will cause the geoms to be detected only after they penetrate by the specified amount.

bound_relaxation

    An offset on the upper bound of each collision avoidance constraint.

compute_qp_inequalities(configuration: Configuration, dt: float) → Constraint

Compute limit as linearized QP inequalities of the form:
where is the robot’s configuration and is the displacement in the tangent space at

.

Parameters:

        configuration (Configuration) – Robot configuration 

        .

        dt (float) – Integration time step in [s].

Returns:

    Pair 

    .
Return type:

    Constraint

Inverse kinematics
mink.solve_ik.solve_ik(configuration: Configuration, tasks: Sequence[BaseTask], dt: float, solver: str, damping: float = 1e-12, safety_break: bool = False, limits: Sequence[Limit] | None = None, constraints: Sequence[Task] | None = None, **kwargs) → ndarray

Solve the differential inverse kinematics problem.

Computes a velocity tangent to the current robot configuration. The computed velocity satisfies at (weighted) best the set of provided kinematic tasks.

Parameters:

        configuration (Configuration) – Robot configuration.

        tasks (Sequence[BaseTask]) – List of kinematic tasks.

        dt (float) – Integration timestep in [s].

        solver (str) – Backend quadratic programming (QP) solver.

        damping (float) – Levenberg-Marquardt damping applied to all tasks. Higher values improve numerical stability but slow down task convergence. This value applies to all dofs, including floating-base coordinates.

        safety_break (bool) – If True, stop execution and raise an exception if the current configuration is outside limits. If False, print a warning and continue execution.

        limits (Sequence[Limit] | None) – List of limits to enforce. Set to empty list to disable. If None, defaults to a configuration limit.

        constraints (Sequence[Task] | None) – List of tasks to enforce as equality constraints. These tasks will be satisfied exactly rather than in a least-squares sense.

        kwargs – Keyword arguments to forward to the backend QP solver.

Raises:

        NotWithinConfigurationLimits – If the current configuration is outside the joint limits and safety_break is True.

        NoSolutionFound – If the QP solver fails to find a solution.

Returns:

    Velocity 

        in tangent space.
    Return type:

        ndarray

mink.solve_ik.build_ik(configuration: Configuration, tasks: Sequence[BaseTask], dt: float, damping: float = 1e-12, limits: Sequence[Limit] | None = None, constraints: Sequence[Task] | None = None) → Problem

Build the quadratic program given the current configuration and tasks.

The quadratic program is defined as:
where

is the velocity in tangent space.

Parameters:

        configuration (Configuration) – Robot configuration.

        tasks (Sequence[BaseTask]) – List of kinematic tasks.

        dt (float) – Integration timestep in [s].

        damping (float) – Levenberg-Marquardt damping. Higher values improve numerical stability but slow down task convergence. This value applies to all dofs, including floating-base coordinates.

        limits (Sequence[Limit] | None) – List of limits to enforce. Set to empty list to disable. If None, defaults to a configuration limit.

        constraints (Sequence[Task] | None) – List of tasks to enforce as equality constraints. These tasks will be satisfied exactly rather than in a least-squares sense.

Returns:

    Quadratic program of the inverse kinematics problem.
Return type:

    Problem


Utilities
mink.utils.move_mocap_to_frame(model: MjModel, data: MjData, mocap_name: str, frame_name: str, frame_type: str) → None

    Initialize mocap body pose at a desired frame.

    Parameters:

            model (MjModel) – Mujoco model.

            data (MjData) – Mujoco data.

            mocap_name (str) – The name of the mocap body.

            frame_name (str) – The desired frame name.

            frame_type (str) – The desired frame type. Can be “body”, “geom” or “site”.

    Return type:

        None

mink.utils.get_freejoint_dims(model: MjModel) → tuple[list[int], list[int]]

    Get all floating joint configuration and tangent indices.

    Parameters:

        model (MjModel) – Mujoco model.
    Returns:

        A (q_ids, v_ids) pair containing all floating joint indices in the configuration and tangent spaces respectively.
    Return type:

        tuple[list[int], list[int]]

mink.utils.custom_configuration_vector(model: MjModel, key_name: str | None = None, **kwargs) → ndarray

    Generate a configuration vector where named joints have specific values.

    Parameters:

            model (MjModel) – Mujoco model.

            key_name (str | None) – Optional keyframe name to initialize the configuration vector from. Otherwise, the default pose qpos0 is used.

            kwargs – Custom values for joint coordinates.

    Returns:

        Configuration vector where named joints have the values specified in

            keyword arguments, and other joints have their neutral value or value defined in the keyframe if provided.

    Return type:

        ndarray

mink.utils.get_body_body_ids(model: MjModel, body_id: int) → list[int]

    Get immediate children bodies belonging to a given body.

    Parameters:

            model (MjModel) – Mujoco model.

            body_id (int) – ID of body.

    Returns:

        A List containing all child body ids.
    Return type:

        list[int]

mink.utils.get_subtree_body_ids(model: MjModel, body_id: int) → list[int]

    Get all bodies belonging to subtree starting at a given body.

    Parameters:

            model (MjModel) – Mujoco model.

            body_id (int) – ID of body where subtree starts.

    Returns:

        A List containing all subtree body ids.
    Return type:

        list[int]

mink.utils.get_body_geom_ids(model: MjModel, body_id: int) → list[int]

    Get immediate geoms belonging to a given body.

    Here, immediate geoms are those directly attached to the body and not its descendants.

    Parameters:

            model (MjModel) – Mujoco model.

            body_id (int) – ID of body.

    Returns:

        A list containing all body geom ids.
    Return type:

        list[int]

mink.utils.get_subtree_geom_ids(model: MjModel, body_id: int) → list[int]

    Get all geoms belonging to subtree starting at a given body.

    Here, a subtree is defined as the kinematic tree starting at the body and including all its descendants.

    Parameters:

            model (MjModel) – Mujoco model.

            body_id (int) – ID of body where subtree starts.

    Returns:

        A list containing all subtree geom ids.
    Return type:

        list[int]


Lie

MuJoCo does not currently offer a native Lie group interface for rigid body transforms, though it does have a collection of functions for manipulating quaternions and rotation matrices. The goal of this library is to provide this unified interface. Whenever possible, the underlying lie operation leverages the corresponding MuJoCo function. For example, from_matrix() uses mujoco.mju_mat2Quat under the hood.

This library is heavily ported from jaxlie, swapping out JAX for Numpy and adding a few additional features.
MatrixLieGroup
class mink.lie.base.MatrixLieGroup

Interface definition for matrix Lie groups.

matrix_dim

    Dimension of square matrix output.

    Type:

        int

parameters_dim

    Dimension of underlying parameters.

    Type:

        int

tangent_dim

    Dimension of tangent space.

    Type:

        int

space_dim

    Dimension of coordinates that can be transformed.

    Type:

        int

abstract classmethod identity() → Self

    Returns identity element.

    Return type:

        Self

abstract classmethod from_matrix(matrix: ndarray) → Self

    Get group member from matrix representation.

    Parameters:

        matrix (ndarray)
    Return type:

        Self

abstract classmethod sample_uniform() → Self

    Draw a uniform sample from the group.

    Return type:

        Self

abstract as_matrix() → ndarray

    Get transformation as a matrix.

    Return type:

        ndarray

abstract parameters() → ndarray

    Get underlying representation.

    Return type:

        ndarray

abstract apply(target: ndarray) → ndarray

    Applies group action to a point.

    Parameters:

        target (ndarray)
    Return type:

        ndarray

abstract multiply(other: Self) → Self

    Composes this transformation with another.

    Parameters:

        other (Self)
    Return type:

        Self

abstract classmethod exp(tangent: ndarray) → Self

    Computes expm(wedge(tangent)).

    Parameters:

        tangent (ndarray)
    Return type:

        Self

abstract log() → ndarray

    Computes vee(logm(transformation matrix)).

    Return type:

        ndarray

abstract adjoint() → ndarray

    Computes the adjoint.

    Return type:

        ndarray

abstract inverse() → Self

    Computes the inverse of the transform.

    Return type:

        Self

abstract normalize() → Self

    Normalize/projects values and returns.

    Return type:

        Self

interpolate(other: Self, alpha: float = 0.5) → Self

    Interpolate between two matrix Lie groups.

    Parameters:

            other (Self) – The other Lie group, which serves as the end point of interpolation.

            alpha (float) – The fraction of interpolation between [self, other]. This must be within [0.0, 1.0]. 0.0 = self, 1.0 = other.

    Returns:

        The interpolated matrix Lie group.
    Return type:

        Self

plus(other: ndarray) → Self

    Alias for rplus.

    Parameters:

        other (ndarray)
    Return type:

        Self

minus(other: Self) → ndarray

        Alias for rminus.

        Parameters:

            other (Self)
        Return type:

            ndarray

SO3
class mink.lie.so3.SO3

Bases: MatrixLieGroup

Special orthogonal group for 3D rotations.

Internal parameterization is (qw, qx, qy, qz). Tangent parameterization is (omega_x, omega_y, omega_z).

parameters() → ndarray

    Get underlying representation.

    Return type:

        ndarray

classmethod from_matrix(matrix: ndarray) → SO3

    Get group member from matrix representation.

    Parameters:

        matrix (ndarray)
    Return type:

        SO3

classmethod identity() → SO3

    Returns identity element.

    Return type:

        SO3

classmethod sample_uniform() → SO3

    Draw a uniform sample from the group.

    Return type:

        SO3

as_matrix() → ndarray

    Get transformation as a matrix.

    Return type:

        ndarray

inverse() → SO3

    Computes the inverse of the transform.

    Return type:

        SO3

normalize() → SO3

    Normalize/projects values and returns.

    Return type:

        SO3

apply(target: ndarray) → ndarray

    Applies group action to a point.

    Parameters:

        target (ndarray)
    Return type:

        ndarray

multiply(other: SO3) → SO3

    Composes this transformation with another.

    Parameters:

        other (SO3)
    Return type:

        SO3

classmethod exp(tangent: ndarray) → SO3

    Computes expm(wedge(tangent)).

    Parameters:

        tangent (ndarray)
    Return type:

        SO3

log() → ndarray

    Computes vee(logm(transformation matrix)).

    Return type:

        ndarray

adjoint() → ndarray

    Computes the adjoint.

    Return type:

        ndarray

clamp(roll_radians: tuple[float, float] = (-inf, inf), pitch_radians: tuple[float, float] = (-inf, inf), yaw_radians: tuple[float, float] = (-inf, inf)) → SO3

        Clamp a SO3 within RPY limits.

        Parameters:

                roll_radians (tuple[float, float]) – The (lower, upper) limits for the roll.

                pitch_radians (tuple[float, float]) – The (lower, upper) limits for the pitch.

                yaw_radians (tuple[float, float]) – The (lower, upper) limits for the yaw.

        Returns:

            A SO3 within the RPY limits.
        Return type:

            SO3

SE3
class mink.lie.se3.SE3

Bases: MatrixLieGroup

Special Euclidean group for proper rigid transforms in 3D.

Internal parameterization is (qw, qx, qy, qz, x, y, z). Tangent parameterization is (vx, vy, vz, omega_x, omega_y, omega_z).

parameters() → ndarray

    Get underlying representation.

    Return type:

        ndarray

classmethod identity() → SE3

    Returns identity element.

    Return type:

        SE3

classmethod from_matrix(matrix: ndarray) → SE3

    Get group member from matrix representation.

    Parameters:

        matrix (ndarray)
    Return type:

        SE3

classmethod sample_uniform() → SE3

    Draw a uniform sample from the group.

    Return type:

        SE3

as_matrix() → ndarray

    Get transformation as a matrix.

    Return type:

        ndarray

classmethod exp(tangent: ndarray) → SE3

    Computes expm(wedge(tangent)).

    Parameters:

        tangent (ndarray)
    Return type:

        SE3

inverse() → SE3

    Computes the inverse of the transform.

    Return type:

        SE3

normalize() → SE3

    Normalize/projects values and returns.

    Return type:

        SE3

apply(target: ndarray) → ndarray

    Applies group action to a point.

    Parameters:

        target (ndarray)
    Return type:

        ndarray

multiply(other: SE3) → SE3

    Composes this transformation with another.

    Parameters:

        other (SE3)
    Return type:

        SE3

log() → ndarray

    Computes vee(logm(transformation matrix)).

    Return type:

        ndarray

adjoint() → ndarray

    Computes the adjoint.

    Return type:

        ndarray

clamp(x_translation: tuple[float, float] = (-inf, inf), y_translation: tuple[float, float] = (-inf, inf), z_translation: tuple[float, float] = (-inf, inf), roll_radians: tuple[float, float] = (-inf, inf), pitch_radians: tuple[float, float] = (-inf, inf), yaw_radians: tuple[float, float] = (-inf, inf)) → SE3

    Clamp a SE3 within translation and RPY limits.

    Parameters:

            x_translation (tuple[float, float]) – The (lower, upper) limits for translation along the x axis.

            y_translation (tuple[float, float]) – The (lower, upper) limits for translation along the y axis.

            z_translation (tuple[float, float]) – The (lower, upper) limits for translation along the z axis.

            roll_radians (tuple[float, float]) – The (lower, upper) limits for the roll.

            pitch_radians (tuple[float, float]) – The (lower, upper) limits for the pitch.

            yaw_radians (tuple[float, float]) – The (lower, upper) limits for the yaw.

    Returns:

        A SE3 within the translation and RPY limits.
    Return type:

        SE3

