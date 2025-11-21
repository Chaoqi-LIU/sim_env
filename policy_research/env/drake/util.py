from tqdm import tqdm
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    KinematicTrajectoryOptimization,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PositionConstraint,
    RigidTransform,
    Role,
    Solve,
    StartMeshcat,
    RevoluteJoint,
    Trajectory,
    Context,
    MultibodyPlant,
    InverseKinematics,
    PrismaticJoint,
)
from typing import Union, List


def discretize_trajectory(
    trajectory: Trajectory,
    time_step: float = 0.05,
):
    timestamps = np.append(
        np.arange(trajectory.start_time(), trajectory.end_time(), time_step),
        trajectory.end_time(),
    )
    positions = np.array([trajectory.value(t) for t in timestamps])
    return positions, timestamps


def add_franka(
    plant, 
    q0=[0, 0, 0, -1.57, 0, 1.57, 0.79, 0.04, 0.04], 
    model_name='franka', 
    xyz=[0, 0, 0],
):
    parser = Parser(plant)
    franka = parser.AddModelsFromUrl(
        f"package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
    )[0]
    plant.RenameModelInstance(franka, model_name)
    X_FM = RigidTransform(xyz)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", franka), X_FM)

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(franka):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
        elif isinstance(joint, PrismaticJoint):
            joint.set_default_translation(q0[index])
            index += 1
    return franka


def add_iiwa(
    plant, 
    q0=[0.0, 0.1, 0, -1.2, 0, 1.6, 0],
    model_name='iiwa', 
    xyz=[0, 0, 0]
):
    parser = Parser(plant)
    iiwa = parser.AddModelsFromUrl(
        f"package://drake_models/iiwa_description/sdf/iiwa14_no_collision.sdf"
    )[0]
    plant.RenameModelInstance(iiwa, model_name)
    X_FM = RigidTransform(xyz)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0", iiwa), X_FM)

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1
    return iiwa


def add_robot(
    env: str,
    *args,
    **kwargs,
):
    if env == 'franka':
        return add_franka(*args, **kwargs)
    elif env == 'iiwa':
        return add_iiwa(*args, **kwargs)
    else:
        raise ValueError(f"Unknown env: {env}")
    

def gripper_frame_name(env: str):
    if env == 'franka':
        return "panda_hand"
    elif env == 'iiwa':
        return "iiwa_link_7"
    else:
        raise ValueError(f"Unknown env: {env}")


def export_animation_html(meshcat, file):
    html = meshcat.StaticHtml()
    with open(file, "w") as f:
        f.write(html)


def publish_position_trajectory(
    trajectory: Union[Trajectory, np.ndarray],
    root_context: Context,
    plant: MultibodyPlant,
    visualizer: MeshcatVisualizer,
    time_step: float = 0.05,
):
    """
    Publishes an animation to Meshcat of a MultibodyPlant using a trajectory of the plant positions.

    Args:
        trajectory: A Trajectory instance.
        root_context: The root context of the diagram containing plant.
        plant: A MultibodyPlant instance.
        visualizer: A MeshcatVisualizer instance.
        time_step: The time step between published frames.
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    if isinstance(trajectory, Trajectory):
        positions, timestamps = discretize_trajectory(trajectory, time_step)
    else:
        positions = trajectory
        timestamps = np.arange(0, positions.shape[0] * time_step, time_step)

    for pos, t in zip(positions, timestamps):
        root_context.SetTime(t)
        plant.SetPositions(plant_context, pos)
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def visualize_position_trajectories(
    trajectories: Union[List[Trajectory], List[np.ndarray]],
    env: str,
    time_step: float = 0.05,
    scene_separation: float = 1.0,      # separation between each scene in the y direction
    demo_save_path: str = None,
):
    # Meshcat prep.
    global meshcat 
    if "meshcat" in globals():
        meshcat.Delete()
    else:
        meshcat = StartMeshcat()
        print("Starting a new Meshcat server.")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    for i in range(len(trajectories)):
        robot = add_robot(env, plant=plant, model_name=f'{env}_{i}', xyz=[0, i * scene_separation, 0])
    parser = Parser(plant)
    plant.Finalize()

    # add the visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration),
    )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    # construct the composed trajectory
    if isinstance(trajectories[0], Trajectory):
        trajectories = [
            discretize_trajectory(traj, time_step)[0][:, :, 0]
            for traj in trajectories
        ]
    max_traj_len = max([len(traj) for traj in trajectories])
    for i in range(len(trajectories)):
        traj = trajectories[i]
        if (to_pad := max_traj_len - len(traj)) > 0:    # pad with last position
            traj = np.concatenate([
                traj, 
                np.repeat(traj[-1][None, :], to_pad, axis=0)
            ], axis=0)
            trajectories[i] = traj
    composed_trajectory = np.concatenate(trajectories, axis=-1)

    # visualize the trajectories
    publish_position_trajectory(composed_trajectory, context, plant, visualizer, time_step)

    # export visualization to html
    if demo_save_path:
        export_animation_html(meshcat, demo_save_path)


def inverse_kinematics(
    env: str,
    desired_translation: np.ndarray,
    tolerance: float = 1e-3,
    verbose: bool = False,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    robot = add_robot(env, plant=plant)
    parser = Parser(plant)
    plant.Finalize()
    diagram = builder.Build()
    root_context = diagram.CreateDefaultContext()
    ik_context = plant.GetMyContextFromRoot(root_context)
    gripper_frame = plant.GetFrameByName(gripper_frame_name(env), robot)

    ik = InverseKinematics(plant, ik_context)
    ik.prog().AddBoundingBoxConstraint(
        plant.GetPositionLowerLimits(), 
        plant.GetPositionUpperLimits(), 
        ik.q()
    )
    ik.AddPositionConstraint(
        gripper_frame,
        [0, 0, 0],
        plant.world_frame(),
        desired_translation - tolerance,
        desired_translation + tolerance,
    )
    result = Solve(ik.prog())
    if not result.is_success():
        if verbose:
            print(f"IK failed. Target: {desired_translation}")
        return None
    return result.GetSolution(ik.q())


def sample_feasible_waypoints(
    env: str,
    num_waypoints: int,
    verbose: bool = False,
    tqdm_bar: bool = False,
):
    lb = np.array([-1, -1, 0.2])
    ub = np.array([1, 1, 0.8])

    sampled_waypoints = []
    if tqdm_bar:
        pbar = tqdm(total=num_waypoints, desc=f"Sampling feasible waypoints for drake-{env}")
    while len(sampled_waypoints) < num_waypoints:
        waypoint = np.random.uniform(lb, ub)
        if inverse_kinematics(env, waypoint, verbose=verbose) is not None:
            sampled_waypoints.append(waypoint)
            if tqdm_bar:
                pbar.update(1)
    if tqdm_bar:
        pbar.close()
    return np.array(sampled_waypoints)


def generate_position_trajectory(
    env: str,
    waypoints: List[RigidTransform],
    tolerance: float = 1e-3,
    verbose: bool = False,
):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    robot = add_robot(env, plant=plant)
    parser = Parser(plant)
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    num_q = plant.num_positions()
    gripper_frame = plant.GetFrameByName(gripper_frame_name(env), robot)

    # distribute waypoint position constraints prop. to distance
    dists = [
        np.linalg.norm(waypoints[i+1].translation() - waypoints[i].translation())
        for i in range(len(waypoints) - 1)
    ]
    cumsum = np.concatenate(([0], np.cumsum(dists)))
    timestamps = cumsum / cumsum[-1]

    # decide number of control points
    num_ctrl_points = 10

    # sketch optimization problem
    trajopt = KinematicTrajectoryOptimization(plant.num_positions(), num_ctrl_points)
    prog = trajopt.get_mutable_prog()
    trajopt.AddDurationCost(1.0)
    trajopt.AddPathLengthCost(1.0)
    trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
    trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())

    # add waypoint position constraints
    for (waypoint, timestamp) in zip(waypoints, timestamps):
        waypoint_constraint = PositionConstraint(
            plant,
            plant.world_frame(),
            waypoint.translation() - tolerance,
            waypoint.translation() + tolerance,
            gripper_frame,
            [0, 0, 0],
            plant_context,
        )
        trajopt.AddPathPositionConstraint(waypoint_constraint, timestamp)

    # start and end with zero velocity
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0)
    trajopt.AddPathVelocityConstraint(np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1)

    # Solve
    result = Solve(prog)
    if not result.is_success():
        if verbose:
            print("Trajectory optimization failed.")
        return None
    trajectory = trajopt.ReconstructTrajectory(result)

    return trajectory


def generate_one_episode(
    env: str,
    num_waypoints: int = 5,
    time_step: float = 0.05,
    verbose: bool = False,
    waypoints_bank: str = None,
):
    # load feasible waypoints bank if provided
    if waypoints_bank is not None:
        feasible_waypoints = np.load(waypoints_bank)

    trajectory = None
    while trajectory is None:

        # sample feasible waypoints
        if waypoints_bank is not None:
            sampled_indices = np.random.choice(len(feasible_waypoints), num_waypoints, replace=False)
            sampled_waypoints = [
                RigidTransform(feasible_waypoints[i]) 
                for i in sampled_indices
            ]
        else:
            sampled_waypoints = [
                RigidTransform(waypoint) 
                for waypoint in sample_feasible_waypoints(env, num_waypoints, verbose=verbose)
            ]

        # generate trajectory
        trajectory = generate_position_trajectory(env, sampled_waypoints, verbose=verbose)

    # discretize trajectory
    positions, _ = discretize_trajectory(trajectory, time_step)
    positions = positions.astype(np.float32)
    positions = positions[:, :, 0]   # (T, 7, 1) -> (T, 7)
    return positions


def unstandardize_franka_action(action: np.ndarray):
    # action: (T, 8)
    T, d = action.shape
    assert d == 8, f"Expect standardized franka action with 8d, got {d}d."
    new_action = np.zeros((T, 9), dtype=action.dtype)
    new_action[:, :7] = action[:, :7]
    finger_trans = np.clip(action[:, -1], 0, 1) * 0.08 / 2 # 8cm max gap, each finger = half the gap
    new_action[:, 7] = new_action[:, 8] = finger_trans
    return new_action
