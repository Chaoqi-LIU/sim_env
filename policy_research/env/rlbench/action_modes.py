import numpy as np
from rlbench.backend.scene import Scene
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import (
    JointPosition, ArmActionMode, RelativeFrame,
    calculate_delta_pose, assert_unit_quaternion
)
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode, assert_action_shape)


def to_unit_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Zero norm quaternion cannot be normalized.")
    return q / norm


class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: RelativeFrame = RelativeFrame.WORLD,
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        action[3:] = to_unit_quaternion(action[3:])
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,


class GripperJointPosition(GripperActionMode):

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(scene.robot))
        a = action if self._absolute_mode else (
            action + scene.robot.gripper.get_joint_positions())
        scene.robot.gripper.set_joint_target_positions(a)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.gripper.set_joint_target_positions(
            scene.robot.gripper.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return 2,



class AbsoluteJointPositionActionMode(ActionMode):
    """Abs joint control for both arm and gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(AbsoluteJointPositionActionMode, self).__init__(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=GripperJointPosition(absolute_mode=True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))
    
    def action_bounds(self):
        return np.array(7 * [-0.1] + 2 * [0.0]), np.array(7 * [0.1] + 2 * [0.04]) 
    

class DeltaJointPositionActionMode(ActionMode):
    """Abs joint control for both arm and gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(DeltaJointPositionActionMode, self).__init__(
            arm_action_mode=JointPosition(absolute_mode=False),
            gripper_action_mode=GripperJointPosition(absolute_mode=False)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))
    
    def action_bounds(self):
        return np.array(9 * [-1.]), np.array(9 * [1.])


class AbsoluteEndEffectorPoseActionMode(ActionMode):
    """Abs end-effector pose control for arm and abs joint position control of gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(AbsoluteEndEffectorPoseActionMode, self).__init__(
            arm_action_mode=EndEffectorPoseViaIK(absolute_mode=True),
            # arm_action_mode=EndEffectorPoseViaPlanning(absolute_mode=True),
            gripper_action_mode=GripperJointPosition(absolute_mode=True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        try:
            self.arm_action_mode.action(scene, arm_action)
            self.gripper_action_mode.action(scene, ee_action)
        except Exception as e:
            print(f"Action failed: {e}")

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))
    
    def action_bounds(self):
        return np.array(7 * [-1.] + 2 * [0.0]), np.array(7 * [1.] + 2 * [0.04])
