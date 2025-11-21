import string
import numpy as np
import gymnasium
import hydra
import gc
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.environment import Environment
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.const import RenderMode

from policy_research.env.rlbench.action_modes import (
    AbsoluteJointPositionActionMode, 
    AbsoluteEndEffectorPoseActionMode,
    DeltaJointPositionActionMode,
)

from typing import Optional, List, Literal



class RlbenchEnv(gymnasium.Env):
    metadata = {
        "render_modes": ['rgb_array'],
        "render_fps": 10
    }

    def __init__(self,
        task_name: Optional[str] = None,
        image_size: int = 128,
        seed: Optional[int] = None,
        camera_names: List[str] = ['left_shoulder','right_shoulder','overhead','wrist','front'],
        state_ports: List[str] = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_open', 'gripper_pose', 'gripper_joint_positions',
            'gripper_touch_forces'
        ],
        video_resolution: int = 512,
        max_episode_steps: int = 250,
        control_mode: Literal['ee_pose', 'qpos', 'delta_qpos'] = 'qpos',
    ):
        super().__init__()

        self.camera_names = camera_names
        self.state_ports = state_ports
        self.image_size = image_size
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False

        # observation config
        obs_config = ObservationConfig()
        for name in camera_names:
            obs_config.__dict__[f"{name}_camera"].rgb = True
            obs_config.__dict__[f"{name}_camera"].depth = False
            obs_config.__dict__[f"{name}_camera"].point_cloud = False
            obs_config.__dict__[f"{name}_camera"].image_size = (image_size, image_size)
        for name in state_ports:
            obs_config.__dict__[name] = True

        # coppelia engine setup
        if control_mode == 'ee_pose':
            self.action_mode = AbsoluteEndEffectorPoseActionMode()
        elif control_mode == 'qpos':
            self.action_mode = AbsoluteJointPositionActionMode()
        elif control_mode == 'delta_qpos':
            self.action_mode = DeltaJointPositionActionMode()
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")
        self.rlbench_env = Environment(
            action_mode=self.action_mode,
            obs_config=obs_config,
            headless=True
        )
        self.rlbench_env.launch()

        # task setup
        if task_name is not None:
            self.set_task(task_name, seed=seed)

    
    def _reset_task_vars(self):
        self.task_name = None
        # self.task_prompt = None
        self.task_env = None
        self.recording_camera = None
        self.observation_space = None
        self.action_space = None
        self.cur_step = 0
        self.done = False
        gc.collect()

    
    def set_task(self, task_name: str, seed: Optional[int] = None):
        # clean up if any
        self._reset_task_vars()

        self.task_name = task_name
        self.task_env = self.rlbench_env.get_task(
            hydra.utils.get_class(
                f"rlbench.tasks.{task_name}."
                f"{''.join([word.capitalize() for word in task_name.split('_')])}"
            )
        )
        if seed is not None:
            np.random.seed(seed)

        # video recording camera setup
        dummy_placeholder = Dummy("cam_cinematic_placeholder")
        self.recording_camera = VisionSensor.create(
            [self.video_resolution, self.video_resolution])
        self.recording_camera.set_pose(dummy_placeholder.get_pose())
        self.recording_camera.set_render_mode(RenderMode.OPENGL3)

        description, obs = self.task_env.reset()
        # self.task_prompt = description[0]   # description is a list of prompts

        # setup gym spaces 
        sample_obs_dict = self._extract_obs(obs)
        # sample_obs_dict.pop('prompt', None)  # remove prompt from observation space
        self.observation_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(
                low=0, high=255, 
                shape=value.shape, dtype=value.dtype
            ) if 'rgb' in key else gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=value.shape, dtype=value.dtype
            ) for key, value in sample_obs_dict.items()
        })
        # self.observation_space.spaces['prompt'] = gymnasium.spaces.Text(
        #     min_length=0, max_length=512,
        #     charset=string.printable
        # )
        action_bounds = self.action_mode.action_bounds()
        self.action_space = gymnasium.spaces.Box(
            low=np.float32(action_bounds[0]), high=np.float32(action_bounds[1]),
            shape=self.rlbench_env.action_shape, dtype=np.float32
        )


    def _extract_obs(self, rlbench_obs: Observation):
        obs_dict = {}

        # state
        for port_name in self.state_ports:
            state_data = getattr(rlbench_obs, port_name, None)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                obs_dict[port_name] = state_data

        # rgb
        for name in self.camera_names:
            obs_dict[f"{name}_rgb"] = getattr(rlbench_obs, f"{name}_rgb")

        # # prompt
        # obs_dict['prompt'] = self.task_prompt

        return obs_dict
    

    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        frame = self.recording_camera.capture_rgb()
        frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
        return frame
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        _, obs = self.task_env.reset()
        self.cur_step = 0
        self.done = False
        # return self._extract_obs(obs), {'prompt': self.task_prompt}
        return self._extract_obs(obs), {}
    
    
    def step(self, action: np.ndarray):
        obs, reward, terminated = self.task_env.step(action)
        self.cur_step += 1
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, {}


    def close(self) -> None:
        self.rlbench_env.shutdown()
