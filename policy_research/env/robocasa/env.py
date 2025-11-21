import string
import gymnasium
import numpy as np
from robocasa.utils.env_utils import create_env
from typing import List, Optional, Dict


class RobocasaEnv(gymnasium.Env):
    metadata = {
        "render_modes": ['rgb_array'],
        "render_fps": 20
    }

    def __init__(self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List[str] = [
            # "robot0_robotview",
            # "robot0_frontview",
            # "robot0_agentview_center",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            'robot0_joint_pos', 
            'robot0_gripper_qpos',
        ],
        video_camera: str = "robot0_agentview_right",
        video_resolution: int = 512,
        max_episode_steps: int = 500,
        enable_render: bool = True,
    ):
        super().__init__()

        env = create_env(
            env_name=task_name,
            robots="PandaOmron",
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_widths=image_size,
            camera_heights=image_size,
            seed=seed,
            has_onscreen_renderer=False,
            has_offscreen_renderer=enable_render,
            use_camera_obs=enable_render,
        )
        # env.hard_reset = False # NOTE: we cannot set to False, it's problematic in robocasa

        self.env = env
        self.task_name = task_name
        self.task_prompt = env.get_ep_meta()['lang']
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False

        # setup gym spaces
        obs_dict = env._get_observations()
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0, high=255, 
                shape=(image_size, image_size, 3), dtype=np.uint8
            )
        observation_space.spaces['prompt'] = gymnasium.spaces.Text(
            min_length=0, max_length=512,
            charset=string.printable
        )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env.action_dim,), dtype=np.float32
        )
    
    def _extract_obs(self, 
        raw_obs: Optional[Dict[str, np.ndarray]]=None
    ) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env._get_observations()

        obs_dict = {}

        # robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # rgb
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = np.flip(
                raw_obs[f"{cam_name}_image"], axis=0).astype(np.uint8)
            
        # prompt
        obs_dict['prompt'] = self.task_prompt

        return obs_dict

    def step(self, action: np.ndarray):
        obs, reward, terminated, info = self.env.step(action)
        if self.env._check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = self.done or terminated or (reward >= 1) \
            or (self.env.timestep >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs_dict = self._extract_obs(obs)
        self.done = False
        return obs_dict, {'prompt': self.task_prompt}

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        frame = np.flip(self.env.sim.render(
            height=self.video_resolution, width=self.video_resolution, 
            camera_name=self.video_camera
        ), axis=0).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()

    def seed(self, *args, **kwargs):
        pass
