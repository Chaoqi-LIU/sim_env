import numpy as np
import gym
import robosuite
from robosuite.controllers import load_composite_controller_config

from typing import List


def create_env(
    task_name: str,
    camera_names: List[str] = [
        "agentview",
        "robot0_eye_in_hand",
    ],
    camera_width: int = 128,
    camera_height: int = 128,
    enable_render: bool = True,
    seed: int = 42,
    robot: str = "Panda",
):
    controller_config = load_composite_controller_config(
        controller=None,
        robot=robot,
    )
    env_kwargs = dict(
        env_name=task_name,
        robots=robot,
        controller_configs=controller_config,
        camera_names=camera_names,
        camera_widths=camera_width,
        camera_heights=camera_height,
        has_renderer=False,
        has_offscreen_renderer=enable_render,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=enable_render,
        camera_depths=False,
        seed=seed,
    )
    env = robosuite.make(**env_kwargs)
    return env


class RobomimicEnv(gym.Env):
    metadata = {
        "render.modes": ["rgb_array"], 
        "video.frames_per_second": 20,
    }

    def __init__(self, 
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List[str] = [
            # "frontview",
            # "birdview",
            # "sideview",
            # "robot0_robotview",
            "agentview",
            "robot0_eye_in_hand",
        ],
        state_ports: List[str] = [
            # 'robot0_joint_pos', 
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ],
        video_camera: str = 'agentview',
        video_resolution: int = 512,
        max_episode_steps: int = 800,
        enable_render: bool = True,
    ):
        env = create_env(
            task_name=task_name,
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_width=image_size,
            camera_height=image_size,
            seed=seed,
            enable_render=enable_render,
            robot="Panda",
        )
        env.hard_reset = False
        self.robosuite_is_v1 = (robosuite.__version__.split(".")[0] == "1")

        self.env = env
        self.task_name = task_name
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False
        self.cur_step = 0

        # setup gym spaces
        obs_dict = (
            env._get_observations(force_update=True) 
            if self.robosuite_is_v1 else 
            env._get_observations()
        )
        observation_space = gym.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gym.spaces.Box(
                low=0, high=255,
                shape=(image_size, image_size, 3), dtype=np.uint8
            )
        self.observation_space = observation_space
        action_bounds = env.action_spec
        self.action_space = gym.spaces.Box(
            low=np.float32(action_bounds[0]), high=np.float32(action_bounds[1]),
            shape=action_bounds[0].shape, dtype=np.float32
        )


    def _extract_obs(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = (
                self.env._get_observations(force_update=True) 
                if self.robosuite_is_v1 else 
                self.env._get_observations()
            )

        obs_dict = {}

        # robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # rgb
        for cam_name in self.camera_names:
            obs_dict[f"{cam_name}_rgb"] = np.flip(
                raw_obs[f"{cam_name}_image"], axis=0).astype(np.uint8)

        return obs_dict
    
    def reset(self):
        raw_obs = self.env.reset()
        obs = self._extract_obs(raw_obs)
        self.done = False
        self.cur_step = 0
        return obs
    
    def step(self, action):
        raw_obs, reward, terminated, info = self.env.step(action)
        self.cur_step += 1
        if self.env._check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extract_obs(raw_obs), reward, self.done, info
    
    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        frame = np.flip(self.env.sim.render(
            height=self.video_resolution, width=self.video_resolution, 
            camera_name=self.video_camera
        ), axis=0).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
